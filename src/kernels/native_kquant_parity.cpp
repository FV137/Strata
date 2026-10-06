// GPU K-quant parity on raw GGUF bytes; no model download. Unlike expert-level comparisons, row dots
// use the very same Q8_1 codes in the double reference, so activation quantization is not a tolerance.
#include "strata/kernels/dequant_bf16.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "native_kquant_fixture.hpp"
#include "ggml-cpu.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace k = strata::kernels;

static void gpu(cudaError_t e) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s\n", cudaGetErrorString(e)); std::exit(1); }
}

static double relative(const std::vector<float>& got, const std::vector<float>& want) {
    double err = 0, norm = 0;
    for (size_t i = 0; i < want.size(); ++i) {
        if (!std::isfinite(got[i])) return INFINITY;
        err += std::fabs((double) got[i] - want[i]);
        norm += std::fabs((double) want[i]);
    }
    return err / (norm + 1e-30);
}

int main() {
    ggml_cpu_init();
    cudaStream_t s;
    gpu(cudaStreamCreate(&s));
    int failures = 0;
    constexpr int rows = 9, columns = 8;
    struct Q81 { ggml_fp16_t d, s; int8_t qs[32]; };
    static_assert(sizeof(Q81) == 36);
    for (int type : {2, 10, 11, 14}) {
        if (!k::native_expert_supported(type, 8, 2560, 640) ||
            !k::native_expert_supported(type, 6, 2560, 640) ||
            (type != 2 && k::native_expert_supported(type, type, 2560, 640)) ||
            !k::native_mmvq_supported(type) || !k::dequant_bf16_supported(type)) {
            std::fprintf(stderr, "type %d: incorrect capability/geometry guard\n", type);
            ++failures;
        }
        for (int n : {256, 2560}) {
            const auto raw = native_kquant_fixture(type, rows * n / (type == 2 ? 32 : 256));
            if (k::iq_row_bytes(type, n) * rows != raw.size() ||
                k::native_mmvq_weight_bytes(type, n, rows) != raw.size()) ++failures;
            std::vector<float> weights(rows * n), got(weights.size());
            ggml_get_type_traits((ggml_type) type)->to_float(raw.data(), weights.data(), weights.size());
            std::vector<Q81> x(columns * n / 32);
            for (size_t b = 0; b < x.size(); ++b) {
                x[b].d = ggml_fp32_to_fp16(type == 2 ? 0.00390625f : 0.0013f * (1 + b % 7));
                // Deliberately unrelated to sum(q): Q2_K's min term must use the quantized sum.
                x[b].s = ggml_fp32_to_fp16(17.f);
                for (int j = 0; j < 32; ++j) x[b].qs[j] = (int8_t) ((int) ((37 * b + 19 * j) % 255) - 127);
                if (type == 2) {   // exact stored sum for Q4_0's fixed -8 offset
                    int sum = 0;
                    for (int j = 0; j < 32; ++j) sum += x[b].qs[j];
                    x[b].s = ggml_fp32_to_fp16(ggml_fp16_to_fp32(x[b].d) * sum);
                }
            }
            void *dw, *dx;
            float* dy;
            uint16_t* dh;
            gpu(cudaMalloc(&dw, raw.size()));
            gpu(cudaMalloc(&dx, x.size() * sizeof(Q81)));
            gpu(cudaMalloc((void**) &dy, weights.size() * 4));
            gpu(cudaMalloc((void**) &dh, 2 * (rows - 1) * n * 2));
            gpu(cudaMemcpy(dw, raw.data(), raw.size(), cudaMemcpyHostToDevice));
            gpu(cudaMemcpy(dx, x.data(), x.size() * sizeof(Q81), cudaMemcpyHostToDevice));

            k::iq_dequant_f32(type, dw, weights.size(), dy, s);
            gpu(cudaStreamSynchronize(s));
            gpu(cudaMemcpy(got.data(), dy, got.size() * 4, cudaMemcpyDeviceToHost));
            if (!(relative(got, weights) < 1e-7)) ++failures;
            // A slice starting after the first row verifies row offsets and two-byte block alignment.
            std::vector<float> slice(weights.begin() + n, weights.end()), got_slice(slice.size());
            k::dequant_f32(type, dw, 1, rows - 1, n, dy, s);
            gpu(cudaStreamSynchronize(s));
            gpu(cudaMemcpy(got_slice.data(), dy, slice.size() * 4, cudaMemcpyDeviceToHost));
            if (!(relative(got_slice, slice) < 1e-7)) ++failures;
            std::vector<uint16_t> h(slice.size());
            for (bool bf16 : {false, true}) {
                if (bf16) k::dequant_bf16(type, dw, 1, rows - 1, n, dh, s);
                else k::dequant_f16(type, dw, 1, rows - 1, n, dh, s);
                gpu(cudaStreamSynchronize(s));
                gpu(cudaMemcpy(h.data(), dh, h.size() * 2, cudaMemcpyDeviceToHost));
                for (size_t i = 0; i < h.size(); ++i) {
                    const uint16_t want = bf16 ? ggml_fp32_to_bf16(slice[i]).bits : ggml_fp32_to_fp16(slice[i]);
                    if (h[i] != want) ++failures;
                }
            }
            // Prompt gate/up layout: adjacent input rows become the two interleaved output rows.
            k::iq_dequant_gu_f16(type, dw, (uint8_t*) dw + k::iq_row_bytes(type, n), rows - 1, n, dh, s);
            gpu(cudaStreamSynchronize(s));
            h.resize(2 * (rows - 1) * n);
            gpu(cudaMemcpy(h.data(), dh, h.size() * 2, cudaMemcpyDeviceToHost));
            for (int r = 0; r < rows - 1; ++r)
                for (int p = 0; p < 2; ++p)
                    for (int i = 0; i < n; ++i)
                        if (h[(2 * r + p) * n + i] != ggml_fp32_to_fp16(weights[(r + p) * n + i])) ++failures;

            double max_error = 0;
            for (int nc : {1, 3, 8}) {
                std::vector<float> want(nc * rows), y(want.size());
                for (int c = 0; c < nc; ++c)
                    for (int r = 0; r < rows; ++r) {
                        double sum = 0;
                        for (int i = 0; i < n; ++i) {
                            const Q81& a = x[c * (n / 32) + i / 32];
                            sum += (double) weights[r * n + i] * ggml_fp16_to_fp32(a.d) * a.qs[i % 32];
                        }
                        want[c * rows + r] = (float) sum;
                    }
                // Expert and dense APIs have separate dispatch lists; exercise both on the same blocks.
                for (bool dense : {false, true}) {
                    if (dense) k::native_mmvq(type, dw, dx, dy, n, rows, nc, s);
                    else k::iq_mmvq(type, dw, dx, dy, n, rows, nc, s);
                    gpu(cudaStreamSynchronize(s));
                    gpu(cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost));
                    const double e = relative(y, want);
                    max_error = (std::max)(max_error, e);
                    if (!(e < 2e-6)) ++failures;
                }
            }
            std::printf("%s, n=%d: dequant/FP16/BF16/interleave, 1/3/8-column dots rel %.3g; failures %d\n",
                        ggml_type_name((ggml_type) type), n, max_error, failures);
            gpu(cudaFree(dw)); gpu(cudaFree(dx)); gpu(cudaFree(dy)); gpu(cudaFree(dh));
        }
    }
    gpu(cudaStreamDestroy(s));
    return failures ? 1 : 0;
}
