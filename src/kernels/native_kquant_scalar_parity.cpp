// No GPU or model required: Strata's scalar K-quant dequantizers against pinned ggml on raw blocks.
#include "strata/artifact/dequant.hpp"
#include "native_kquant_fixture.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

int main() {
    int failures = 0;
    for (int type : {10, 11, 14}) {
        constexpr int blocks = 1024;
        const auto raw = native_kquant_fixture(type, blocks);
        std::vector<float> want(blocks * 256), got(want.size());
        ggml_get_type_traits((ggml_type) type)->to_float(raw.data(), want.data(), want.size());
        const size_t stride = ggml_type_size((ggml_type) type);
        for (int b = 0; b < blocks; ++b) {
            const auto dequant = type == 10 ? strata::dequantize_q2_K :
                                 type == 11 ? strata::dequantize_q3_K : strata::dequantize_q6_K;
            dequant(raw.data() + b * stride, got.data() + b * 256);
        }
        double err = 0, norm = 0;
        for (size_t i = 0; i < want.size(); ++i) {
            if (!std::isfinite(got[i])) ++failures;
            err += std::fabs((double) got[i] - want[i]);
            norm += std::fabs((double) want[i]);
        }
        const double rel = err / (norm + 1e-30);
        std::printf("%s scalar vs ggml: %d raw blocks, relative error %.3g\n",
                    ggml_type_name((ggml_type) type), blocks, rel);
        if (!(rel < 1e-7)) ++failures;
    }
    return failures ? 1 : 0;
}
