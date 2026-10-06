// Deterministic raw GGUF blocks for native K-quant parity, including every packed bit and signed scale.
#pragma once

#include "ggml.h"

#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

inline std::vector<uint8_t> native_kquant_fixture(int type, int blocks) {
    const size_t stride = ggml_type_size((ggml_type) type);
    std::vector<uint8_t> raw(stride * blocks);
    std::mt19937 rng(319 + type);
    for (auto& b : raw) b = (uint8_t) rng();
    // Zero, negative, small and large finite scales. Other bytes stay random, exercising min nibbles,
    // Q3_K's packed signed six-bit scales and the full int8 range of Q6_K's subblock scales.
    const float scales[] = {0.f, 0.00013f, -0.007f, 0.03125f, 0.19f, -1.3f};
    for (int b = 0; b < blocks; ++b) {
        const ggml_fp16_t d = ggml_fp32_to_fp16(scales[b % 6]);
        std::memcpy(raw.data() + b * stride + (type == 2 ? 0 : type == 10 ? 80 : type == 11 ? 108 : 208), &d, 2);
        if (type == 10) {
            const ggml_fp16_t m = ggml_fp32_to_fp16(scales[(b + 3) % 6]);
            std::memcpy(raw.data() + b * stride + 82, &m, 2);
        }
    }
    return raw;
}
