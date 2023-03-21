/*
 * SPDX-License-Identifier: Apache-2.0
 */

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once
#include <cstdint>
#include <cstdlib>

namespace ONNX_NAMESPACE {
namespace optimization {

inline float FP32FromBits(uint32_t bits) {
  union {
    uint32_t as_bits;
    float as_value;
  } fp32{bits};
  return fp32.as_value;
}

inline uint32_t FP32ToBits(float value) {
  union {
    float as_value;
    uint32_t as_bits;
  } fp32{value};
  return fp32.as_bits;
}

}  // namespace optimization
}  // namespace ONNX_NAMESPACE
