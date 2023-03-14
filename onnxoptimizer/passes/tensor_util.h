/*
 * SPDX-License-Identifier: Apache-2.0
 */

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include <vector>

#include "onnx/common/tensor.h"
#include "onnxoptimizer/passes/logging.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct Complex64 {
  using base_type = float;
  Complex64(float r, float i) : real_part(r), imaginary_part(i) {}

  bool operator==(const Complex64& rhs) const {
    return real_part == rhs.real_part && imaginary_part == rhs.imaginary_part;
  }

  bool operator!=(const Complex64& rhs) const {
    return !(*this == rhs);
  }

  float real_part;
  float imaginary_part;
};

struct Complex128 {
  using base_type = double;
  Complex128(double r, double i) : real_part(r), imaginary_part(i) {}

  bool operator==(const Complex128& rhs) const {
    return real_part == rhs.real_part && imaginary_part == rhs.imaginary_part;
  }
  bool operator!=(const Complex128& rhs) const {
    return !(*this == rhs);
  }
  double real_part;
  double imaginary_part;
};

struct Float16 {
    Float16()=default;
  // bit-wise convert
  Float16(int32_t v) : bits(static_cast<uint32_t>(v)) {}
  Float16(uint16_t v) : bits(v) {}
  bool operator==(const Float16& rhs) const {
    return bits == rhs.bits;
  }
  bool operator!=(const Float16& rhs) const {
    return !(*this == rhs);
  }
  uint16_t bits;
};

struct BFloat16 {
    BFloat16()=default;
  // bit-wise convert
  BFloat16(int32_t v) : bits(static_cast<uint32_t>(v)) {}
  BFloat16(uint16_t v) : bits(v) {}
  bool operator==(const BFloat16& rhs) const {
    return bits == rhs.bits;
  }
  bool operator!=(const BFloat16& rhs) const {
    return !(*this == rhs);
  }
  uint16_t bits;
};

template <typename Complex>
std::size_t ComplexHashHelper(const Complex& complex) {
  auto hasher = std::hash<typename Complex::base_type>();
  auto r = hasher(complex.real_part);
  auto i = hasher(complex.imaginary_part);
  return r ^= i + 0x9e3779b9 + (r << 6) + (r >> 2);
}

int64_t ElemCntOfTensor(const Tensor* tensor);
int64_t ElemCntOfTensor(const Tensor& tensor);

template <typename T>
const std::vector<T> ParseTensorData(const Tensor* tensor);

}  // namespace optimization
}  // namespace ONNX_NAMESPACE

namespace std {

#define DEFINE_COMPLEX_HASH(type)                                            \
  template <>                                                                \
  struct hash<type> {                                                        \
    std::size_t operator()(const type& complex) const {                      \
      return ONNX_NAMESPACE::optimization::ComplexHashHelper<type>(complex); \
    }                                                                        \
  };

DEFINE_COMPLEX_HASH(ONNX_NAMESPACE::optimization::Complex128)
DEFINE_COMPLEX_HASH(ONNX_NAMESPACE::optimization::Complex64)
#undef DEFINE_COMPLEX_HASH

template <>
struct hash<ONNX_NAMESPACE::optimization::Float16> {
  std::size_t operator()(const ONNX_NAMESPACE::optimization::Float16& v) const {
    return hash<uint16_t>{}(v.bits);
  };
};

template <>
struct hash<ONNX_NAMESPACE::optimization::BFloat16> {
  std::size_t operator()(
      const ONNX_NAMESPACE::optimization::BFloat16& v) const {
    return hash<uint16_t>{}(v.bits);
  };
};

}  // namespace std
