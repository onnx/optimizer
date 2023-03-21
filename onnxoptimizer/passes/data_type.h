/*
 * SPDX-License-Identifier: Apache-2.0
 */

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include "onnxoptimizer/passes/bitscast.h"
#include "onnxoptimizer/passes/logging.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct Complex64 {
  using base_type = float;
  Complex64() : real_part(0.f), imaginary_part(0.f) {}
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
  Complex128() : real_part(0.0), imaginary_part(0.0) {}
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

/// The IEEE 754 specifies a half-precision as having format: 1 bit for sign, 5
/// bits for the exponet and 11 bits for the mantissa.

struct Float16 {
  Float16() : bits(0) {}
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

/// Bfloat16 representation uses  1 bit for the sign, 8 bits for the exponent
/// and 7 bits for the mantissa. It is assumed that floats are in IEEE 754
/// format so the representation is just bits 16-31 of a single precision float.
struct BFloat16 {
  BFloat16() : bits(0) {}
  // bit-wise convert
  BFloat16(int32_t v) : bits(static_cast<uint32_t>(v)) {}
  BFloat16(uint16_t v) : bits(v) {}
  BFloat16(float v) : bits(static_cast<uint16_t>(FP32ToBits(v) >> 16)) {}

  /// to float
  operator float() {
    return FP32FromBits(static_cast<uint32_t>(bits) << 16);
  }
  /// from float
  BFloat16& operator=(float val) {
    bits = static_cast<uint16_t>(FP32ToBits(val) >> 16);
    return *this;
  }

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
