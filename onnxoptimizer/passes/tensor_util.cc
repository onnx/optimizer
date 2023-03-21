/*
 * SPDX-License-Identifier: Apache-2.0
 */

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#include <algorithm>

#include "onnx/common/platform_helpers.h"
#include "onnxoptimizer/passes/tensor_util.h"

namespace ONNX_NAMESPACE {
namespace optimization {

int64_t ElemCntOfTensor(const Tensor* tensor) {
  ONNX_ASSERT(tensor != nullptr);
  const auto& sizes = tensor->sizes();
  return std::accumulate(sizes.begin(), sizes.end(), (int64_t)1,
                         std::multiplies<int64_t>{});
}

int64_t ElemCntOfTensor(const Tensor& tensor) {
  return ElemCntOfTensor(&tensor);
}

/// reference onnx/defs/tensor_util.cc

#define DEFINE_PARSE_TENSOR_DATA(type, typed_data_fetch)                   \
  template <>                                                              \
  const std::vector<type> ParseTensorData(const Tensor* tensor) {          \
    ONNX_ASSERT(tensor != nullptr);                                        \
    std::vector<type> res;                                                 \
    if (!tensor->is_raw_data()) {                                          \
      const auto& data = tensor->typed_data_fetch();                       \
      res.insert(res.end(), data.begin(), data.end());                     \
      return res;                                                          \
    }                                                                      \
    /* make copy as we may have to reverse bytes */                        \
    std::string raw_data = tensor->raw();                                  \
    /* okay to remove const qualifier as we have already made a copy */    \
    char* bytes = const_cast<char*>(raw_data.c_str());                     \
    /*onnx is little endian serialized always-tweak byte order if needed*/ \
    if (!is_processor_little_endian()) {                                   \
      const size_t element_size = sizeof(type);                            \
      const size_t num_elements = raw_data.size() / element_size;          \
      for (size_t i = 0; i < num_elements; ++i) {                          \
        char* start_byte = bytes + i * element_size;                       \
        char* end_byte = start_byte + element_size - 1;                    \
        /* keep swapping */                                                \
        for (size_t count = 0; count < element_size / 2; ++count) {        \
          char temp = *start_byte;                                         \
          *start_byte = *end_byte;                                         \
          *end_byte = temp;                                                \
          ++start_byte;                                                    \
          --end_byte;                                                      \
        }                                                                  \
      }                                                                    \
    }                                                                      \
    /* raw_data.c_str()/bytes is a byte array and may not be properly  */  \
    /* aligned for the underlying type */                                  \
    /* We need to copy the raw_data.c_str()/bytes as byte instead of  */   \
    /* copying as the underlying type, otherwise we may hit memory   */    \
    /* misalignment issues on certain platforms, such as arm32-v7a */      \
    const size_t raw_data_size = raw_data.size();                          \
    const size_t elem_cnt = static_cast<size_t>(ElemCntOfTensor(tensor));  \
    ONNX_ASSERT(elem_cnt == (raw_data_size / sizeof(type)));               \
    res.resize(elem_cnt);                                                  \
    memcpy(reinterpret_cast<char*>(res.data()), bytes, raw_data_size);     \
    return res;                                                            \
  }

DEFINE_PARSE_TENSOR_DATA(int32_t, int32s)
DEFINE_PARSE_TENSOR_DATA(int64_t, int64s)
DEFINE_PARSE_TENSOR_DATA(float, floats)
DEFINE_PARSE_TENSOR_DATA(double, doubles)
DEFINE_PARSE_TENSOR_DATA(uint64_t, uint64s)
DEFINE_PARSE_TENSOR_DATA(uint8_t, int32s)
DEFINE_PARSE_TENSOR_DATA(int8_t, int32s)
DEFINE_PARSE_TENSOR_DATA(uint16_t, int32s)
DEFINE_PARSE_TENSOR_DATA(int16_t, int32s)
DEFINE_PARSE_TENSOR_DATA(uint32_t, uint64s)
DEFINE_PARSE_TENSOR_DATA(Float16, int32s)
DEFINE_PARSE_TENSOR_DATA(BFloat16, int32s)

#undef DEFINE_PARSE_TENSOR_DATA

template <>
const std::vector<bool> ParseTensorData<bool>(const Tensor* tensor) {
  std::vector<bool> res;
  if (!tensor->is_raw_data()) {
    std::transform(tensor->int32s().cbegin(), tensor->int32s().cend(),
                   std::back_inserter(res),
                   [](int32_t d) -> bool { return static_cast<bool>(d); });
    return res;
  }
  const auto& raw_data = tensor->raw();
  const size_t elem_cnt = static_cast<size_t>(ElemCntOfTensor(tensor));
  ONNX_ASSERT(elem_cnt == raw_data.size());
  res.reserve(elem_cnt);
  std::transform(raw_data.cbegin(), raw_data.cend(), std::back_inserter(res),
                 [](char d) { return static_cast<bool>(d); });
  return res;
}

namespace {
template <typename Complex>
const std::vector<Complex> FlattenToComplex(
    const std::vector<typename Complex::base_type>& flatten) {
  std::size_t elem_cnt = flatten.size() / 2;
  ONNX_ASSERT(flatten.size() % 2 == 0);
  std::vector<Complex> res;
  res.reserve(elem_cnt);
  for (std::size_t i = 0; i < elem_cnt; ++i) {
    const auto& real = flatten[i * 2];
    const auto& imaginary = flatten[i * 2 + 1];
    res.emplace_back(Complex{real, imaginary});
  }
  return res;
}
}  // namespace

#define DEFINE_PARSE_TENSOR_DATA_WITH_COMPLEX(type, typed_data_fetch)      \
  template <>                                                              \
  const std::vector<type> ParseTensorData<type>(const Tensor* tensor) {    \
    ONNX_ASSERT(tensor != nullptr);                                        \
    if (!tensor->is_raw_data()) {                                          \
      return FlattenToComplex<type>(tensor->typed_data_fetch());           \
    }                                                                      \
    /* make copy as we may have to reverse bytes */                        \
    std::string raw_data = tensor->raw();                                  \
    /* okay to remove const qualifier as we have already made a copy */    \
    char* bytes = const_cast<char*>(raw_data.c_str());                     \
    /*onnx is little endian serialized always-tweak byte order if needed*/ \
    if (!is_processor_little_endian()) {                                   \
      const size_t element_size = sizeof(typename type::base_type);        \
      const size_t num_elements = raw_data.size() / element_size;          \
      for (size_t i = 0; i < num_elements; ++i) {                          \
        char* start_byte = bytes + i * element_size;                       \
        char* end_byte = start_byte + element_size - 1;                    \
        /* keep swapping */                                                \
        for (size_t count = 0; count < element_size / 2; ++count) {        \
          char temp = *start_byte;                                         \
          *start_byte = *end_byte;                                         \
          *end_byte = temp;                                                \
          ++start_byte;                                                    \
          --end_byte;                                                      \
        }                                                                  \
      }                                                                    \
    }                                                                      \
    /* raw_data.c_str()/bytes is a byte array and may not be properly  */  \
    /* aligned for the underlying type */                                  \
    /* We need to copy the raw_data.c_str()/bytes as byte instead of  */   \
    /* copying as the underlying type, otherwise we may hit memory   */    \
    /* misalignment issues on certain platforms, such as arm32-v7a */      \
    const size_t raw_data_size = raw_data.size();                          \
    const size_t elem_cnt = static_cast<size_t>(ElemCntOfTensor(tensor));  \
    ONNX_ASSERT(elem_cnt == (raw_data_size / sizeof(type)));               \
    std::vector<typename type::base_type> flatten;                         \
    flatten.resize(elem_cnt * 2);                                          \
    memcpy(reinterpret_cast<char*>(flatten.data()), bytes, raw_data_size); \
    return FlattenToComplex<type>(flatten);                                \
  }

DEFINE_PARSE_TENSOR_DATA_WITH_COMPLEX(Complex64, floats)
DEFINE_PARSE_TENSOR_DATA_WITH_COMPLEX(Complex128, doubles)

#undef DEFINE_PARSE_TENSOR_DATA_WITH_COMPLEX

template <>
const std::vector<std::string> ParseTensorData<std::string>(
    const Tensor* tensor) {
  ONNX_ASSERT(tensor != nullptr);
  ONNX_ASSERTM(!tensor->is_raw_data(),
               "data type is string. string content is required to be stored "
               "in repeated bytes string_data field."
               "raw_data type cannot be string.");
  return tensor->strings();
}

}  // namespace optimization
}  // namespace ONNX_NAMESPACE