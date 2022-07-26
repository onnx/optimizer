/*
 * SPDX-License-Identifier: Apache-2.0
 */

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#include "pass_util.h"

#include "onnx/defs/tensor_util.h"

namespace ONNX_NAMESPACE {
namespace optimization {

#define FetchSoleValueOfTensor_Template(pb_type, cpp_type)               \
  template <>                                                            \
  bool FetchSoleValueOfTensor<cpp_type>(const Value* t, cpp_type& val) { \
    const Tensor* tensor = FetchConstantTensor(t);                       \
    if (!tensor || tensor->elem_type() !=                                \
                       ONNX_NAMESPACE::TensorProto_DataType_##pb_type) { \
      return false;                                                      \
    }                                                                    \
    const auto data = ParseData<cpp_type>(tensor);                       \
    if (data.size() != 1) {                                              \
      return false;                                                      \
    }                                                                    \
    val = data[0];                                                       \
    return true;                                                         \
  }

FetchSoleValueOfTensor_Template(INT32, int32_t)
FetchSoleValueOfTensor_Template(INT64, int64_t)
FetchSoleValueOfTensor_Template(FLOAT, float)
FetchSoleValueOfTensor_Template(DOUBLE, double)

bool FetchSoleIntValueOfTensor(const Value* t, int64_t& val) {
  int32_t i32_val;
  const bool r1 = FetchSoleValueOfTensor<int64_t>(t, val);
  const bool r2 = FetchSoleValueOfTensor<int32_t>(t, i32_val);
  if (r2) {
    val = i32_val;
  }
  return r1 || r2;
}

bool FetchSoleIntValueOfAttr(const Node* node, Symbol attr_name, int64_t& val) {
  if (node->kindOf(attr_name) == AttributeKind::is) {
    const auto attrs = node->is(attr_name);
    if (attrs.size() != 1) {
      return false;
    }
    val = attrs[0];
    return true;
  } else if (node->kindOf(attr_name) == AttributeKind::i) {
    val = node->i(attr_name);
    return true;
  } else {
    return false;
  }
};

}  // namespace optimization
}  // namespace ONNX_NAMESPACE
