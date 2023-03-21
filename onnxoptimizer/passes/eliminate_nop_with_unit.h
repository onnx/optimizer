/*
 * SPDX-License-Identifier: Apache-2.0
 */

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include "onnx/defs/tensor_util.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct EliminateOpWithUnit final : public PredicateBasedPass {
  explicit EliminateOpWithUnit()
      : PredicateBasedPass(PassType::Nop, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "eliminate_nop_with_unit";
  }

#define PROTO_DTYPE_LIST(_)        \
  _(TensorProto_DataType_BFLOAT16) \
  _(TensorProto_DataType_FLOAT16)  \
  _(TensorProto_DataType_FLOAT)    \
  _(TensorProto_DataType_DOUBLE)   \
  _(TensorProto_DataType_UINT8)    \
  _(TensorProto_DataType_INT8)     \
  _(TensorProto_DataType_UINT16)   \
  _(TensorProto_DataType_INT16)    \
  _(TensorProto_DataType_UINT32)   \
  _(TensorProto_DataType_INT32)    \
  _(TensorProto_DataType_UINT64)   \
  _(TensorProto_DataType_INT64)    \
  _(TensorProto_DataType_BOOL)

  bool isAllOf(const Tensor& tensor, int value) {
    int elem_type = tensor.elem_type();
#define CASE_BRANCH_CONTENT(pb_dtype)                                        \
  case pb_dtype: {                                                           \
    using cpp_dtype = ToCppType<pb_dtype>::type;                             \
    const std::vector<cpp_dtype> data = ParseTensorData<cpp_dtype>(&tensor); \
    return std::all_of(                                                      \
        data.cbegin(), data.cend(),                                          \
        [value](const cpp_dtype& v) { return v == cpp_dtype(value); });      \
  }

    switch (elem_type) { PROTO_DTYPE_LIST(CASE_BRANCH_CONTENT) }
#undef CASE_BRANCH_CONTENT
#undef PROTO_DTYPE_LIST
    return false;
  }

  bool isAllOne(const Tensor& tensor) {
    if (tensor.elem_type() == TensorProto_DataType_FLOAT16) {
      return isAllOf(tensor, 0x3c00);
    }
    if (tensor.elem_type() == TensorProto_DataType_BFLOAT16) {
      return isAllOf(tensor, BFloat16(1.f));
    }
    return isAllOf(tensor, 1);
  }

  bool patternMatchPredicate(Node* node) override {
    return true;
  };

  bool isUnit(const Tensor& tensor, NodeKind kind, int index) {
    if (kind == Symbol("And") || kind == kMul) {
      return isAllOne(tensor);
    }
    if (kind == Symbol("Or") || kind == kAdd) {
      return isAllOf(tensor, 0);
    }
    if (kind == kSub) {
      return index == 1 && isAllOf(tensor, 0);
    }
    if (kind == kDiv || kind == kPow) {
      return index == 1 && isAllOne(tensor);
    }
    if (kind == kConcat) {
      return ElemCntOfTensor(tensor) == 0;
    }
    return false;
  }
  bool isBroadcastBinaryOp(NodeKind kind) {
    return kind == kAdd || kind == kMul || kind == kDiv || kind == kSub ||
           kind == kPow || kind == Symbol("And") || kind == Symbol("Or");
  }

  bool runTransform(Node* node, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    for (int i = 0; i < node->inputs().size(); i++) {
      auto* input = node->inputs()[i];
      if (auto* tensor = FetchConstantTensor(input)) {
        NodeKind kind = node->kind();
        if (isUnit(*tensor, kind, i)) {
          if (isBroadcastBinaryOp(kind)) {
            // replace the node with the other input
            auto* other_input = node->inputs()[1 - i];
            if (isABroadcastToB(tensor->sizes(), other_input->sizes())) {
              return tryReplacingAllUsesWith(node->output(), other_input);
            }
          }
          if (kind == kConcat) {
            node->removeInput(i);
            return true;
          }
        }
      }
    }

    return false;
  }
};

}  // namespace optimization
}  // namespace ONNX_NAMESPACE
