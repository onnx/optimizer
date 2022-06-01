/*
 * SPDX-License-Identifier: Apache-2.0
 */

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include "onnx/common/tensor.h"
#include "onnxoptimizer/pass.h"

namespace ONNX_NAMESPACE {
namespace optimization {

inline int AddYIfNegative(int x, int y) {
  if (x < 0) {
    return x + y;
  }
  return x;
};

struct EliminateShapeOp final : public PredicateBasedPass {
  explicit EliminateShapeOp()
      : PredicateBasedPass(PassType::Fuse, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "eliminate_shape_op";
  }

  bool patternMatchPredicate(Node *node) override {
    if (node->kind() != Symbol("Shape")) {
      return false;
    }
    const Value *input = node->input();
    if (!input->has_sizes()) {
      return false;
    }
    const int start = AddYIfNegative(
        node->hasAttribute(Symbol("start")) ? node->i(Symbol("start")) : 0,
        input->sizes().size());
    const int end = AddYIfNegative(node->hasAttribute(Symbol("end"))
                                       ? node->i(Symbol("end"))
                                       : input->sizes().size(),
                                   input->sizes().size());
    return std::all_of(input->sizes().cbegin() + start,
                       input->sizes().cbegin() + end,
                       [](const auto &dim) { return dim.is_int; });
  }

  bool runTransform(Node *node, Graph &graph,
                    NodeDestroyType &destroy_current) override {
    // TODO: add implicit conversion from const char* to Symbol
    const Value *input = node->input();
    const int start = AddYIfNegative(
        node->hasAttribute(Symbol("start")) ? node->i(Symbol("start")) : 0,
        input->sizes().size());
    const int end = AddYIfNegative(node->hasAttribute(Symbol("end"))
                                       ? node->i(Symbol("end"))
                                       : input->sizes().size(),
                                   input->sizes().size());
    Tensor tensor;
    tensor.sizes().push_back(end - start);
    tensor.elem_type() = ONNX_NAMESPACE::TensorProto_DataType_INT64;
    std::transform(input->sizes().begin() + start, input->sizes().begin() + end,
                   std::back_inserter(tensor.int64s()),
                   [](const auto &dim) { return dim.dim; });

    Value *value = graph.addInitializerAndCreateValue(tensor);

    const bool replacing_success =
        tryReplacingAllUsesWith(node->output(), value);
    if (!replacing_success) {
      return false;
    }
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

}  // namespace optimization
}  // namespace ONNX_NAMESPACE
