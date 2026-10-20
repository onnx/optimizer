// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include <algorithm>
#include <unordered_set>

#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct EliminateConsecutiveIdempotentOps final : public PredicateBasedPass {
  explicit EliminateConsecutiveIdempotentOps()
      : PredicateBasedPass(PassType::Fuse, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "eliminate_consecutive_idempotent_ops";
  }

  bool patternMatchPredicate(Node* node) override {
    static const std::unordered_set<std::string> idempotent_ops = {
        "Ceil", "Floor", "Round", "Relu", "Reshape", "Sign"};
    for (const auto& op : idempotent_ops) {
      // TODO: support uses().size() > 1 for ops except Reshape
      if (CheckKind(node, Symbol(op), 0, Symbol(op)) &&
          node->input(0)->uses().size() == 1) {
        return true;
      }
    }
    return false;
  }

  static bool reshapeFusionSafe(Node* node) {
    if (node->hasAttribute(Symbol("allowzero")) &&
        node->i(Symbol("allowzero")) == 1) {
      return true;
    }
    const Tensor* new_shape_tensor = FetchConstantTensor(node->input(1));
    if (!new_shape_tensor || new_shape_tensor->elem_type() !=
                                 ONNX_NAMESPACE::TensorProto_DataType_INT64) {
      return false;
    }
    const auto new_shape = ParseTensorData<int64_t>(new_shape_tensor);
    return std::none_of(new_shape.begin(), new_shape.end(),
                        [](int64_t dim) { return dim == 0; });
  }

  bool runTransform(Node* node, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    if (node->kind() == kReshape && !reshapeFusionSafe(node)) {
      return false;
    }
    Node* previous_node = node->input(0)->node();
    std::vector<Dimension> sizes = previous_node->input(0)->sizes();
    bool replacing_success =
        tryReplacingAllUsesWith(node->input(0), previous_node->input(0));
    if (replacing_success) {
      if (node->kind() == kReshape) {
        // restore the correct sizes
        previous_node->input(0)->setSizes(sizes);
      }
      return true;
    }
    return false;
  }
};

}  // namespace optimization
}  // namespace ONNX_NAMESPACE
