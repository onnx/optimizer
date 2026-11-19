// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include <algorithm>

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

  // Reshape(Reshape(X, s1), s2) equals Reshape(X, s2) only if no `0` entry
  // of s2 copies a dimension from the input. With the default allowzero=0,
  // a `0` copies the dimension from the intermediate tensor, which differs
  // from the dimension of X once the first Reshape is bypassed.
  static bool IsReshapeFusionSafe(Node* node) {
    if (node->hasAttribute(Symbol("allowzero")) &&
        node->i(Symbol("allowzero")) == 1) {
      return true;
    }
    const Tensor* shape = FetchConstantTensor(node->input(1));
    if (shape == nullptr ||
        shape->elem_type() != ONNX_NAMESPACE::TensorProto_DataType_INT64) {
      return false;
    }
    const auto dims = ParseTensorData<int64_t>(shape);
    return std::none_of(dims.begin(), dims.end(),
                        [](int64_t d) { return d == 0; });
  }

  bool runTransform(Node* node, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    if (node->kind() == kReshape && !IsReshapeFusionSafe(node)) {
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
