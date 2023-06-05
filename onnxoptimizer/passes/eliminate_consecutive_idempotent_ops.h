/*
 * SPDX-License-Identifier: Apache-2.0
 */

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

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
        "Ceil", "Floor", "Round", "Relu", "Reshape"};
    for (const auto& op : idempotent_ops) {
      // TODO: support uses().size() > 1 for ops except Reshape
      if (CheckKind(node, Symbol(op), 0, Symbol(op)) &&
          node->input(0)->uses().size() == 1) {
        return true;
      }
    }
    return false;
  }
  bool runTransform(Node* node, Graph& graph,
                    NodeDestroyType& destroy_current) override {
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
