/*
 * SPDX-License-Identifier: Apache-2.0
 */

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include "onnxoptimizer/pass.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct EliminateNopTranspose final : public PredicateBasedPass {
  explicit EliminateNopTranspose()
      : PredicateBasedPass(PassType::Nop, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "eliminate_nop_transpose";
  }

  static bool is_nop_transpose(const std::vector<int64_t>& perm) {
    for (size_t i = 0; i < perm.size(); i++)
      if (perm[i] != (int)i)
        return false;
    return true;
  }

  bool patternMatchPredicate(Node* node) override {
    return (node->kind() == kTranspose && node->hasAttribute(kperm)) &&
           is_nop_transpose(node->is(kperm));
  }

  bool runTransform(Node* node, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    const bool replacing_success =
        tryReplacingAllUsesWith(node->output(), node->input());
    if (!replacing_success) {
      return false;
    }
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

}  // namespace optimization
}  // namespace ONNX_NAMESPACE
