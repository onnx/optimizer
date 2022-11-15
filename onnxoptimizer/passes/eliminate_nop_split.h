/*
 * SPDX-License-Identifier: Apache-2.0
 */

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include "onnxoptimizer/pass.h"
#include "pass_util.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct EliminateNopSplit final : public PredicateBasedPass {
  explicit EliminateNopSplit()
      : PredicateBasedPass(PassType::Nop, PassEfficiency::Complete,
                           PassOptimizationType::Memory) {}

  std::string getPassName() const override {
    return "eliminate_nop_split";
  }

  bool patternMatchPredicate(Node* node) override {
    // TODO: support Split-18 where `split` is an input instead of an attribute
    return CheckKind(node, "Split") && node->inputs().size() == 1 &&
           node->input()->has_sizes() && node->hasAttribute(kaxis) &&
           node->hasAttribute(ksplit) && node->outputs().size() == 1;
  }

  bool runTransform(Node* node, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    auto* input = node->input();
    const auto& sizes = input->sizes();
    int64_t axis =
        AddYIfNegative(node->i(kaxis), static_cast<int64_t>(sizes.size()));
    const auto split = node->is(ksplit);
    if (!sizes[axis].is_int || sizes[axis].dim != split[0]) {
      return false;
    }

    const bool replacing_success =
        tryReplacingAllUsesWith(node->output(), input);
    if (!replacing_success) {
      return false;
    }
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

}  // namespace optimization
}  // namespace ONNX_NAMESPACE
