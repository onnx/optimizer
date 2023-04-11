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
    return CheckKind(node, "Split") && node->inputs()[0]->has_sizes() &&
           node->outputs().size() == 1;
  }

  bool runTransform(Node* node, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    auto* input = node->inputs()[0];
    const auto& sizes = input->sizes();
    int64_t axis = GetValueFromAttrWithDefault(node, kaxis, int64_t{0});
    axis = AddYIfNegative(axis, static_cast<int64_t>(sizes.size()));
    std::vector<int64_t> split;
    if (GetValueFromAttrOrInput(node, ksplit, 1, split) && !split.empty() &&
        (!sizes[axis].is_int || sizes[axis].dim != split[0])) {
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
