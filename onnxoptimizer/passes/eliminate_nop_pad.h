/*
 * SPDX-License-Identifier: Apache-2.0
 */

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include "onnx/defs/tensor_util.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"
#include "onnxoptimizer/passes/string_utils.h"
#include "onnxoptimizer/passes/logging.h"


namespace ONNX_NAMESPACE {
namespace optimization {

struct EliminateNopPad final : public PredicateBasedPass {
  explicit EliminateNopPad()
      : PredicateBasedPass(PassType::Nop, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "eliminate_nop_pad";
  }

  static bool is_nop_pad(Node* node, Graph& graph) {
    std::vector<int64_t> pads;
    if (!GetValueFromAttrOrInput(node, kpads, 1, pads) || pads.empty()) {
      return false;
    }
    VLOG(1) << Str("pads",pads);
    for (const auto& p : pads) {
      if (p != 0) {
        return false;
      }
    }
    return true;
  }

  bool patternMatchPredicate(Node* node) override {
    return node->kind() == kPad;
  }

  bool runTransform(Node* node, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    if (!is_nop_pad(node, graph))
      return false;
    const bool replacing_success =
        tryReplacingAllUsesWith(node->output(), node->inputs()[0]);
    if (!replacing_success) {
      return false;
    }
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

}  // namespace optimization
}  // namespace ONNX_NAMESPACE
