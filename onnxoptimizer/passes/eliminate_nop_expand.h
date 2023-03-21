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

struct EliminateNopExpand final : public PredicateBasedPass {
  explicit EliminateNopExpand()
      : PredicateBasedPass(PassType::Nop, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "eliminate_nop_expand";
  }

  bool patternMatchPredicate(Node* node) override {
    return node->kind() == kExpand && IsConstantTensor(node, 1);
  }

  bool runTransform(Node* node, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    auto& input_value = node->inputs()[0];
    const auto* shape_tensor = FetchConstantTensor(node->input(1));

    if (!shape_tensor ||
        !isABroadcastToB(ParseTensorData<int64_t>(shape_tensor),
                         input_value->sizes()) ||
        !tryReplacingAllUsesWith(node->output(), input_value)) {
      return false;
    }

    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

}  // namespace optimization
}  // namespace ONNX_NAMESPACE
