/*
 * SPDX-License-Identifier: Apache-2.0
 */

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include "onnx/defs/tensor_util.h"
#include "onnxoptimizer/pass.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct EliminateNopExpand final : public PredicateBasedPass {
  explicit EliminateNopExpand()
      : PredicateBasedPass(PassType::Nop, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "eliminate_nop_expand";
  }

  bool isABroadcastToB(const std::vector<int64_t>& dims_a,
                       const std::vector<Dimension>& dims_b) {
    int ndim_a = dims_a.size();
    int ndim_b = dims_b.size();
    if (ndim_a > ndim_b) {
      return false;
    }

    ndim_a--;
    ndim_b--;

    for (; ndim_a >= 0; ndim_a--, ndim_b--) {
      int d_a = dims_a[ndim_a];
      auto const& d_b = dims_b[ndim_b];
      if (d_a == 1) {
        continue;
      }
      if (!d_b.is_int || (d_a != d_b.dim)) {
        return false;
      }
    }
    return true;
  }

  bool patternMatchPredicate(Node* node) override {
    return node->kind() == kExpand && node->inputs()[1]->node()->kind() == kParam;
  }

  bool runTransform(Node* node, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    auto& input_value = node->inputs()[0];
    const auto* shape_value = node->input(1);
    const auto shape_tensor_name = shape_value->uniqueName();
    const auto shape_tensor = graph.getInitializer(shape_tensor_name);

    if (!graph.is_constant_initializer(shape_value) ||
        !isABroadcastToB(ParseData<int64_t>(&*shape_tensor),
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
