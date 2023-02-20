/*
 * SPDX-License-Identifier: Apache-2.0
 */

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include <numeric>

#include "onnx/defs/tensor_util.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct FuseConsecutiveUnsqueezes final : public PredicateBasedPass {
  explicit FuseConsecutiveUnsqueezes()
      : PredicateBasedPass(PassType::Fuse, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override {
    return "fuse_consecutive_unsqueezes";
  }

  bool patternMatchPredicate(Node* node) override {
    return CheckKind(node, kUnsqueeze, 0, kUnsqueeze) &&
           GetInputsOfPreNode(node, 0)[0]->has_sizes();
  }

  bool runTransform(Node* n, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    destroy_current = NodeDestroyType::DestroyZero;
    Node* prev = PrevNode(n, 0);
    bool axes_is_attr = n->hasAttribute(kaxes);
    std::vector<int64_t> axes_of_prev, axes;
    if (!GetValueFromAttrOrInput(n, kaxes, 1, axes) ||
        !GetValueFromAttrOrInput(prev, kaxes, 1, axes_of_prev)) {
      return false;
    }
    const auto dims = prev->input(0)->sizes();
    for (auto& axis : axes_of_prev) {
      axis = AddYIfNegative(
          axis, static_cast<int64_t>(dims.size() + axes_of_prev.size()));
    }
    VLOG(1) << "axes of prev node: " << axes_of_prev;
    for (auto& axis : axes) {
      axis = AddYIfNegative(
          axis, static_cast<int64_t>(dims.size() + axes_of_prev.size() +
                                     axes.size()));
    }
    VLOG(1) << "axes : " << axes;
    std::sort(axes_of_prev.begin(), axes_of_prev.end());
    std::sort(axes.begin(), axes.end());

    std::vector<int64_t> fused_axes;
    for (auto& n : axes_of_prev) {
      for (const auto& m : axes) {
        if (m <= n) {
          n++;
        }
      }
    }
    fused_axes = axes_of_prev;
    std::transform(axes.cbegin(), axes.cend(), std::back_inserter(fused_axes),
                   [](const auto& d) { return d; });
    std::sort(fused_axes.begin(), fused_axes.end());
    VLOG(1) << "fused axes: " << fused_axes;
    n->replaceInput(0, prev->input(0));

    if (axes_is_attr) {
      n->is_(kaxes, std::move(fused_axes));
    } else {
      Tensor axes_t;
      axes_t.sizes().push_back(fused_axes.size());
      axes_t.elem_type() = ONNX_NAMESPACE::TensorProto_DataType_INT64;
      axes_t.int64s().swap(fused_axes);
      n->replaceInput(1, graph.addInitializerAndCreateValue(axes_t));
    }
    return true;
  }
};

}  // namespace optimization
}  // namespace ONNX_NAMESPACE