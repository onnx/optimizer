/*
 * SPDX-License-Identifier: Apache-2.0
 */

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

/// this pass can simplify focus network

#pragma once

#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct FuseConsecutiveSlices final : public PredicateBasedPass {
  explicit FuseConsecutiveSlices()
      : PredicateBasedPass(PassType::Fuse, PassEfficiency::Complete,
                           PassOptimizationType::Memory) {}

  std::string getPassName() const override {
    return "fuse_consecutive_slices";
  }

  bool patternMatchPredicate(Node *node) override {
    std::vector<int64_t> slice1_axes, slice2_axes;
    if (CheckKind(node, kSlice, 0, kSlice) && node->inputs().size() == 5 &&
           GetInputsOfPreNode(node, 0).size() == 5 &&
           GetValueFromInput(node, 3, slice1_axes) &&
           GetValueFromInput(PrevNode(node, 0), 3, slice2_axes)) {
      if (!node->input(0)->has_sizes()) {
        return false;
      }
      for (auto& axis : slice1_axes) {
        axis = AddYIfNegative(axis, node->inputs()[0]->sizes().size());
      }
      for (auto& axis : slice2_axes) {
        axis = AddYIfNegative(axis, node->inputs()[0]->sizes().size());
      }
      bool has_intersection = HasIntersection(slice1_axes, slice2_axes);
      return !has_intersection;
    }
    return false;
  }
  bool runTransform(Node *n, Graph &graph,
                    NodeDestroyType &destroy_current) override {
    /*
          X
          |
        Slice2
          |
        Slice1
          |
          Y
    */
    Node *slice1 = n;
    Node *slice2 = PrevNode(n, 0);

    std::vector<Node *> new_nodes;
    for (int i = 1; i < 5; ++i) {
      Node *node = graph.create(kConcat, 1);
      node->addInput(slice2->input(i));
      node->addInput(slice1->input(i));
      node->i_(kaxis, 0);
      new_nodes.push_back(node);
    }

    Node *new_slice = graph.create(kSlice, 1);
    new_slice->insertBefore(slice1);
    new_slice->addInput(slice2->input(0));
    for (auto *node : new_nodes) {
      new_slice->addInput(node->output());
      node->insertBefore(new_slice);
    }

    const bool replacing_success =
        tryReplacingAllUsesWith(slice1->output(), new_slice->output());
    if (!replacing_success) {
      return false;
    }
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

}  // namespace optimization
}  // namespace ONNX_NAMESPACE
