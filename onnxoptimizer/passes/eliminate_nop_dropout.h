// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include "onnxoptimizer/pass.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct EliminateNopDropout final : public PredicateBasedPass {
  explicit EliminateNopDropout()
      : PredicateBasedPass(PassType::Nop, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "eliminate_nop_dropout";
  }

  bool patternMatchPredicate(Node* node) override {
    // in opset 12, ratio is an input of Dropout rather than an attribute,
    // however we don't want to to remove Dropout fro opset 12+, since it
    // supports training-friendly models, for which the Dropout ops are required
    return (node->kind() == kDropout && node->hasAttribute(kratio)) &&
           node->f(kratio) == 0.0;
  }

  bool runTransform(Node* node, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    // The optional mask is all ones, not the data input.
    if (node->outputs().size() > 1 && !node->outputs()[1]->uses().empty()) {
      return false;
    }
    const bool replacing_success =
        tryReplacingAllUsesWith(node->outputs()[0], node->input());
    if (!replacing_success) {
      return false;
    }
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

}  // namespace optimization
}  // namespace ONNX_NAMESPACE
