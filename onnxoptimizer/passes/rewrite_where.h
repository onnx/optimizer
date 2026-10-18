// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"

namespace ONNX_NAMESPACE {
namespace optimization {

// where(not(b), x, y) -> where(b, y, x)
// https://github.com/microsoft/onnxruntime/blob/v1.15.1/onnxruntime/core/optimizer/not_where_fusion.h
struct RewriteWhere final : public PredicateBasedPass {
  explicit RewriteWhere()
      : PredicateBasedPass(PassType::Nop, PassEfficiency::Partial,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "rewrite_where";
  }

  bool patternMatchPredicate(Node* node) override {
    bool isWhere = CheckKind(node, Symbol("Where"));
    if (isWhere) {
        return CheckKind(node->inputs()[0]->node(), Symbol("Not"));
    }
    return false;
  }
  bool runTransform(Node* node, Graph& graph,
                    NodeDestroyType& destroy_current) override {
        destroy_current = NodeDestroyType::DestroyZero;
        Node* previous_node = node->input(0)->node();
        if (previous_node->output()->uses().size() == 1) {
          const bool replacing_success =
              tryReplacingAllUsesWith(node->input(0), previous_node->input(0));
          if (!replacing_success) {
            return false;
          }
          auto x = node->inputs()[1];
          auto y = node->inputs()[2];
          node->replaceInput(1, y);
          node->replaceInput(2, x);
          previous_node->destroy();
          return true;
        }
        return false;
    }
};

}  // namespace optimization
}  // namespace ONNX_NAMESPACE
