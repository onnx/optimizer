/*
 * SPDX-License-Identifier: Apache-2.0
 */

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include "onnxoptimizer/pass.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct SetUniqueNameForNodes final : public PredicateBasedPass {
  explicit SetUniqueNameForNodes()
      : PredicateBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::None) {}

  std::string getPassName() const override {
    return "set_unique_name_for_nodes";
  }

  bool patternMatchPredicate(Node* node) override {
    return !node->has_name();
  }

  bool runTransform(Node* node, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    node->setName(ONNX_NAMESPACE::to_string(graph.getNextUnique()));
    destroy_current = NodeDestroyType::DestroyZero;
    return true;
  }
};

}  // namespace optimization
}  // namespace ONNX_NAMESPACE
