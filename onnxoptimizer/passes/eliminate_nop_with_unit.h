/*
 * SPDX-License-Identifier: Apache-2.0
 */

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

// before: B = add(A, 0) or B = add(0, A)
// after: B = A

#include "onnx/defs/tensor_util.h"
#include "onnxoptimizer/pass.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct EliminateOpWithUnit final : public PredicateBasedPass {
  explicit EliminateOpWithUnit()
      : PredicateBasedPass(PassType::Nop, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "eliminate_nop_with_unit";
  }

  bool patternMatchPredicate(Node* node) override;

  bool runTransform(Node* node, Graph& graph,
                    NodeDestroyType& destroy_current) override;
};

}  // namespace optimization
}  // namespace ONNX_NAMESPACE
