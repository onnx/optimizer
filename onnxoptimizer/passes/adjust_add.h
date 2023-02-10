/*
 * SPDX-License-Identifier: Apache-2.0
 */

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

/*

 Before:
  bias    MatMul(X,weights)
    |          |
     |        |
       |     |
         Add

  This case could not be fused in TensorRT, but adjust bias to the second input,
  it will be fused Gemm+biasAdd.

After:

  MatMul(X,weights)    bias
           |             |
              |        |
                |     |
                  Add

*/

#include <numeric>

#include "onnx/defs/tensor_util.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"
namespace ONNX_NAMESPACE {
namespace optimization {

struct AdjustAdd final : public PredicateBasedPass {
  explicit AdjustAdd()
      : PredicateBasedPass(PassType::Immutable, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override {
    return "adjust_add";
  }

  bool patternMatchPredicate(Node* node) override {
    return CheckKind(node, kAdd) && IsConstantTensor(node, 0) &&
           !IsConstantTensor(node, 1);
  }

  bool runTransform(Node* n, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    auto* old = n->replaceInput(0, n->inputs()[1]);
    n->replaceInput(1, old);
    destroy_current = NodeDestroyType::DestroyZero;
    return true;
  }
};

}  // namespace optimization
}  // namespace ONNX_NAMESPACE
