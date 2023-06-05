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

struct EliminateNopReshape final : public PredicateBasedPass {
  explicit EliminateNopReshape()
      : PredicateBasedPass(PassType::Nop, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "eliminate_nop_reshape";
  }

  bool patternMatchPredicate(Node *node) override {
    return node->kind() == kReshape && !node->inputs()[0]->sizes().empty() &&
           IsConstantTensor(node, 1);
  }

  bool runTransform(Node *node, Graph &graph,
                    NodeDestroyType &destroy_current) override {
    const auto &old_shape = node->inputs()[0]->sizes();
    const auto *new_shape_input = node->inputs()[1];

    const Tensor *new_shape_tensor = FetchConstantTensor(new_shape_input);
    if (!new_shape_tensor) {
      return false;
    }

    if (new_shape_tensor->elem_type() !=
        ONNX_NAMESPACE::TensorProto_DataType_INT64) {
      return false;
    }
    const auto new_shape = ParseTensorData<int64_t>(new_shape_tensor);

    if (new_shape.size() != old_shape.size()) {
      return false;
    }

    int unknown_dim_count = 0;
    for (int i = 0; i < new_shape.size(); ++i) {
      const auto new_dim = new_shape[i];
      // the dim can be copied from the input only when allowzero == 0
      if (new_dim == 0 && !(node->hasAttribute(Symbol("allowzero")) &&
                      node->i(Symbol("allowzero")) == 1)) {
        continue;
      }
      if (old_shape[i].is_int) {
        if (new_dim == -1) {
          unknown_dim_count++;
        } else if (old_shape[i].dim != new_dim) {
          return false;
        }
      } else {
        unknown_dim_count++;
      }
    }
    if (unknown_dim_count > 1) {
      return false;
    }

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
