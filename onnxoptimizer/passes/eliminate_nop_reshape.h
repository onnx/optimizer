/*
 * SPDX-License-Identifier: Apache-2.0
 */

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include "onnx/defs/tensor_util.h"
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
    const auto &data_input_dims = node->inputs()[0]->sizes();
    const auto *shape_input = node->inputs()[1];

    const Tensor *shape_tensor = FetchConstantTensor(shape_input);
    if (!shape_tensor) {
      return false;
    }

    if (shape_tensor->elem_type() !=
        ONNX_NAMESPACE::TensorProto_DataType_INT64) {
      return false;
    }
    const auto shape_input_data = ParseData<int64_t>(shape_tensor);

    if (shape_input_data.size() != data_input_dims.size()) {
      return false;
    }

    int unknown_dim_count = 0;
    for (int i = 0; i < shape_input_data.size(); ++i) {
      const auto d = shape_input_data[i];
      if (d == 0) {
        continue;
      }
      if (data_input_dims[i].is_int) {
        if (data_input_dims[i].dim != d) {
          return false;
        }
        continue;
      }
      unknown_dim_count++;
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
