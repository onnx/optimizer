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

struct EliminateNopReshape final : public PredicateBasedPass {
  explicit EliminateNopReshape()
      : PredicateBasedPass(PassType::Nop, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "eliminate_nop_reshape";
  }

  bool patternMatchPredicate(Node *node) override {
    return node->kind() == kReshape && !node->inputs()[0]->sizes().empty();
  }

  bool runTransform(Node *node, Graph &graph,
                    NodeDestroyType &destroy_current) override {
    const auto &data_dims = node->inputs()[0]->sizes();
    auto *shape = node->inputs()[1];
    const auto &initializer_names = graph.initializer_names();

    const Tensor *shape_tensor = nullptr;
    if (shape->node()->kind() == kConstant) {
      shape_tensor = &shape->node()->t(kvalue);
    } else if (shape->node()->kind() == kParam &&
               std::find(initializer_names.cbegin(), initializer_names.cend(),
                         shape->uniqueName()) != initializer_names.cend()) {
      shape_tensor = &*graph.getInitializer(shape->uniqueName());
    } else {
      return false;
    }

    if (shape_tensor->elem_type() != ONNX_NAMESPACE::TensorProto_DataType_INT64) {
      return false;
    }
    const auto shape_data = ParseData<int64_t>(shape_tensor);

    if (shape_data.size() != data_dims.size()) {
      return false;
    }

    int unknown_dim_count = 0;
    for (int i = 0; i < shape_data.size(); ++i) {
      const auto d = shape_data[i];
      if (d == 0) {
        continue;
      }
      if (data_dims[i].is_int) {
        if (data_dims[i].dim != d) {
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
