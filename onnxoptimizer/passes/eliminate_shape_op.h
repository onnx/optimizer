/*
 * SPDX-License-Identifier: Apache-2.0
 */

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include "onnx/common/tensor.h"
#include "onnxoptimizer/pass.h"
#include "pass_util.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct EliminateShapeOp final : public PredicateBasedPass {
  explicit EliminateShapeOp()
      : PredicateBasedPass(PassType::Fuse, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "eliminate_shape_op";
  }

  bool patternMatchPredicate(Node *node) override {
    if (!CheckKind(node, "Shape")) {
      return false;
    }
    const Value *input = node->input();
    if (!input->has_sizes()) {
      return false;
    }

    const auto [start, end] = FetchStartAndEndAttrOfShape(node);

    return std::all_of(input->sizes().cbegin() + start,
                       input->sizes().cbegin() + end,
                       [](const auto &dim) { return dim.is_int && dim.dim >= 0; });
  }

  bool runTransform(Node *node, Graph &graph,
                    NodeDestroyType &destroy_current) override {
    const Value *input = node->input();
    const auto [start, end] = FetchStartAndEndAttrOfShape(node);

    Tensor tensor;
    tensor.sizes().push_back(end - start);
    tensor.elem_type() = ONNX_NAMESPACE::TensorProto_DataType_INT64;
    std::transform(input->sizes().begin() + start, input->sizes().begin() + end,
                   std::back_inserter(tensor.int64s()),
                   [](const auto &dim) { return dim.dim; });

    Value *value = graph.addInitializerAndCreateValue(tensor);

    const bool replacing_success =
        tryReplacingAllUsesWith(node->output(), value);
    if (!replacing_success) {
      return false;
    }
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

}  // namespace optimization
}  // namespace ONNX_NAMESPACE
