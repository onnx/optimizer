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

struct EliminateShapeGather final : public PredicateBasedPass {
  explicit EliminateShapeGather()
      : PredicateBasedPass(PassType::Fuse, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "eliminate_shape_gather";
  }

  bool patternMatchPredicate(Node *node) override {
    return CheckKind(node, "Gather") && IsConstantTensor(node, 1) &&
           CheckKind(node->inputs()[0], "Shape") &&
           node->inputs()[0]->node()->input()->has_sizes();
  }

  bool runTransform(Node *node, Graph &graph,
                    NodeDestroyType &destroy_current) override {
    auto *x = node->inputs()[0];
    auto *indices = node->inputs()[1];
    Node *shape = x->node();
    const auto &dims = shape->input()->sizes();

    int64_t indices_val;
    if (!FetchSoleIntValueOfTensor(indices, indices_val)) {
      return false;
    }

    const auto [start, end] = FetchStartAndEndAttrOfShape(shape);

    indices_val = AddYIfNegative(indices_val, end - start);
    indices_val += start;

    ONNX_ASSERT(indices_val < dims.size());

    if (!dims[indices_val].is_int || dims[indices_val].dim == -1) {
      return false;
    }

    Tensor tensor;
    if (indices->sizes().size() == 1) {
      tensor.sizes().push_back(1);
    }
    tensor.elem_type() = ONNX_NAMESPACE::TensorProto_DataType_INT64;
    tensor.int64s().push_back(dims[indices_val].dim);
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
