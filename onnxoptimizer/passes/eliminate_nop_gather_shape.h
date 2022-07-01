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

struct EliminateNopGatherShape final : public PredicateBasedPass {
  explicit EliminateNopGatherShape()
      : PredicateBasedPass(PassType::Nop, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "eliminate_nop_gather_shape";
  }

  bool patternMatchPredicate(Node *node) override {
    return node->kind() == Symbol("Gather") &&
           node->inputs()[0]->node()->kind() == Symbol("Shape") &&
           node->inputs()[0]->node()->input()->has_sizes();
  }

  bool runTransform(Node *node, Graph &graph,
                    NodeDestroyType &destroy_current) override {
    auto *x = node->inputs()[0];
    auto *indices = node->inputs()[1];
    Node *shape = x->node();
    const auto &dims = shape->input()->sizes();

    const Tensor *indices_tensor = nullptr;
    if (indices->node()->kind() == kConstant) {
      indices_tensor = &indices->node()->t(kvalue);
    } else if (graph.is_constant_initializer(indices)) {
      indices_tensor = &*graph.getInitializer(indices->uniqueName());
    } else {
      return false;
    }
    assert(indices_tensor->sizes().size() <= 1);
    int indices_val;
    if (indices_tensor->elem_type() ==
        ONNX_NAMESPACE::TensorProto_DataType_INT32) {
      indices_val = ParseData<int32_t>(indices_tensor)[0];
    } else if (indices_tensor->elem_type() ==
               ONNX_NAMESPACE::TensorProto_DataType_INT64) {
      indices_val = ParseData<int64_t>(indices_tensor)[0];
    } else {
      return false;
    }

    auto add_y_if_negative = [](int x, int y) -> int {
      return x < 0 ? x + y : x;
    };

    const int start = add_y_if_negative(
        shape->hasAttribute(Symbol("start")) ? shape->i(Symbol("start")) : 0,
        dims.size());
    const int end = add_y_if_negative(shape->hasAttribute(Symbol("end"))
                                          ? shape->i(Symbol("end"))
                                          : dims.size(),
                                      dims.size());

    indices_val = add_y_if_negative(indices_val, end - start);
    indices_val += start;

    assert(indices_val < dims);

    if (!dims[indices_val].is_int) {
      return false;
    }

    Tensor tensor;
    if (indices_tensor->sizes().size() == 1) {
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
