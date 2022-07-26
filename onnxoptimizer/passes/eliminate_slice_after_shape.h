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

struct EliminateSliceAfterShape final : public PredicateBasedPass {
  explicit EliminateSliceAfterShape()
      : PredicateBasedPass(PassType::Fuse, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "eliminate_slice_after_shape";
  }

  bool patternMatchPredicate(Node *node) override {
    return node->kind() == kSlice && CheckKind(node->inputs()[0], "Shape") &&
           node->inputs()[0]->node()->input()->has_sizes();
  }

  bool runTransform(Node *node, Graph &graph,
                    NodeDestroyType &destroy_current) override {
    Node *shape_node = node->inputs()[0]->node();
    const auto &dims_of_shape_node_input = shape_node->input()->sizes();
    std::vector<Dimension> result_of_shape_op;

    {
      const auto [shape_start, shape_end] =
          FetchStartAndEndAttrOfShape(shape_node);

      for (int i = shape_start; i < shape_end; ++i) {
        result_of_shape_op.push_back(dims_of_shape_node_input[i]);
      }
    }

    int64_t slice_start = 0, slice_end = result_of_shape_op.size(),
            slice_step = 1;
    const int opset_version = getOpsetVersion(graph);
    if (opset_version < 10) {
      if (!FetchSoleIntValueOfAttr(node, kstarts, slice_start) ||
          !FetchSoleIntValueOfAttr(node, kends, slice_end)) {
        return false;
      }
    } else {
      if (!FetchSoleIntValueOfTensor(node->inputs()[1], slice_start) ||
          !FetchSoleIntValueOfTensor(node->inputs()[2], slice_end) ||
          (node->inputs().size() == 5 &&
           !FetchSoleIntValueOfTensor(node->inputs()[4], slice_step)) ||
          slice_step == 0) {
        return false;
      }
    }

    slice_start =
        AddYIfNegative<int64_t>(slice_start, result_of_shape_op.size());
    slice_end = AddYIfNegative<int64_t>(slice_end, result_of_shape_op.size());

    std::vector<int64_t> result_of_slice_op;
    if (slice_step > 0) {
      slice_start =
          std::clamp<int64_t>(slice_start, 0, result_of_shape_op.size());
      slice_end = std::clamp<int64_t>(slice_end, 0, result_of_shape_op.size());
      for (; slice_start < slice_end; slice_start += slice_step) {
        assert(slice_start < result_of_shape_op.size());
        const auto &d = dims_of_shape_node_input[slice_start];
        if (!d.is_int) {
          return false;
        }
        result_of_slice_op.push_back(d.dim);
      }
    } else {
      slice_start =
          std::clamp<int64_t>(slice_start, 0, result_of_shape_op.size() - 1);
      slice_end = std::clamp<int64_t>(slice_end, -1, result_of_shape_op.size());
      for (; slice_start > slice_end; slice_start += slice_step) {
        assert(slice_start < result_of_shape_op.size() && slice_start >= 0);
        const auto &d = dims_of_shape_node_input[slice_start];
        if (!d.is_int) {
          return false;
        }
        result_of_slice_op.push_back(d.dim);
      }
    }
    Tensor tensor;
    tensor.sizes().push_back(result_of_slice_op.size());
    tensor.elem_type() = ONNX_NAMESPACE::TensorProto_DataType_INT64;
    tensor.int64s().swap(result_of_slice_op);
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
