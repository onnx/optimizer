/*
 * SPDX-License-Identifier: Apache-2.0
 */

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

// before: B = mul(A, 1) or B = mul(1, A)
// after: B = A

#include "onnx/defs/tensor_util.h"
#include "onnxoptimizer/pass.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct EliminateNopMulOne final : public PredicateBasedPass {
  explicit EliminateNopMulOne()
      : PredicateBasedPass(PassType::Nop, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "eliminate_nop_mul_one";
  }

  bool isConstantTensor(Graph& graph, const std::string& name) {
    auto& initializer_names = graph.initializer_names();
    return std::find(initializer_names.cbegin(), initializer_names.cend(),
                     name) != initializer_names.cend();
  }

  bool isABroadcastToB(const std::vector<int64_t>& dims_a,
                       const std::vector<Dimension>& dims_b) {
    int ndim_a = dims_a.size();
    int ndim_b = dims_b.size();
    if (ndim_a > ndim_b) {
      return false;
    }

    ndim_a--;
    ndim_b--;

    for (; ndim_a >= 0; ndim_a--, ndim_b--) {
      int d_a = dims_a[ndim_a];
      auto const& d_b = dims_b[ndim_b];
      if (d_a == 1) {
        continue;
      }
      if (!d_b.is_int || (d_a != d_b.dim)) {
        return false;
      }
    }
    return true;
  }

  template <typename T>
  bool isAllValue(const std::vector<T>& data, const T& val = {1}) {
    return std::all_of(data.cbegin(), data.cend(),
                       [&val](const T& v) { return v == val; });
  }

  bool isTensorOfOne(const Tensor& tensor) {
    int elem_type = tensor.elem_type();

#define CASE_BRANCH_CONTENT(pb_dtype, cpp_dtype, one)                  \
  case ONNX_NAMESPACE::TensorProto_DataType_##pb_dtype: {              \
    const std::vector<cpp_dtype> data = ParseData<cpp_dtype>(&tensor); \
    return isAllValue<cpp_dtype>(data, one);                           \
  }

    switch (elem_type) {
      CASE_BRANCH_CONTENT(FLOAT, float, float{1})
      CASE_BRANCH_CONTENT(INT32, int32_t, int32_t{1})
      CASE_BRANCH_CONTENT(INT64, int64_t, int64_t{1})
      CASE_BRANCH_CONTENT(DOUBLE, double, double{1})
      CASE_BRANCH_CONTENT(FLOAT16, int32_t, 15360)
      CASE_BRANCH_CONTENT(UINT8, int32_t, int32_t{1})
      CASE_BRANCH_CONTENT(INT8, int32_t, int32_t{1})
      CASE_BRANCH_CONTENT(UINT16, int32_t, int32_t{1})
      CASE_BRANCH_CONTENT(INT16, int32_t, int32_t{1})
      // TODO: support uint64
      //  CASE_BRANCH_CONTENT(UINT32, uint64_t, uint64_t{1})
      //  CASE_BRANCH_CONTENT(UINT64, uint64_t, uint64_t{1})
    }
#undef CASE_BRANCH_CONTENT
    return false;
  }

  bool patternMatchPredicate(Node* node) override {
    return node->kind() == kMul &&
           (node->inputs()[0]->node()->kind() == kParam ||
            node->inputs()[1]->node()->kind() == kParam);
  }

  bool runTransform(Node* node, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    auto& a_value = node->inputs()[0];
    auto& b_value = node->inputs()[1];

    const auto a_name = a_value->uniqueName();
    const auto b_name = b_value->uniqueName();

    const auto a_tensor = graph.getInitializer(a_name);
    const auto b_tensor = graph.getInitializer(b_name);

    bool replacing_success = false;

    if (isConstantTensor(graph, a_name) && isTensorOfOne(*a_tensor)) {
      replacing_success =
          isABroadcastToB(a_tensor->sizes(), b_value->sizes()) &&
          tryReplacingAllUsesWith(node->output(), b_value);
    }

    if (!replacing_success && isConstantTensor(graph, b_name) &&
        isTensorOfOne(*b_tensor)) {
      replacing_success =
          isABroadcastToB(b_tensor->sizes(), a_value->sizes()) &&
          tryReplacingAllUsesWith(node->output(), a_value);
    }

    if (!replacing_success) {
      return false;
    }
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

}  // namespace optimization
}  // namespace ONNX_NAMESPACE
