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

struct EliminateOpWithUnit final : public PredicateBasedPass {
  explicit EliminateOpWithUnit()
      : PredicateBasedPass(PassType::Nop, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "eliminate_nop_with_unit";
  }

#define PROTO_DTYPE_TO_CPP_DTYPE_LIST(_)             \
  _(TensorProto_DataType_FLOAT, float, 1, 0)         \
  _(TensorProto_DataType_INT32, int32_t, 1, 0)       \
  _(TensorProto_DataType_INT64, int64_t, 1, 0)       \
  _(TensorProto_DataType_DOUBLE, double, 1, 0)       \
  _(TensorProto_DataType_FLOAT16, int32_t, 15360, 0) \
  _(TensorProto_DataType_UINT8, int32_t, 1, 0)       \
  _(TensorProto_DataType_INT8, int32_t, 1, 0)        \
  _(TensorProto_DataType_UINT16, int32_t, 1, 0)      \
  _(TensorProto_DataType_INT16, int32_t, 1, 0)       \
  _(TensorProto_DataType_BOOL, int32_t, 1, 0)
  // _(TensorProto_DataType_UINT32, uint64_t, 1, 0)     \
  // _(TensorProto_DataType_UINT64, uint64_t, 1, 0)

  bool isOnes(const Tensor& tensor) {
    int elem_type = tensor.elem_type();
#define CASE_BRANCH_CONTENT(pb_dtype, cpp_dtype, one, zero)                 \
  case pb_dtype: {                                                          \
    const std::vector<cpp_dtype> data = ParseData<cpp_dtype>(&tensor);      \
    return std::all_of(data.cbegin(), data.cend(), [](const cpp_dtype& v) { \
      return v == cpp_dtype{one};                                           \
    });                                                                     \
  }

    switch (elem_type) { PROTO_DTYPE_TO_CPP_DTYPE_LIST(CASE_BRANCH_CONTENT) }
#undef CASE_BRANCH_CONTENT
    return false;
  }

  bool isZeros(const Tensor& tensor) {
    int elem_type = tensor.elem_type();
#define CASE_BRANCH_CONTENT(pb_dtype, cpp_dtype, one, zero)                 \
  case pb_dtype: {                                                          \
    const std::vector<cpp_dtype> data = ParseData<cpp_dtype>(&tensor);      \
    return std::all_of(data.cbegin(), data.cend(), [](const cpp_dtype& v) { \
      return v == cpp_dtype{zero};                                          \
    });                                                                     \
  }

    switch (elem_type) { PROTO_DTYPE_TO_CPP_DTYPE_LIST(CASE_BRANCH_CONTENT) }
#undef CASE_BRANCH_CONTENT
    return false;
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
  bool patternMatchPredicate(Node* node) override {
    return true;
  };

  bool runTransform(Node* node, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    uint32_t kind = node->kind();
    bool is_mul = kind == kMul || kind == Symbol("And");
    bool is_add = kind == kAdd || kind == Symbol("Or");
    bool is_sub = kind == kSub;
    bool is_div = kind == kDiv;
    bool is_pow = kind == kPow;
    if (!is_mul && !is_add && !is_sub && !is_div && !is_pow) {
      return false;
    }

    Value* x = node->inputs()[0];
    Value* y = node->inputs()[1];
    bool x_is_constant =
        x->node()->kind() == kParam && isConstantTensor(graph, x->uniqueName());
    bool y_is_constant =
        y->node()->kind() == kParam && isConstantTensor(graph, y->uniqueName());
    if (!x_is_constant && !y_is_constant) {
      return false;
    }
    const auto x_tensor = graph.getInitializer(x->uniqueName());
    const auto y_tensor = graph.getInitializer(y->uniqueName());
    bool x_is_one = x_is_constant && isOnes(*x_tensor);
    bool x_is_zero = x_is_one ? false : (x_is_constant && isZeros(*x_tensor));
    bool y_is_one = y_is_constant && isOnes(*y_tensor);
    bool y_is_zero = y_is_one ? false : (y_is_constant && isZeros(*y_tensor));
    bool x_broadcast_to_y =
        x_is_constant && isABroadcastToB(x_tensor->sizes(), y->sizes());
    bool y_broadcast_to_x =
        y_is_constant && isABroadcastToB(y_tensor->sizes(), x->sizes());

    bool is_one = x_is_one || y_is_one;
    bool is_zero = x_is_zero || y_is_zero;
    if ((!x_is_one || !x_broadcast_to_y) && (!x_is_zero || !x_broadcast_to_y) &&
        (!y_is_one || !y_broadcast_to_x) && (!y_is_zero || !y_broadcast_to_x)) {
      return false;
    }
    bool replacing_success = false;
    if ((is_mul && is_one) || (is_add && is_zero)) {
      if (x_is_one || x_is_zero) {
        // 1 * y=y or 0 + y =y
        replacing_success = tryReplacingAllUsesWith(node->output(), y);
      } else if (y_is_one || y_is_zero) {
        // x * 1=x or x + 0 =x
        replacing_success = tryReplacingAllUsesWith(node->output(), x);
      }
    } else if (is_sub) {
      if (y_is_zero) {
        // x - 0 = x
        replacing_success = tryReplacingAllUsesWith(node->output(), x);
      } else if (x_is_zero) {
        // 0 - y = Neg(y)
        Node* neg = graph.create(kNeg, 1);
        neg->addInput(y);
        neg->output()->copyMetadata(node->output());
        neg->insertBefore(node);
        replacing_success = tryReplacingAllUsesWith(node, neg);
      }
    } else if (is_div && y_is_one) {
      // x / 1 = x
      replacing_success = tryReplacingAllUsesWith(node->output(), x);
    } else if (is_pow && y_is_one) {
      replacing_success = tryReplacingAllUsesWith(node->output(), x);
    }
    if (!replacing_success) {
      return false;
    }
    destroy_current = DestroyOne;
    return true;
  }
};

}  // namespace optimization
}  // namespace ONNX_NAMESPACE
