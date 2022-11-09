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

// before
// Z = Reshape(X, Concat(...)) or Z = Reshape(X, Cast(Concat(...), to=INT64 ))
// after
// Z = Reshape(X, Y) , Y is a constant tensor

// this pass can handle the case when:
//   1. the number of unknown dims in the result of Concat should be not more
//   than 1

struct FuseConcatIntoReshape final : public PredicateBasedPass {
  explicit FuseConcatIntoReshape()
      : PredicateBasedPass(PassType::Fuse, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "fuse_concat_into_reshape";
  }

  inline bool matchConcatReshape(Node *node) {
    return CheckKind(node, kReshape) && CheckKind(node->inputs()[1], kConcat) &&
           node->input(1)->node()->i(kaxis) == 0;
  }

  inline bool matchConcatCastReshape(Node *node) {
    return CheckKind(node, kReshape) && CheckKind(node->inputs()[1], kCast) &&
           node->inputs()[1]->node()->i(kto) ==
               ONNX_NAMESPACE::TensorProto_DataType_INT64 &&
           CheckKind(node->inputs()[1]->node()->input(), kConcat) &&
           node->inputs()[1]->node()->input()->node()->i(kaxis) == 0;
  }

  bool patternMatchPredicate(Node *node) override {
    return matchConcatReshape(node) || matchConcatCastReshape(node);
  }

  bool runTransform(Node *node, Graph &graph,
                    NodeDestroyType &destroy_current) override {
    const bool has_cast = matchConcatCastReshape(node);

    Node *concat = nullptr;
    if (has_cast) {
      concat = node->inputs()[1]->node()->input()->node();
    } else {
      concat = node->inputs()[1]->node();
    }

    std::vector<int64_t> shapes;
    for (const auto *v : concat->inputs()) {
      const Tensor *tensor = FetchConstantTensor(v);

      if (tensor == nullptr) {
        // If the value of v is unknown, and v has only one element, we can
        // represent it with -1 in Reshape op.
        // TODO:
        //  support the case that v is a shape and only one of the dims is
        //  unknown. Example: Concat [?, 2, 3] and [4] into [-1, 2, 3, 4]
        if (v->sizes().size() != 1 || !v->sizes()[0].is_int ||
            v->sizes()[0].dim != 1) {
          return false;
        }
        shapes.push_back(-1);
        continue;
      }
      // only support INT64 when the pattern is concat->reshape
      if (!has_cast &&
          tensor->elem_type() != ONNX_NAMESPACE::TensorProto_DataType_INT64) {
        return false;
      }
#define DO_CASE(pb_type, cpp_type)                                   \
  case ONNX_NAMESPACE::TensorProto_DataType_##pb_type: {             \
    const auto data = ParseData<cpp_type>(tensor);                   \
    std::transform(                                                  \
        data.cbegin(), data.cend(), std::back_inserter(shapes),      \
        [](const cpp_type &v) { return static_cast<cpp_type>(v); }); \
    break;                                                           \
  }

      switch (tensor->elem_type()) {
        DO_CASE(FLOAT, float)
        DO_CASE(INT32, int32_t)
        DO_CASE(INT64, int64_t)
        DO_CASE(DOUBLE, double)
        DO_CASE(UINT8, int32_t)
        DO_CASE(INT8, int32_t)
        DO_CASE(UINT16, int32_t)
        DO_CASE(INT16, int32_t)
        default:
          return false;
      }
    }
    int unknown_dim_count = 0;
    for (auto dim : shapes) {
      unknown_dim_count += int64_t(dim == -1);
    }
    if (unknown_dim_count > 1) {
      return false;
    }

    Tensor t;
    t.sizes().push_back(shapes.size());
    t.elem_type() = ONNX_NAMESPACE::TensorProto_DataType_INT64;
    t.int64s().swap(shapes);
    Value *value = graph.addInitializerAndCreateValue(t);

    node->replaceInput(1, value);
    destroy_current = NodeDestroyType::DestroyZero;
    return true;
  }
};

}  // namespace optimization
}  // namespace ONNX_NAMESPACE
