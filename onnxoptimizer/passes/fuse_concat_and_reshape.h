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

struct FuseConcatAndReshape final : public PredicateBasedPass {
  explicit FuseConcatAndReshape()
      : PredicateBasedPass(PassType::Fuse, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "fuse_concat_and_reshape";
  }

  bool patternMatchPredicate(Node *node) override {
    return node->kind() == kReshape &&
           node->inputs()[1]->node()->kind() == kConcat;
  }

  bool runTransform(Node *node, Graph &graph,
                    NodeDestroyType &destroy_current) override {
    auto *shape_value = node->inputs()[1];
    auto *concat = shape_value->node();
    const auto &initializer_names = graph.initializer_names();

    std::vector<int64_t> shapes;
    for (const auto *v : concat->inputs()) {
      const uint32_t kind = v->node()->kind();
      const Tensor *tensor = nullptr;
      if (kind == kConstant) {
        tensor = &v->node()->t(kvalue);
      } else if (kind == kParam &&
                 std::find(initializer_names.cbegin(), initializer_names.cend(),
                           v->uniqueName()) != initializer_names.cend()) {
        tensor = &*graph.getInitializer(v->uniqueName());
      } else {
        if (v->sizes().size() != 1) {
          return false;
        }
        shapes.push_back(-1);
        continue;
      }
      if (tensor->elem_type() != ONNX_NAMESPACE::TensorProto_DataType_INT64) {
        return false;
      }
      const auto data = ParseData<int64_t>(tensor);
      std::copy(data.cbegin(), data.cend(), std::back_inserter(shapes));
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

    const bool replacing_success = tryReplacingAllUsesWith(shape_value, value);
    if (!replacing_success) {
      return false;
    }
    destroy_current = NodeDestroyType::DestroyZero;
    return true;
  }
};

}  // namespace optimization
}  // namespace ONNX_NAMESPACE
