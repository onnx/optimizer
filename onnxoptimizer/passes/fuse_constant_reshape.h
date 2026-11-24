/*
 * SPDX-License-Identifier: Apache-2.0
 */


#pragma once

// Before:
//   B = Reshape(Constant)
// After:
//   B = Constant (Constant with new shape)

#include <limits>

#include "onnx/defs/tensor_util.h"
#include "onnxoptimizer/pass.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct FuseConstantReshape final : public PredicateBasedPass {
  explicit FuseConstantReshape()
      : PredicateBasedPass(PassType::Fuse, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override {
    return "fuse_constant_reshape";
  }

  bool patternMatchPredicate(Node* node) override {
    return node->kind() == kReshape && !node->inputs().empty() &&
           node->inputs()[0]->node()->kind() == kConstant;
  }
  bool runTransform(Node* n, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    destroy_current = NodeDestroyType::DestroyZero;

    // check if Constant is only used by Reshape
    if (n->inputs()[0]->uses().size() > 1) {
      return false;
    }

    Node* reshape = n;
    Node* constant = n->inputs()[0]->node();
    if (!constant->hasAttribute(kvalue)) {
      return false;
    }

    // Process 'reshape' data
    std::vector<int64_t> shape;
    if (reshape->hasAttribute(kshape)) {
      // Before opset 5.
      shape = reshape->is(kshape);
    } else {
      if (reshape->inputs().size() < 2 ||
          reshape->inputs()[1]->node()->kind() != kConstant) {
        return false;
      }
      if (reshape->inputs()[1]->uses().size() > 1) {
        return false;
      }
      Node* shape_const = reshape->inputs()[1]->node();
      if (!shape_const->hasAttribute(kvalue)) {
        return false;
      }
      Tensor t = shape_const->t(kvalue);
      if (t.elem_type() != TensorProto_DataType_INT64 ||
          t.sizes().size() != 1) {
        return false;
      }
      shape = ParseData<int64_t>(&t);
    }

    int64_t allow_zero = 0;
    Symbol sym = Symbol("allowzero");
    if (reshape->hasAttribute(sym)) {
      allow_zero = reshape->i(sym);
    }
    if (allow_zero != 0 && allow_zero != 1) {
      return false;
    }

    Tensor t = constant->t(kvalue);
    const auto& ori_size = t.sizes();

    const auto multiply = [](int64_t& product, int64_t dimension) {
      if (dimension < 0 ||
          (dimension != 0 &&
           product > std::numeric_limits<int64_t>::max() / dimension)) {
        return false;
      }
      product *= dimension;
      return true;
    };
    int64_t numel = 1;
    for (int64_t dimension : ori_size) {
      if (!multiply(numel, dimension)) {
        return false;
      }
    }

    size_t unknown_index = shape.size();
    int64_t known_product = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
      if (shape[i] == 0 && allow_zero == 0) {
        if (i >= ori_size.size()) {
          return false;
        }
        shape[i] = ori_size[i];
      }
      if (shape[i] == -1) {
        if (unknown_index != shape.size()) {
          return false;
        }
        unknown_index = i;
      } else if (!multiply(known_product, shape[i])) {
        return false;
      }
    }
    if (unknown_index != shape.size()) {
      if (known_product == 0 || numel % known_product != 0) {
        return false;
      }
      shape[unknown_index] = numel / known_product;
    } else if (known_product != numel) {
      return false;
    }

    const bool replacing_success = tryReplacingAllUsesWith(reshape->output(), reshape->inputs()[0]);
    if (!replacing_success) {
      return false;
    }
    t.sizes() = shape;
    constant->output()->setSizes(
        std::vector<Dimension>(shape.begin(), shape.end()));
    constant->output()->setElemType(t.elem_type());
    constant->t_(kvalue, std::move(t));
    destroy_current = NodeDestroyType::DestroyOne;
    return true;

  }
};

}  // namespace optimization
}  // namespace ONNX_NAMESPACE
