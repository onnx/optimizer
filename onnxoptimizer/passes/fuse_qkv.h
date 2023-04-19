/*
 * SPDX-License-Identifier: Apache-2.0
 */

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include <numeric>

#include "onnx/defs/tensor_util.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct FuseQKV final : public PredicateBasedPass {
  explicit FuseQKV()
      : PredicateBasedPass(PassType::Fuse, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override {
    return "fuse_qkv";
  }

  bool patternMatchPredicate(Node* node) override {
    return CheckKind(node, kMatMul) && node->input(0)->uses().size() == 3;
  }

  bool runTransform(Node* n, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    destroy_current = NodeDestroyType::DestroyZero;
    const auto uses = n->input(0)->uses();
    for (const auto& use : uses) {
      if (use.offset != 0 || !CheckKind(use.user, kMatMul) ||
          use.user->output()->uses().size() != 1 ||
          !IsConstantTensor(use.user, 1)) {
        return false;
      }
    }
    /// q k v not a one-to-one correspondence actually
    Node* q = uses[0].user;
    Node* k = uses[1].user;
    Node* v = uses[2].user;

    const Tensor* q_t = FetchConstantTensor(q->input(1));
    const Tensor* k_t = FetchConstantTensor(k->input(1));
    const Tensor* v_t = FetchConstantTensor(v->input(1));
    if (q_t->sizes() != k_t->sizes() || q_t->sizes() != v_t->sizes()) {
      return false;
    }
    Node* prev = PrevNode(n, 0);
    Node* cat = graph.create(kConcat, 1);
    cat->insertAfter(prev);
    cat->addInput(q->input(1));
    cat->addInput(k->input(1));
    cat->addInput(v->input(1));
    cat->i_(kaxis, q_t->sizes().size() - 1);
    Node* matmul = graph.create(kMatMul, 1);
    matmul->insertAfter(cat);
    matmul->addInput(q->input(0));
    matmul->addInput(cat->output());

    Node* split = graph.create("Split"_sym, 3);
    split->i_(kaxis, -1);
    split->insertAfter(matmul);
    split->addInput(matmul->output());
    const auto opset_version = getOpsetVersion(graph);
    if (opset_version >= 13) {
      Tensor split_t;
      split_t.sizes().push_back(3);
      split_t.elem_type() = ONNX_NAMESPACE::TensorProto_DataType_INT64;
      split_t.int64s().push_back(q_t->sizes().back());
      split_t.int64s().push_back(k_t->sizes().back());
      split_t.int64s().push_back(v_t->sizes().back());
      split->addInput(graph.addInitializerAndCreateValue(split_t));
    } else {
      split->is_(ksplit,
                 std::vector<int64_t>{q_t->sizes().back(), k_t->sizes().back(),
                                      v_t->sizes().back()});
    }
    if (!tryReplacingAllUsesWith(q->output(), split->outputs()[0])) {
      return false;
    }
    if (!tryReplacingAllUsesWith(k->output(), split->outputs()[1])) {
      return false;
    }
    if (!tryReplacingAllUsesWith(v->output(), split->outputs()[2])) {
      return false;
    }
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

}  // namespace optimization
}  // namespace ONNX_NAMESPACE
