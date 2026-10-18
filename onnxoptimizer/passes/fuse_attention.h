// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include "onnx/defs/tensor_util.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"

namespace ONNX_NAMESPACE {
namespace optimization {

// Fuse the sub-graph that `torch.onnx.export` (the TorchScript exporter)
// produces for `torch.nn.functional.scaled_dot_product_attention` into a
// single ONNX `Attention` operator (opset 23+).
//
// The exported pattern (working forward) looks like:
//
//   key_t   = Transpose(K, perm=[..., -1, -2])   // swap last two axes
//   q_s     = Mul(Q,     scale_q)                // scale_q == sqrt(scale)
//   k_s     = Mul(key_t, scale_k)                // scale_k == sqrt(scale)
//   qk      = MatMul(q_s, k_s)
//   biased  = Add(qk, attn_mask)                 // optional (mask / causal)
//   weights = Softmax(biased, axis=-1)
//   out     = MatMul(weights, V)
//
// `scale_q` and `scale_k` are either
//   * two constant scalars (when an explicit `scale=` was passed), or
//   * two `Sqrt` nodes fed from the default `1 / sqrt(head_size)` sub-graph
//     that torch derives from `Shape(Q)` (when `scale` was left to default).
//
// Because `Attention` internally computes `softmax(scale * (Q @ K^T) + mask) @
// V` with a default `scale` of `1 / sqrt(head_size)`, the whole sub-graph is
// equivalent to `Attention(Q, K, V, attn_mask)` where the `scale` attribute is
// set to `scale_q * scale_k` for the explicit-scale case and left unset (i.e.
// the default) for the derived-scale case.
struct FuseAttention final : public PredicateBasedPass {
  explicit FuseAttention()
      : PredicateBasedPass(PassType::Fuse, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "fuse_attention";
  }

  // The anchor is the final `MatMul(Softmax(...), V)`.
  bool patternMatchPredicate(Node* node) override {
    return CheckKind(node, kMatMul) && CheckKind(node->input(0), kSoftmax);
  }

  // Returns true when `axis` refers to the last dimension of a `rank`-D tensor.
  static bool IsLastAxis(int64_t axis, int64_t rank) {
    if (rank <= 0) {
      // Rank is unknown: only the canonical `-1` is accepted.
      return axis == -1;
    }
    if (axis < 0) {
      axis += rank;
    }
    return axis == rank - 1;
  }

  // Checks that `perm` swaps the last two axes and leaves the rest untouched,
  // e.g. [0, 1, 3, 2] for a rank-4 tensor.
  static bool IsLastTwoAxesSwap(const std::vector<int64_t>& perm) {
    const int64_t rank = static_cast<int64_t>(perm.size());
    if (rank < 2) {
      return false;
    }
    for (int64_t i = 0; i < rank - 2; ++i) {
      if (perm[i] != i) {
        return false;
      }
    }
    return perm[rank - 2] == rank - 1 && perm[rank - 1] == rank - 2;
  }

  // Recognizes the default `1 / sqrt(head_size)` scale sub-graph that torch
  // derives from the last dimension of `q`:
  //   Cast(Div(Constant(1), Sqrt(Cast(Slice(Shape(q), starts=[-1], ...)))))
  // The leading/trailing `Cast`s are optional (they only appear for some
  // dtypes). `q` must be the query fed into the fused attention.
  static bool IsDefaultScaleSource(const Value* x, const Value* q) {
    const Node* n = x->node();
    if (CheckKind(n, kCast)) {
      n = n->input(0)->node();
    }
    if (!CheckKind(n, kDiv)) {
      return false;
    }
    float numerator;
    if (!FetchSoleValueOfTensor(n->input(0), numerator) || numerator != 1.0f) {
      return false;
    }
    const Node* sqrt = n->input(1)->node();
    if (!CheckKind(sqrt, kSqrt)) {
      return false;
    }
    const Node* dim = sqrt->input(0)->node();
    if (CheckKind(dim, kCast)) {
      dim = dim->input(0)->node();
    }
    if (!CheckKind(dim, kSlice)) {
      return false;
    }
    // The slice must select the last dimension of the shape.
    int64_t start;
    if (dim->inputs().size() < 2 ||
        !FetchSoleIntValueOfTensor(dim->input(1), start) || start != -1) {
      return false;
    }
    const Node* shape = dim->input(0)->node();
    return CheckKind(shape, "Shape") && shape->input(0) == q;
  }

  bool runTransform(Node* n, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    destroy_current = NodeDestroyType::DestroyZero;

    // `Attention` was introduced in opset 23; refuse to emit it otherwise.
    if (getOpsetVersion(graph) < 23) {
      return false;
    }

    // out = MatMul(weights, V)
    Node* softmax = n->input(0)->node();
    if (n->input(0)->uses().size() != 1) {
      return false;
    }
    Value* value = n->input(1);

    // weights = Softmax(biased, axis=-1)
    {
      int64_t rank =
          softmax->input(0)->has_sizes()
              ? static_cast<int64_t>(softmax->input(0)->sizes().size())
              : -1;
      int64_t axis = GetValueFromAttrWithDefault(softmax, kaxis, (int64_t)-1);
      if (!IsLastAxis(axis, rank)) {
        return false;
      }
    }
    if (softmax->input(0)->uses().size() != 1) {
      return false;
    }

    // biased = Add(qk, attn_mask)  (optional)
    Node* score = softmax->input(0)->node();
    Value* attn_mask = nullptr;
    Node* qk = nullptr;
    if (CheckKind(score, kMatMul)) {
      qk = score;
    } else if (CheckKind(score, kAdd)) {
      for (size_t i = 0; i < 2; ++i) {
        if (CheckKind(score->input(i), kMatMul)) {
          qk = score->input(i)->node();
          attn_mask = score->input(1 - i);
          break;
        }
      }
      if (qk == nullptr || score->output()->uses().size() != 1 ||
          qk->output()->uses().size() != 1) {
        return false;
      }
    } else {
      return false;
    }

    // qk = MatMul(q_scaled, k_scaled)
    Node* mul_q = qk->input(0)->node();
    Node* mul_k = qk->input(1)->node();
    if (!CheckKind(mul_q, kMul) || !CheckKind(mul_k, kMul) ||
        qk->input(0)->uses().size() != 1 || qk->input(1)->uses().size() != 1) {
      return false;
    }

    // k_scaled = Mul(Transpose(K), scale_k)
    Node* transpose = nullptr;
    Value* scale_k = nullptr;
    for (size_t i = 0; i < 2; ++i) {
      if (CheckKind(mul_k->input(i), kTranspose)) {
        transpose = mul_k->input(i)->node();
        scale_k = mul_k->input(1 - i);
        break;
      }
    }
    if (transpose == nullptr || transpose->output()->uses().size() != 1) {
      return false;
    }
    std::vector<int64_t> perm;
    if (!GetValueFromAttr(transpose, kperm, perm) || !IsLastTwoAxesSwap(perm)) {
      return false;
    }
    Value* key = transpose->input(0);

    // q_scaled = Mul(Q, scale_q): the scale operand is a constant scalar or a
    // Sqrt node, the other operand is the query.
    auto is_scale_operand = [](const Value* v) {
      float tmp;
      return CheckKind(v, kSqrt) || FetchSoleValueOfTensor(v, tmp);
    };
    Value* query = nullptr;
    Value* scale_q = nullptr;
    if (is_scale_operand(mul_q->input(1))) {
      query = mul_q->input(0);
      scale_q = mul_q->input(1);
    } else if (is_scale_operand(mul_q->input(0))) {
      query = mul_q->input(1);
      scale_q = mul_q->input(0);
    } else {
      return false;
    }

    // Resolve the effective scale applied to Q @ K^T.
    bool has_scale = false;
    float scale = 0.0f;
    float scale_q_val;
    float scale_k_val;
    if (FetchSoleValueOfTensor(scale_q, scale_q_val) &&
        FetchSoleValueOfTensor(scale_k, scale_k_val)) {
      // Explicit scale: torch emits sqrt(scale) constants on both branches.
      has_scale = true;
      scale = scale_q_val * scale_k_val;
    } else if (CheckKind(scale_q, kSqrt) && CheckKind(scale_k, kSqrt) &&
               scale_q->node()->input(0) == scale_k->node()->input(0) &&
               IsDefaultScaleSource(scale_q->node()->input(0), query)) {
      // Default scale: 1 / sqrt(head_size), which is `Attention`'s default.
      has_scale = false;
    } else {
      return false;
    }

    Node* attention = graph.create("Attention"_sym, 1);
    attention->addInput(query);
    attention->addInput(key);
    attention->addInput(value);
    if (attn_mask != nullptr) {
      attention->addInput(attn_mask);
    }
    if (has_scale) {
      attention->f_(kscale, scale);
    }
    attention->insertBefore(n);

    if (!tryReplacingAllUsesWith(n->output(), attention->output())) {
      return false;
    }
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

}  // namespace optimization
}  // namespace ONNX_NAMESPACE
