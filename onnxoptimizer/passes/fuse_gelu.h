// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

// Fuse the erf-based (exact) GELU decomposition into a single Gelu operator,
// mirroring the GELU fusion performed by onnxslim.
//
// It matches the subgraph
//   div = x / sqrt(2)
//   erf = Erf(div)
//   add = erf + 1
//   mul = x * add
//   out = mul * 0.5
// and rewrites it to
//   out = Gelu(x)          // approximate = "none" (the default)
//
// The multiplication by 0.5 and the multiplication by (1 + erf(...)) are both
// commutative, so either operand ordering is accepted. The division, however,
// is not commutative: `x` must be the numerator.
//
// The standard-domain Gelu operator was introduced in opset 20, so the fusion
// is only applied when the model targets opset >= 20; otherwise the resulting
// node would be invalid.

#include <cmath>

#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct FuseGelu final : public PredicateBasedPass {
  explicit FuseGelu()
      : PredicateBasedPass(PassType::Fuse, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "fuse_gelu";
  }

  // Returns true when `v` is a constant tensor holding a single float or double
  // element approximately equal to `target`.
  static bool IsScalarCloseTo(const Value* v, double target) {
    float f;
    if (FetchSoleValueOfTensor<float>(v, f)) {
      return std::abs(static_cast<double>(f) - target) < 1e-4;
    }
    double d;
    if (FetchSoleValueOfTensor<double>(v, d)) {
      return std::abs(d - target) < 1e-4;
    }
    return false;
  }

  // The intermediate nodes of the pattern must feed only the next node in the
  // chain, otherwise the decomposition is shared and cannot be safely fused.
  static bool SingleUse(const Node* n) {
    return n->output()->uses().size() == 1;
  }

  bool patternMatchPredicate(Node* node) override {
    // Anchor on the final `Mul` (out = mul * 0.5).
    return CheckKind(node, kMul) && node->inputs().size() == 2;
  }

  bool runTransform(Node* mul1, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    destroy_current = NodeDestroyType::DestroyZero;

    // The standard Gelu op lives in the default domain from opset 20 onwards.
    if (getOpsetVersion(graph) < 20) {
      return false;
    }

    // out = Mul(mul0, 0.5): locate the 0.5 scalar and the inner `Mul` node.
    Node* mul0 = nullptr;
    for (int i = 0; i < 2; ++i) {
      Value* a = mul1->input(i);
      Value* b = mul1->input(1 - i);
      if (IsScalarCloseTo(b, 0.5) && CheckKind(a, kMul) &&
          a->node()->inputs().size() == 2 && SingleUse(a->node())) {
        mul0 = a->node();
        break;
      }
    }
    if (!mul0) {
      return false;
    }

    // mul0 = Mul(x, add): locate the `Add` node and the data input `x`.
    Node* add = nullptr;
    Value* x = nullptr;
    for (int i = 0; i < 2; ++i) {
      Value* a = mul0->input(i);
      Value* b = mul0->input(1 - i);
      if (CheckKind(a, kAdd) && a->node()->inputs().size() == 2 &&
          SingleUse(a->node())) {
        add = a->node();
        x = b;
        break;
      }
    }
    if (!add) {
      return false;
    }

    // add = Add(erf, 1): locate the `Erf` node and check the addend.
    Node* erf = nullptr;
    for (int i = 0; i < 2; ++i) {
      Value* a = add->input(i);
      Value* b = add->input(1 - i);
      if (CheckKind(a, "Erf") && a->node()->inputs().size() == 1 &&
          SingleUse(a->node()) && IsScalarCloseTo(b, 1.0)) {
        erf = a->node();
        break;
      }
    }
    if (!erf) {
      return false;
    }

    // erf = Erf(div), div = Div(x, sqrt(2)). Div is not commutative, so `x`
    // must be the numerator and the sqrt(2) constant the denominator.
    Node* div = erf->input(0)->node();
    if (!CheckKind(div, kDiv) || div->inputs().size() != 2 || !SingleUse(div)) {
      return false;
    }
    if (div->input(0) != x ||
        !IsScalarCloseTo(div->input(1), std::sqrt(2.0))) {
      return false;
    }

    Node* gelu = graph.create(Symbol("Gelu"), 1);
    gelu->addInput(x);
    gelu->output()->copyMetadata(mul1->output());
    gelu->insertBefore(mul1);

    if (!tryReplacingAllUsesWith(mul1->output(), gelu->output())) {
      gelu->destroy();
      return false;
    }
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

}  // namespace optimization
}  // namespace ONNX_NAMESPACE
