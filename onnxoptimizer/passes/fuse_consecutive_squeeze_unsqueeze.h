// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

// Eliminate a Squeeze immediately followed by an inverse Unsqueeze (or an
// Unsqueeze immediately followed by an inverse Squeeze) when the two operators
// cancel each other out.
//
// Squeeze then Unsqueeze:
//   X has shape [2, 1, 3, 1]
//   Y = Squeeze(X, axes=[1, 3])   -> shape [2, 3]
//   Z = Unsqueeze(Y, axes=[1, 3]) -> shape [2, 1, 3, 1]
// After:
//   Z = X
//
// Unsqueeze then Squeeze:
//   X has shape [2, 3]
//   Y = Unsqueeze(X, axes=[1, 3]) -> shape [2, 1, 3, 1]
//   Z = Squeeze(Y, axes=[1, 3])   -> shape [2, 3]
// After:
//   Z = X
#include <algorithm>

#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/logging.h"
#include "onnxoptimizer/passes/pass_util.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct FuseConsecutiveSqueezeUnsqueeze final : public PredicateBasedPass {
  explicit FuseConsecutiveSqueezeUnsqueeze()
      : PredicateBasedPass(PassType::Nop, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "fuse_consecutive_squeeze_unsqueeze";
  }

  bool patternMatchPredicate(Node *node) override {
    return CheckKind(node, kUnsqueeze, 0, kSqueeze) ||
           CheckKind(node, kSqueeze, 0, kUnsqueeze);
  }

  // Normalize (possibly negative) axes to non-negative indices with respect to
  // `rank`, then sort. Returns false when `rank` is unknown (< 0) but a
  // negative axis is present, since we cannot resolve it safely.
  static bool NormalizeAxes(std::vector<int64_t> &axes, int64_t rank) {
    for (auto &axis : axes) {
      if (axis < 0) {
        if (rank < 0) {
          return false;
        }
        axis += rank;
      }
    }
    std::sort(axes.begin(), axes.end());
    return true;
  }

  // For a canceling Squeeze/Unsqueeze pair both axes lists are indexed against
  // the same "outer" rank: the Squeeze's input rank, which equals the
  // Unsqueeze's output rank. Recover it from whichever value carries shape
  // info; returns -1 when it is unknown.
  static int64_t ReferenceRank(const Node *squeeze, const Node *unsqueeze) {
    if (squeeze->input(0)->has_sizes()) {
      return static_cast<int64_t>(squeeze->input(0)->sizes().size());
    }
    if (unsqueeze->output()->has_sizes()) {
      return static_cast<int64_t>(unsqueeze->output()->sizes().size());
    }
    return -1;
  }

  bool runTransform(Node *node, Graph &graph,
                    NodeDestroyType &destroy_current) override {
    destroy_current = NodeDestroyType::DestroyZero;
    Node *prev = PrevNode(node, 0);

    std::vector<int64_t> axes, prev_axes;
    if (!GetValueFromAttrOrInput(node, kaxes, 1, axes) ||
        !GetValueFromAttrOrInput(prev, kaxes, 1, prev_axes)) {
      return false;
    }

    const Node *squeeze = node->kind() == kSqueeze ? node : prev;
    const Node *unsqueeze = node->kind() == kUnsqueeze ? node : prev;
    const int64_t rank = ReferenceRank(squeeze, unsqueeze);
    if (!NormalizeAxes(axes, rank) || !NormalizeAxes(prev_axes, rank)) {
      return false;
    }

    // The pair cancels out only when both operators touch exactly the same set
    // of axes. Since Squeeze/Unsqueeze only remove/insert size-1 dimensions,
    // matching axes make the two nodes a no-op.
    if (axes != prev_axes) {
      return false;
    }

    if (!tryReplacingAllUsesWith(node->output(), prev->input(0))) {
      return false;
    }
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

}  // namespace optimization
}  // namespace ONNX_NAMESPACE
