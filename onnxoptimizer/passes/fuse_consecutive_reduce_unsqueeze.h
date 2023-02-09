/*
 * SPDX-License-Identifier: Apache-2.0
 */

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include "onnx/defs/tensor_proto_util.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/logging.h"
#include "onnxoptimizer/passes/pass_util.h"

namespace ONNX_NAMESPACE {
namespace optimization {

const std::unordered_set<NodeKind> reduction_operators{
    kReduceL1,   kReduceL2,  kReduceLogSum, kReduceLogSumExp, kReduceMax,
    kReduceMean, kReduceMin, kReduceProd,   kReduceSum,       kReduceSumSquare};

struct FuseConsecutiveReduceUnsqueeze final : public PredicateBasedPass {
  explicit FuseConsecutiveReduceUnsqueeze()
      : PredicateBasedPass(PassType::Fuse, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "fuse_consecutive_reduce_unsqueeze";
  }

  bool patternMatchPredicate(Node *node) override {
    // check that the current node is of type Unsqueeze and has defined axes
    bool cur_node_check = node->kind() == kUnsqueeze;
    if (cur_node_check) {
      Node *prev_node = node->inputs()[0]->node();
      // check that the previous node a reduction operator and has defined
      // axes/keepdims
      bool reduction_node_check = reduction_operators.find(prev_node->kind()) !=
                                      reduction_operators.end() &&
                                  prev_node->hasAttribute(kkeepdims);
      if (reduction_node_check) {
        // insure that keepdims is set to false currently
        return prev_node->i(kkeepdims) == 0;
      }
    }
    return false;
  }
  bool runTransform(Node *node, Graph &graph,
                    NodeDestroyType &destroy_current) override {
    Node *reduction_op = node->inputs()[0]->node();
    // This pass will modify reduction_op, so it must have only one user.
    if (reduction_op->output()->uses().size() != 1) {
      return false;
    }
    std::vector<int64_t> axes;
    std::vector<int64_t> prev_axes;
    if (!GetValueFromAttrOrInput(node, kaxes, 1, axes) ||
        !GetValueFromAttrOrInput(reduction_op, kaxes, 1, prev_axes) ||
        axes != prev_axes) {
      return false;
    }

    const bool replacing_success =
        tryReplacingAllUsesWith(node->output(), node->inputs()[0]);
    if (replacing_success) {
      // set keepdims flag to be true
      reduction_op->i_(kkeepdims, 1);
      // remove unnecessary unsqueeze
      reduction_op->output()->setSizes(node->output()->sizes());
      reduction_op->output()->setElemType(node->output()->elemType());
      destroy_current = NodeDestroyType::DestroyOne;
      return true;
    } else {
      return false;
    }
  }
};

}  // namespace optimization
}  // namespace ONNX_NAMESPACE
