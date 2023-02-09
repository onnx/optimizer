/*
 * SPDX-License-Identifier: Apache-2.0
 */

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

// Before:
//   X is a tensor with shape=[1, 1, 2, 3, 1, 5, 1]
//   Y = Squeeze(X, axes=[1, 4]) -> shape=[1, 2, 3, 5, 1]
//   Z = Squeeze(Y, axes=[0, 4]) -> shape=[2, 3, 5]
// After:
//   Z = Squeeze(X, axes=[0, 1, 4, 6])
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/logging.h"
#include "onnxoptimizer/passes/pass_util.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct FuseConsecutiveSqueezes final : public PredicateBasedPass {
  explicit FuseConsecutiveSqueezes()
      : PredicateBasedPass(PassType::Fuse, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "fuse_consecutive_squeezes";
  }
  static bool IsAxesAnAttr(const Graph &graph) {
    const int opset_version = getOpsetVersion(graph);
    return opset_version <= 12 && opset_version != 0;
  }

  // modify the vector `composed_axes` such that squeeze by it is equivalent
  // to squeeze by `axes_1` and then by `axes_2`
  static bool compose_squeezes(const Node *input_n, const Node *n,
                               const Graph &graph,
                               std::vector<int64_t> &composed_axes) {
    std::vector<int64_t> axes_1;
    std::vector<int64_t> axes_2;
    if (!GetValueFromAttrOrInput(input_n, kaxes, 1, axes_1) ||
        !GetValueFromAttrOrInput(n, kaxes, 1, axes_2)) {
      return false;
    }

    std::vector<int64_t> &ret = composed_axes;
    ret.clear();
    ret.reserve(axes_1.size() + axes_2.size());
    std::vector<int64_t> sorted_axes_1(axes_1.begin(), axes_1.end());
    std::sort(sorted_axes_1.begin(), sorted_axes_1.end());
    std::copy(sorted_axes_1.begin(), sorted_axes_1.end(),
              std::back_inserter(ret));

    for (int64_t i : axes_2) {
      for (auto iter = sorted_axes_1.begin(); iter != sorted_axes_1.end();
           ++iter) {
        // if current axis 1 - prev_num is bigger than axis 2
        // put axis 2 + prev_num as new axis
        int64_t prev_num = std::distance(sorted_axes_1.begin(), iter);
        if (*iter - prev_num > i) {
          ret.push_back(i + prev_num);
          break;
        }
        // if no current axis 1 - prev_num is bigger than axis 2
        // put axis 2 + prev_num + 1 as new axis
        if (std::next(iter) == sorted_axes_1.end()) {
          ret.push_back(i + prev_num + 1);
        }
      }
    }
    std::sort(ret.begin(), ret.end());
    return true;
  }

  bool patternMatchPredicate(Node *node) override {
    return node->kind() == kSqueeze &&
           node->inputs()[0]->node()->kind() == kSqueeze;
  }
  bool runTransform(Node *n, Graph &graph,
                    NodeDestroyType &destroy_current) override {
    auto orig_input = n->inputs()[0];
    std::vector<int64_t> rs;
    bool success = compose_squeezes(orig_input->node(), n, graph, rs);
    if (!success) {
      return false;
    }
    n->replaceInput(0, orig_input->node()->inputs()[0]);
    if (orig_input->uses().size() == 0) {
      orig_input->node()->destroy();
    }
    if (IsAxesAnAttr(graph)) {
      n->is_(kaxes, std::move(rs));
    } else {
      Tensor t;
      t.sizes().push_back(rs.size());
      t.int64s() = rs;
      t.elem_type() = TensorProto_DataType_INT64;
      auto axes_v = n->inputs()[1];
      Value *tv = graph.addInitializerAndCreateValue(t);
      n->replaceInput(1, tv);
      if (axes_v->uses().size() == 0) {
        if (axes_v->node()->kind() == kConstant) {
          axes_v->node()->destroy();
        } else {
          graph.eraseInitializerAndInput(axes_v);
        }
      }
    }
    destroy_current = NodeDestroyType::DestroyZero;
    return true;
  }
};

}  // namespace optimization
}  // namespace ONNX_NAMESPACE
