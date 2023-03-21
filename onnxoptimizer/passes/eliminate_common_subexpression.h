/*
 * SPDX-License-Identifier: Apache-2.0
 */

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.
#pragma once

#include <unordered_map>

#include "onnx/defs/tensor_util.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/cse_util.h"
#include "onnxoptimizer/passes/logging.h"
#include "onnxoptimizer/passes/pass_util.h"
#include "onnxoptimizer/passes/string_utils.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct EliminateCommonSubexpression final : public FullGraphBasedPass {
  explicit EliminateCommonSubexpression()
      : FullGraphBasedPass(PassType::Nop, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override {
    return "eliminate_common_subexpression";
  }
  PassAnalysisType getPassAnalysisType() const override {
    return PassAnalysisType::CountBased;
  }

  unsigned int EliminateCommonSubexpressions(Graph &graph) {
    auto node_list = graph.nodes();
    unsigned int cse_removed = 0;
    std::unordered_map<Node *, Node *, CSENodeHash, CSEEqual> hash_map;
    for (auto it = node_list.begin(); it != node_list.end(); ++it) {
      auto node = *it;
      auto kind = node->kind();
      if (!node->hasUses() || !IsSupportedByCSE(node)) {
        continue;
      }
      VLOG(2) << Str("kind: ", kind.toString(), ", ", node->name(),
                     " is processing");
      if (hash_map.find(node) == hash_map.end()) {
        hash_map[node] = node;
      } else {
        auto other = hash_map.at(node);
        auto outputs = other->outputs();
        auto replaced_outputs = node->outputs();
        for (int i = 0; i < outputs.size(); ++i) {
          if (tryReplacingAllUsesWith(replaced_outputs[i], outputs[i])) {
            VLOG(1) << Str("kind: ", kind.toString(), ", ", node->name(), " [",
                           i, "] output has been replaced by ", other->name());
            cse_removed++;
          }
        }
      }
    }
    return cse_removed;
  }

  std::shared_ptr<PostPassAnalysis> runPass(Graph &graph) override {
    auto cse_removed = this->EliminateCommonSubexpressions(graph);
    VLOG(1) << Str("cse_removed count: ", cse_removed);
    return std::shared_ptr<PostPassAnalysis>(
        new CountBasedPassAnalysis(this, cse_removed, false, false));
  }
};
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
