/*
 * SPDX-License-Identifier: Apache-2.0
 */

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.
#pragma once

#include <unordered_map>

#include "onnx/defs/tensor_util.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/logging.h"
#include "onnxoptimizer/passes/node_hash.h"
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

  unsigned int EliminateCSE(Graph &graph) {
    auto node_list = graph.nodes();
    unsigned int cse_removed = 0;
    std::unordered_map<size_t, Node *> hash_map;
    for (auto it = node_list.begin(); it != node_list.end(); ++it) {
      auto node = *it;
      auto kind = node->kind();
      if (!node->hasUses() || !IsSupportedHash(node)) {
        continue;
      }
      std::size_t hash = std::hash<Node>()(*node);
      VLOG(1) << Str(node->name(), " hash: ", hash);
      if (hash_map.count(hash) == 0) {
        hash_map[hash] = node;
      } else {
        auto other = hash_map.at(hash);
        auto outputs = other->outputs();
        auto replaced_outputs = node->outputs();
        ONNX_ASSERT(*other == *node);
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
    auto cse_removed = this->EliminateCSE(graph);
    VLOG(1) << Str("cse_removed count: ", cse_removed);
    return std::shared_ptr<PostPassAnalysis>(
        new CountBasedPassAnalysis(this, cse_removed, false, false));
  }
};
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
