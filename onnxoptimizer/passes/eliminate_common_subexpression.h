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
#include "onnxoptimizer/passes/pass_util.h"
#include "onnxoptimizer/passes/string_utils.h"

namespace ONNX_NAMESPACE {
namespace optimization {
namespace details {
enum class OpType : int {
  UNARY,
  BINARY,
  TERNARY,
};
const static std::unordered_map<NodeKind, OpType> cse_supported_nodes{
    {"Abs"_sym, OpType::UNARY},
    {"Cos"_sym, OpType::UNARY},
    {"Acosh"_sym, OpType::UNARY},
    {kAdd, OpType::BINARY},
    {"And"_sym, OpType::BINARY},
    {"Asin"_sym, OpType::UNARY},
    {"Asinh"_sym, OpType::UNARY},
    {"Atan"_sym, OpType::UNARY},
    {"Atanh"_sym, OpType::UNARY},

    {"BitwiseAnd"_sym, OpType::BINARY},
    {"BitwiseNot"_sym, OpType::UNARY},
    {"BitwiseOr"_sym, OpType::BINARY},
    {"BitwiseXor"_sym, OpType::BINARY},
    {"Ceil"_sym, OpType::UNARY},
    {"Clip"_sym, OpType::TERNARY},
    {"Cos"_sym, OpType::UNARY},
    {"Det"_sym, OpType::UNARY},
    {kDiv, OpType::BINARY},
    {"Equal"_sym, OpType::BINARY},
    {"Erf"_sym, OpType::UNARY},
    {kExp, OpType::UNARY},
    {kExpand, OpType::BINARY},
    {"GlobalAveragePool"_sym, OpType::UNARY},
    {"GlobalMaxPool"_sym, OpType::UNARY},
    {kGreater, OpType::BINARY},
    {"GreaterOrEqual"_sym, OpType::BINARY},
    {"HardSwish"_sym, OpType::UNARY},
    {kIdentity, OpType::UNARY},
    {kLess, OpType::BINARY},
    {"kLessOrEqual"_sym, OpType::BINARY},
    {kLog, OpType::UNARY},
    {kMatMul, OpType::BINARY},
    {kMul, OpType::BINARY},
    {kNeg, OpType::UNARY},
    {"NonZero"_sym, OpType::UNARY},
    {"Not"_sym, OpType::UNARY},
    {"Or"_sym, OpType::BINARY},
    {kPRelu, OpType::BINARY},
    {kPow, OpType::BINARY},
    {"Relu"_sym, OpType::UNARY},
    {kSigmoid, OpType::UNARY},
    {"Sign"_sym, OpType::UNARY},
    {"Sin"_sym, OpType::UNARY},
    {"Sinh"_sym, OpType::UNARY},
    {"Softplus"_sym, OpType::UNARY},
    {kSub, OpType::BINARY}

};
}  // namespace details

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
    std::unordered_map<std::string, Node *> hash_map;
    for (auto it = node_list.begin(); it != node_list.end(); ++it) {
      auto node = *it;
      auto kind = node->kind();
      if (details::cse_supported_nodes.count(kind) == 0 || !node->hasUses()) {
        continue;
      }
      auto optype = details::cse_supported_nodes.at(kind);
      const auto inputs = node->inputs();
      if (inputs.size() != static_cast<int>(optype) + 1) {
        continue;
      }
      std::string name = kind.toString();
      for (auto *input : inputs) {
        name = Str(name, "+", input->uniqueName());
      }
      if (hash_map.count(name) == 0) {
        hash_map[name] = node;
      } else {
        auto other = hash_map.at(name);
        if (tryReplacingAllUsesWith(node->output(), other->output())) {
          VLOG(1) << Str("kind: ", kind.toString(), ", ", node->name(),
                         " has been replaced by ", other->name());
          cse_removed++;
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
