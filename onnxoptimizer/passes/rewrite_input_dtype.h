/*
 * SPDX-License-Identifier: Apache-2.0
 */

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include "onnxoptimizer/pass.h"
#include "pass_util.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct RewriteInputDtype final : public FullGraphBasedPass {
  explicit RewriteInputDtype()
      : FullGraphBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::None) {}

  std::string getPassName() const override {
    return "rewrite_input_dtype";
  }

  PassAnalysisType getPassAnalysisType() const override {
    return PassAnalysisType::Empty;
  }

  void rewrite_input_dtype(Graph& graph) {
    std::unordered_set<std::string> initializer_names(
        graph.initializer_names().begin(), graph.initializer_names().end());

    for (auto& value : graph.inputs()) {
      // ignore when input also in initializer
      if (initializer_names.count(value->uniqueName()) > 0 ||
          value->elemType() != ONNX_NAMESPACE::TensorProto_DataType_INT64) {
        continue;
      }
      auto use_list = value->uses();

      Node* cast = graph.create(kCast, 1);
      cast = graph.appendNode(cast);
      cast->i_(kto, static_cast<int64_t>(
                        ONNX_NAMESPACE::TensorProto_DataType_INT64));
      cast->addInput(value);
      cast->output()->setUniqueName(
          ONNX_NAMESPACE::to_string(graph.getNextUnique()));

      for (auto& use : use_list) {
        if (!cast->isBefore(use.user)) {
          cast->moveBefore(use.user);
        }
        use.user->replaceInput(use.offset, cast->output());
      }
      value->setElemType(ONNX_NAMESPACE::TensorProto_DataType_INT32);
    }
  }

  std::shared_ptr<PostPassAnalysis> runPass(Graph& graph) override {
    rewrite_input_dtype(graph);
    return std::shared_ptr<PostPassAnalysis>(new PostPassAnalysis());
  }
};

}  // namespace optimization
}  // namespace ONNX_NAMESPACE