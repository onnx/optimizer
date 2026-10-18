// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

// Before:
//   graph.input  = [X, W, B, ...]   (W, B, ... also appear in graph.initializer)
//   graph.initializer = [W, B, ...]
// After:
//   graph.input  = [X]
//   graph.initializer = [W, B, ...]
//
// Models exported under IR version < 4 (e.g. the onnx-caffe2 exports in the
// ONNX Model Zoo) list *every* initializer in graph.input as well. Under ONNX
// semantics an initializer that is also a graph input is an overridable default,
// not a constant, so `Graph::is_constant_initializer` returns false for it and
// `IsConstantTensor` treats the weight as a runtime value. That silently
// disables every constant-dependent pass -- fuse_bn_into_conv,
// fuse_add_bias_into_conv, fuse_matmul_add_bias_into_gemm, constant folding,
// ... -- so such models come out of the optimizer completely unchanged.
//
// This pass removes an initializer's entry from graph.input (keeping the
// initializer itself), which is the standard IR>=4 form and lets the downstream
// passes see the weights as the constants they are. It should run early, before
// the constant-dependent fusions.
//
// this pass can handle the case satisfy all following conditions:
//   condition 1: the value is a graph input
//   condition 2: the value also has a graph initializer of the same name

#include <unordered_set>

#include "onnxoptimizer/pass.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct EliminateInitializerFromInput final : public FullGraphBasedPass {
  explicit EliminateInitializerFromInput()
      : FullGraphBasedPass(PassType::Nop, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "eliminate_initializer_from_input";
  }

  PassAnalysisType getPassAnalysisType() const override {
    return PassAnalysisType::Empty;
  }

  void eliminate_initializer_from_input(Graph& graph) {
    const std::unordered_set<std::string> initializer_names(
        graph.initializer_names().begin(), graph.initializer_names().end());

    // Walk inputs back to front so erasing an entry keeps the remaining
    // indices valid.
    for (int i = static_cast<int>(graph.inputs().size()) - 1; i >= 0; --i) {
      if (initializer_names.count(graph.inputs()[i]->uniqueName()) > 0) {
        graph.eraseInput(i);
      }
    }

    // Subgraphs (If/Loop/Scan bodies) can carry the same legacy encoding.
    for (auto it = graph.begin(); it != graph.end(); ++it) {
      DescendOnGraphAttributesUnconstrained(
          *it, [this](Graph& subgraph) {
            eliminate_initializer_from_input(subgraph);
          });
    }
  }

  std::shared_ptr<PostPassAnalysis> runPass(Graph& graph) override {
    eliminate_initializer_from_input(graph);
    return std::shared_ptr<PostPassAnalysis>(new PostPassAnalysis());
  }
};

}  // namespace optimization
}  // namespace ONNX_NAMESPACE
