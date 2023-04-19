/*
 * SPDX-License-Identifier: Apache-2.0
 */

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.
#pragma once

// Before:
//   A, B are in the initializer list, and A is equal to B
//   E = Add(D, A)
//   F = Add(F, B)
//   G = Add(E, F)
// After:
//   A is in the initializer list
//   E = Add(D, A)
//   F = Add(F, A)
//   G = Add(E, F)
//
// NOTE: ONNX IR has an bug that an initializer must also
// be an graph input. Currently we are using a workaround
// that adds initializers to inputs before optimization
// and removes the added initializers from inputs after
// optimization. That makes us cannot distinguish the
// initializers really in the inputs and the initializers
// not in the inputs. While only the latter can be eliminated,
// we eliminate all duplicated initializers instead. That
// may cause unexpected behavior in some rare cases.

#include <unordered_map>
#include <unordered_set>

#include "onnx/defs/tensor_util.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/cse_util.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct EliminateDuplicateInitializer final : public FullGraphBasedPass {
  explicit EliminateDuplicateInitializer()
      : FullGraphBasedPass(PassType::Nop, PassEfficiency::Complete,
                           PassOptimizationType::Memory) {}
  std::string getPassName() const override {
    return "eliminate_duplicate_initializer";
  }
  PassAnalysisType getPassAnalysisType() const override {
    return PassAnalysisType::CountBased;
  }

  Value *findInitializerValueByName(Node *initializer_node,
                                    const std::string &name) {
    for (size_t i = 0; i < initializer_node->outputs().size(); i++) {
      if (initializer_node->outputs()[i]->uniqueName() == name) {
        return initializer_node->outputs()[i];
      }
    }
    return nullptr;
  }

  unsigned int EliminateInitializer(Graph &graph) {
    unsigned int initializers_removed = 0;
    const std::vector<Tensor> &initializers = graph.initializers();

    // Make {name : Value} map
    std::unordered_set<std::string> input_set;
    for (auto inp : graph.inputs()) {
      if (inp->has_unique_name()) {
        input_set.emplace(inp->uniqueName());
      }
    }

    std::unordered_set<std::string> output_set;
    for (auto out : graph.outputs()) {
      if (out->has_unique_name()) {
        output_set.emplace(out->uniqueName());
      }
    }
    std::unordered_map<const Tensor *, std::string, CSETensorHash, CSETensorEqual>
        initializer_map;
    std::vector<std::pair<std::string, std::string>> replaced_table;
    for (const auto& initializer : initializers) {
      if (!initializer.hasName()) {
        continue;
      }
      const auto &name = initializer.name();
      // Ignore initializer which is an input
      if (input_set.find(name) != input_set.end()) {
        continue;
      }
      // Ignore initializer which is output
      if (output_set.find(name) != output_set.end()) {
        continue;
      }
      if (initializer_map.count(&initializer) == 0) {
        initializer_map[&initializer] = name;
      } else {
        replaced_table.emplace_back(
            std::make_pair(name, initializer_map.at(&initializer)));
      }
    }
    if (replaced_table.empty()) {
      return initializers_removed;
    }
    // workaround to  fetch initializer_node_ pointer in graph
    Tensor dummy_tensor;
    dummy_tensor.setName(ONNX_NAMESPACE::to_string(graph.getNextUnique()));
    Node *initializer_node =
        graph.addInitializerAndCreateValue(dummy_tensor)->node();
    VLOG(1) << Str("====== Graph: ", graph.name(), "=====");
    for (const auto &p : replaced_table) {
      VLOG(1) << Str("<", p.first, ",", p.second, ">");
      Value *old_value = findInitializerValueByName(initializer_node, p.first);
      Value *new_value = findInitializerValueByName(initializer_node, p.second);
      if (!old_value || !new_value) {
        continue;
      }
      old_value->replaceAllUsesWith(new_value);
      graph.eraseInitializerAndInput(old_value);
      initializers_removed++;
    }
    VLOG(1) << Str("====== Graph: ", graph.name(),
                   "=====, removed: ", initializers_removed);
    graph.eraseInitializer(dummy_tensor.name());
    return initializers_removed;
  }
  std::shared_ptr<PostPassAnalysis> runPass(Graph &graph) override {
    auto initializers_removed = this->EliminateInitializer(graph);
    return std::shared_ptr<PostPassAnalysis>(
        new CountBasedPassAnalysis(this, initializers_removed, false, false));
  }
};
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
