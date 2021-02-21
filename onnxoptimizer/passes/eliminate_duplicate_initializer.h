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

#include "onnx/defs/tensor_util.h"
#include "onnxoptimizer/pass.h"

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
  unsigned int EliminateInitializer(Graph &graph) {
    unsigned int initializers_removed = 0;
    const std::vector<Tensor> &initializers = graph.initializers();
    std::map<std::vector<int64_t>, std::vector<std::string>> init_dict_by_shape;

    // Make {name : Value} map
    std::map<std::string, Value *> input_map;
    for (auto inp : graph.inputs()) {
      if (inp->has_unique_name()) {
        input_map[inp->uniqueName()] = inp;
      }
    }

    std::map<std::string, Value *> output_map;
    for (auto out : graph.outputs()) {
      if (out->has_unique_name()) {
        output_map[out->uniqueName()] = out;
      }
    }

    // Cluster initializers by shape
    for (auto initializer : initializers) {
      if (!initializer.hasName()) {
        continue;
      }
      // Ignore initializer which is not an input
      if (input_map.find(initializer.name()) == input_map.end()) {
        continue;
      }
      // Ignore initializer which is output
      if (output_map.find(initializer.name()) != output_map.end()) {
        continue;
      }
      auto initializers_iter = init_dict_by_shape.find(initializer.sizes());
      if (initializers_iter != init_dict_by_shape.end()) {
        initializers_iter->second.emplace_back(initializer.name());
      } else {
        std::vector<std::string> vec{initializer.name()};
        init_dict_by_shape.insert(
            std::make_pair(std::move(initializer.sizes()), vec));
      }
    }

    for (auto pair : init_dict_by_shape) {
      std::set<std::string> visited;

      // pair.second --> vector initializers with same shape
      // Use iter_i, iter_j to loop it
      for (auto iter_i = pair.second.begin(); iter_i != pair.second.end();
           ++iter_i) {
        if (visited.find(*iter_i) != visited.end()) {
          continue;
        }
        const auto iter_i_initializer = graph.getInitializer(*iter_i);
        if (iter_i_initializer == graph.initializers().end()) {
          continue;
        }
        Tensor i_tensor = *iter_i_initializer;
        Value *i_value = input_map.find(i_tensor.name())->second;

#define DO_COMPARISON(data_type)                                             \
  const std::vector<data_type> i_data = ParseData<data_type>(&i_tensor);     \
  for (auto iter_j = iter_i + 1; iter_j != pair.second.end(); ++iter_j) {    \
    const auto iter_j_initializer = graph.getInitializer(*iter_j);           \
    if (iter_j_initializer == graph.initializers().end()) {                  \
      visited.insert(*iter_j);                                               \
      continue;                                                              \
    }                                                                        \
    Tensor j_tensor = *iter_j_initializer;                                   \
    if (i_tensor.elem_type() != j_tensor.elem_type()) {                      \
      continue;                                                              \
    } else {                                                                 \
      const std::vector<data_type> j_data = ParseData<data_type>(&j_tensor); \
      if (std::equal(i_data.begin(), i_data.end(), j_data.begin())) {        \
        visited.insert(*iter_j);                                             \
        Value *j_value = input_map.find(j_tensor.name())->second;            \
        j_value->replaceAllUsesWith(i_value);                                \
        graph.eraseInitializerAndInput(j_value);                             \
        initializers_removed++;                                              \
      }                                                                      \
    }                                                                        \
  }
#define CASE_DO_COMPARISON(ONNX_DTYPE_SUFFIX, CPP_DTYPE)           \
  case ONNX_NAMESPACE::TensorProto_DataType_##ONNX_DTYPE_SUFFIX: { \
    DO_COMPARISON(CPP_DTYPE)                                       \
    break;                                                         \
  }
        switch (i_tensor.elem_type()) {
          CASE_DO_COMPARISON(FLOAT, float)
          CASE_DO_COMPARISON(DOUBLE, double)
          CASE_DO_COMPARISON(INT32, int32_t)
          CASE_DO_COMPARISON(INT64, int64_t)
          default:
            break;
        }
#undef CASE_DO_COMPARISON
#undef DO_COMPARISON
      }
    }
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
