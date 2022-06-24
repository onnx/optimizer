/*
 * SPDX-License-Identifier: Apache-2.0
 */

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

// fetch input/output pattern from environment variable
// OPTIMIZER_RENAME_INPUT_PATTERN(default: input_%d) and
// OPTIMIZER_RENAME_OUTPUT_PATTERN(default: output_%d).

#include "onnxoptimizer/pass.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct RenameInputOutput final : public FullGraphBasedPass {
  explicit RenameInputOutput()
      : FullGraphBasedPass(PassType::Other, PassEfficiency::Complete,
                           PassOptimizationType::None) {}

  std::string getPassName() const override {
    return "rename_input_output";
  }

  PassAnalysisType getPassAnalysisType() const override {
    return PassAnalysisType::Empty;
  }

  std::vector<std::string> fetchPatternFromEnv() {
    auto fetch_env = [](const std::string& env) -> std::string {
      const auto* res = std::getenv(env.c_str());
      return res ? std::string(res) : std::string{};
    };
    auto split_pattern =
        [](const std::string& s, const std::string& pattern,
           const std::string& default_s) -> std::vector<std::string> {
      std::vector<std::string> res(2);
      const std::string* tmp = &s;
      auto n = tmp->find(pattern);
      if (n == std::string::npos) {
        tmp = &default_s;
        n = tmp->find(pattern);
      }

      res[0] = tmp->substr(0, n);
      res[1] = tmp->substr(n + pattern.size());
      return res;
    };
    const std::string pattern{"%d"};
    const std::vector<std::string> envs{"OPTIMIZER_RENAME_INPUT_PATTERN",
                                        "OPTIMIZER_RENAME_OUTPUT_PATTERN"};
    const std::vector<std::string> default_str{"input_%d", "output_%d"};
    std::vector<std::string> result;
    for (int i = 0; i < 2; ++i) {
      auto env_str = fetch_env(envs[i]);
      auto split_str = split_pattern(env_str, pattern, default_str[i]);
      std::copy(split_str.begin(), split_str.end(), std::back_inserter(result));
    }
    return result;
  }

  void rename_input_output(Graph& graph) {
    std::unordered_set<std::string> initializer_names(
        graph.initializer_names().begin(), graph.initializer_names().end());

    const auto rename_patterns = fetchPatternFromEnv();

    for (int i = 0; i < graph.inputs().size(); ++i) {
      auto& value = graph.inputs()[i];
      // ignore when input also in initializer
      if (initializer_names.count(value->uniqueName()) > 0) {
        continue;
      }
      const std::string current_name =
          rename_patterns[0] + std::to_string(i) + rename_patterns[1];

      value->setUniqueName(current_name);
    }

    for (int i = 0; i < graph.outputs().size(); ++i) {
      auto& value = graph.outputs()[i];
      const std::string current_name =
          rename_patterns[2] + std::to_string(i) + rename_patterns[3];
      value->setUniqueName(current_name);
    }
  }

  std::shared_ptr<PostPassAnalysis> runPass(Graph& graph) override {
    rename_input_output(graph);
    return std::shared_ptr<PostPassAnalysis>(new PostPassAnalysis());
  }
};

}  // namespace optimization
}  // namespace ONNX_NAMESPACE
