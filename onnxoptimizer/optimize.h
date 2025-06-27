/*
 * SPDX-License-Identifier: Apache-2.0
 */

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include "onnx/common/ir.h"
#include "onnx/common/ir_pb_converter.h"
#include "onnx/proto_utils.h"

#include "onnxoptimizer/pass_manager.h"
#include "onnxoptimizer/pass_registry.h"

#include "vector"

namespace ONNX_NAMESPACE {
namespace optimization {

struct Optimizer {
  static GlobalPassRegistry passes;

 public:
  Optimizer(const std::vector<std::string> &names, const bool fixed_point);
  ~Optimizer();

  ModelProto optimize(const ModelProto &_mp_in) {
    ModelProto mp_in = _mp_in;
    if (mp_in.ir_version() == 3) {
      // Upgrade ir_version to 4 so that initializer can be not in input
      mp_in.set_ir_version(4);
    }
    std::shared_ptr<Graph> g(ImportModelProto(mp_in));

    if (g.get() == nullptr) {
      std::cerr << "Warning: onnx optimizer is unable to parse input model. "
                << "(The IR version of the ONNX model may be too old.)"
                << std::endl;
      // If we can't parse the file, just return the input.
      return mp_in;
    }

    ModelProto mp_out = PrepareOutput(mp_in);
    this->pass_manager->run(*g);
    ExportModelProto(&mp_out, g);
    return mp_out;
  }

 private:
  std::shared_ptr<PassManager> pass_manager;

  ModelProto AddInitializerToInput(const ModelProto &original_model) {
    ModelProto model = original_model;
    std::vector<std::string> input_names;
    for (const auto &x : model.graph().input()) {
      input_names.push_back(x.name());
    }
    for (const auto &x : model.graph().initializer()) {
      if (std::find(input_names.begin(), input_names.end(), x.name()) ==
          input_names.end()) {
        auto *value_info = model.mutable_graph()->add_input();
        value_info->set_name(x.name());
        TypeProto *type = value_info->mutable_type();
        auto *tensor = type->mutable_tensor_type();
        tensor->set_elem_type(x.data_type());
        auto *shape = tensor->mutable_shape();
        for (const auto &dim : x.dims()) {
          TensorShapeProto::Dimension *new_dim = shape->add_dim();
          new_dim->set_dim_value(dim);
        }
      }
    }
    return model;
  }
};

const std::vector<std::string> GetAvailablePasses();

const std::vector<std::string> GetFuseAndEliminationPass();

ModelProto Optimize(const ModelProto &mp_in,
                    const std::vector<std::string> &names);

ModelProto OptimizeFixed(const ModelProto &mp_in,
                         const std::vector<std::string> &names);
}  // namespace optimization
}  // namespace ONNX_NAMESPACE
