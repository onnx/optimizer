/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "onnx/py_utils.h"
#include "onnxoptimizer/model_util.h"
#include "onnxoptimizer/optimize.h"

namespace ONNX_NAMESPACE {
namespace py = pybind11;
using namespace pybind11::literals;
PYBIND11_MODULE(onnx_opt_cpp2py_export, onnx_opt_cpp2py_export) {
  onnx_opt_cpp2py_export.doc() = "ONNX Optimizer";

  onnx_opt_cpp2py_export.def(
      "optimize",
      [](const py::bytes& bytes, const std::vector<std::string>& names) {
        ModelProto proto{};
        ParseProtoFromPyBytes(&proto, bytes);
        auto const result = optimization::Optimize(proto, names);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out);
      });

  onnx_opt_cpp2py_export.def(
      "optimize_fixedpoint",
      [](const py::bytes& bytes, const std::vector<std::string>& names) {
        ModelProto proto{};
        ParseProtoFromPyBytes(&proto, bytes);
        auto const result = optimization::OptimizeFixed(proto, names);
        std::string out;
        result.SerializeToString(&out);
        return py::bytes(out);
      });

  onnx_opt_cpp2py_export.def(
      "optimize_from_path", [](const std::string& import_model_path,
                               const std::string& export_model_path,
                               const std::vector<std::string>& names,
                               const std::string& export_data_file_name) {
        ModelProto proto{};
        optimization::loadModel(&proto, import_model_path, true);
        auto result = optimization::Optimize(proto, names);
        optimization::saveModel(&result, export_model_path, true,
                                export_data_file_name);
      });

  onnx_opt_cpp2py_export.def(
      "optimize_fixedpoint_from_path",
      [](const std::string& import_model_path,
         const std::string& export_model_path,
         const std::vector<std::string>& names,
         const std::string& export_data_file_name) {
        ModelProto proto{};
        optimization::loadModel(&proto, import_model_path, true);
        auto result = optimization::OptimizeFixed(proto, names);
        optimization::saveModel(&result, export_model_path, true,
                                export_data_file_name);
      });
  onnx_opt_cpp2py_export.def("get_available_passes",
                             &optimization::GetAvailablePasses);
  onnx_opt_cpp2py_export.def("get_fuse_and_elimination_passes",
                             &optimization::GetFuseAndEliminationPass);
}
}  // namespace ONNX_NAMESPACE
