/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <onnx/checker.h>
#include <onnx/onnx_pb.h>
#include <onnxoptimizer/model_util.h>
#include <onnxoptimizer/optimize.h>

#include <filesystem>
#include <fstream>

void printUsage() {
  std::string usage =
      R"(Usage: onnx_optimizer_exec [model.onnx] [model_out.onnx]  [optional: model_data_out.data])";
  std::cout << usage << std::endl;
}

int main(int argc, char** argv) {
  if (argc != 3 && argc != 4) {
    printUsage();
    return -1;
  }
  std::string model_in_path(argv[1]);
  std::string model_out_path(argv[2]);
  std::string model_data_path{};
  if (argc == 4) {
    model_data_path = std::filesystem::relative(
        std::string(argv[3]),
        std::filesystem::path(model_out_path).parent_path()).string();
  }

  try {
    ONNX_NAMESPACE::ModelProto model;
    onnx::optimization::loadModel(&model, model_in_path, true);
    onnx::checker::check_model(model);
    auto new_model = onnx::optimization::Optimize(
        model, onnx::optimization::GetFuseAndEliminationPass());
    onnx::checker::check_model(new_model);
    bool save_external_data = !model_data_path.empty();
    onnx::optimization::saveModel(&new_model, model_out_path,
                                  save_external_data, model_data_path);

  } catch (std::exception& e) {
    std::cout << e.what() << std::endl;
    return -1;
  }
  return 0;
}
