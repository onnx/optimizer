/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <onnxoptimizer/optimize.h>

#include <onnx/onnx_pb.h>

#include <fstream>

int main(int argc, char **argv) {
  ONNX_NAMESPACE::ModelProto model;
  std::ifstream ifs(argv[1]);
  bool success = model.ParseFromIstream(&ifs);
  if (!success) {
    std::cout << "load failed" << std::endl;
    return -1;
  }
  onnx::optimization::Optimize(
      model,
      {"eliminate_deadend", "eliminate_nop_dropout", "eliminate_nop_cast",
       "eliminate_nop_monotone_argmax", "eliminate_nop_pad",
       "extract_constant_to_initializer", "eliminate_unused_initializer",
       "eliminate_nop_transpose", "eliminate_nop_flatten", "eliminate_identity",
       "fuse_add_bias_into_conv", "fuse_consecutive_concats",
       "fuse_consecutive_log_softmax", "fuse_consecutive_reduce_unsqueeze",
       "fuse_consecutive_squeezes", "fuse_consecutive_transposes",
       "fuse_matmul_add_bias_into_gemm", "fuse_pad_into_conv",
       "fuse_transpose_into_gemm"});
  std::ofstream ofs(argv[2]);
  success = model.SerializePartialToOstream(&ofs);
  if (!success) {
    std::cout << "save failed" << std::endl;
    return -1;
  }
  return 0;
}
