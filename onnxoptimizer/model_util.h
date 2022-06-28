/*
 * SPDX-License-Identifier: Apache-2.0
 */

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include "onnx/onnx_pb.h"

namespace ONNX_NAMESPACE {
namespace optimization {

void loadModel(ModelProto* m, const std::string& model_path,
               const bool load_external_data = false);

void saveModel(ModelProto* m, const std::string& model_path,
               const bool save_external_data = false,
               const std::string& data_file_name = {});

}  // namespace optimization
}  // namespace ONNX_NAMESPACE