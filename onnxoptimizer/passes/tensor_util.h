/*
 * SPDX-License-Identifier: Apache-2.0
 */

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include <vector>

#include "onnx/common/tensor.h"
#include "onnxoptimizer/passes/data_type.h"

namespace ONNX_NAMESPACE {
namespace optimization {

int64_t ElemCntOfTensor(const Tensor* tensor);
int64_t ElemCntOfTensor(const Tensor& tensor);

template <typename T>
const std::vector<T> ParseTensorData(const Tensor* tensor);

}  // namespace optimization
}  // namespace ONNX_NAMESPACE