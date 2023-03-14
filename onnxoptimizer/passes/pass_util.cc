/*
 * SPDX-License-Identifier: Apache-2.0
 */

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#include "onnx/defs/tensor_util.h"
#include "pass_util.h"

namespace ONNX_NAMESPACE {
namespace optimization {

bool FetchSoleIntValueOfTensor(const Value* t, int64_t& val) {
  int32_t i32_val;
  const bool r1 = FetchSoleValueOfTensor<int64_t>(t, val);
  const bool r2 = FetchSoleValueOfTensor<int32_t>(t, i32_val);
  if (r2) {
    val = i32_val;
  }
  return r1 || r2;
}

}  // namespace optimization
}  // namespace ONNX_NAMESPACE
