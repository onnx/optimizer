/*
 * SPDX-License-Identifier: Apache-2.0
 */

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

// Before:
//   P = Pad(X) - opset 10 and below (or) Pad(X, Pads, [constant_value]) - opset
//   11 and above Z = pool(P, Y)
// After:
//   Z = pool(X, Y) with "pads" attribute set
//
// the pass handles the case when Pad is zero-padding the input
// (i.e. mode=constant and constant_value=0)

#include <numeric>

#include "onnx/defs/tensor_util.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct FusePadIntoPool final : public PredicateBasedPass {
  explicit FusePadIntoPool()
      : PredicateBasedPass(PassType::Fuse, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override {
    return "fuse_pad_into_pool";
  }

  bool patternMatchPredicate(Node* node) override {
    return CheckKind(node, "AveragePool", 0, kPad) ||
           CheckKind(node, "MaxPool", 0, kPad);
  }

  bool runTransform(Node* n, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    destroy_current = NodeDestroyType::DestroyZero;

    // check if Pad is only used by pool
    if (n->inputs()[0]->uses().size() > 1) {
      return false;
    }

    Node* pool = n;
    Node* pad = n->inputs()[0]->node();

    // Process 'pads' data
    std::vector<int64_t> pads;
    if (!GetValueFromAttrOrInput(pad, kpads, 1, pads)) {
      return false;
    }

    // Process 'mode'
    std::string default_pad_mode{"constant"};
    std::string pad_mode =
        GetValueFromAttrWithDefault(pad, kmode, default_pad_mode);

    // cannot fuse if the pad mode is not "Constant"
    if (pad_mode != default_pad_mode) {
      return false;
    }

    // Process 'Constant_value'
    {
      union ConstantValueType {
        int32_t i32;
        int64_t i64;
        float f32;
        double f64;
        uint8_t ui8;
        int8_t i8;
        uint16_t ui16;
        int16_t i16;
      } cv;

#define Define_GetConstantValueFromInput(token) \
  (GetValueFromInput(pad, 2, cv.token) && cv.token == decltype(cv.token)(0))

      do {
        if (GetValueFromAttr(pad, kvalue, cv.f64) && cv.f64 == double(0)) {
          break;
        }
        if (pad->inputs().size() >= 3) {
          if (pad->input(2)->uniqueName().empty()) {
            break;
          }
          if (Define_GetConstantValueFromInput(i32) ||
              Define_GetConstantValueFromInput(i64) ||
              Define_GetConstantValueFromInput(f32) ||
              Define_GetConstantValueFromInput(f64) ||
              Define_GetConstantValueFromInput(ui8) ||
              Define_GetConstantValueFromInput(i8) ||
              Define_GetConstantValueFromInput(ui16) ||
              Define_GetConstantValueFromInput(i16)) {
            break;
          }
          return false;
        }
      } while (0);

#undef Define_GetConstantValueFromInput
    }

    // check if some values in 'pads' prevents us from fusing it into 'Conv'
    // node
    int pads_size = static_cast<int>(pads.size());

    // check if padding is applied only on feature dims
    if (pads[0] != 0 || pads[1] != 0 || pads[pads_size / 2] != 0 ||
        pads[pads_size / 2 + 1] != 0) {
      return false;
    }

    // check if padding is only positive
    if (std::any_of(pads.begin(), pads.end(),
                    [](int64_t local_value) { return local_value < 0; })) {
      return false;
    }

    int pool_pads_size = pads_size - 4;
    std::vector<int64_t> pool_pads(pool_pads_size, 0);
    // Fuse into existing padding, if available
    if (pool->hasAttribute(kpads)) {
      pool_pads = pool->is(kpads);
    }

    for (int i = 2, j = 0; i < pads_size / 2; ++i, ++j) {
      pool_pads[j] += pads[i];
      pool_pads[pool_pads_size / 2 + j] += pads[pads_size / 2 + i];
    }

    if (pool->kind() == Symbol("AveragePool")) {
      int64_t count_include_pad = 1;
      pool->i_(kcount_include_pad, count_include_pad);
    }

    pool->is_(kpads, std::move(pool_pads));
    pool->replaceInput(0, pad->inputs()[0]);
    pad->destroy();

    return true;
  }
};

}  // namespace optimization
}  // namespace ONNX_NAMESPACE
