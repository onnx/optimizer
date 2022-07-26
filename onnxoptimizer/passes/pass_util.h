/*
 * SPDX-License-Identifier: Apache-2.0
 */

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include "onnxoptimizer/pass.h"

namespace ONNX_NAMESPACE {
namespace optimization {

template <typename T>
T AddYIfNegative(T x, T y) {
  return x < 0 ? x + y : x;
}

inline bool IsConstantTensor(const Value* v) {
  auto* graph = v->owningGraph();
  return v->node()->kind() == kConstant || graph->is_constant_initializer(v);
}

inline bool IsConstantTensor(const Node* n, size_t which_input) {
  ONNX_ASSERT(which_input < n->inputs().size());
  return IsConstantTensor(n->inputs()[which_input]);
}

inline const Tensor* FetchConstantTensor(const Value* v) {
  const uint32_t kind = v->node()->kind();
  auto* graph = v->owningGraph();
  if (kind == kConstant) {
    return &v->node()->t(kvalue);
  } else if (graph->is_constant_initializer(v)) {
    return &*graph->getInitializer(v->uniqueName());
  } else {
    return nullptr;
  }
}

// fetch the only element when the tensor is a scalar or a tensor that only has
// a element
template <typename T>
bool FetchSoleValueOfTensor(const Value* t, T& val);

// FetchSoleIntValueOfTensor is a wraper that fetchs int value(INT32 or INT64)
// easier. E.g: get axis from axes tensor
bool FetchSoleIntValueOfTensor(const Value* t, int64_t& val);

bool FetchSoleIntValueOfAttr(const Node* node, Symbol attr_name, int64_t& val);

inline bool CheckKind(const Value* v, const Symbol& symbol) {
  return v->node()->kind() == symbol;
}

inline bool CheckKind(const Value* v, const char* symbol) {
  return CheckKind(v, Symbol(symbol));
}

inline bool CheckKind(const Node* n, const Symbol& symbol) {
  return n->kind() == symbol;
}

inline bool CheckKind(const Node* n, const char* symbol) {
  return CheckKind(n, Symbol(symbol));
}

inline std::pair<int64_t, int64_t> FetchStartAndEndAttrOfShape(
    const Node* shape) {
  ONNX_ASSERT(CheckKind(shape, "Shape") && shape->input()->has_sizes());

  const int64_t start = AddYIfNegative<int64_t>(
      shape->hasAttribute(Symbol("start")) ? shape->i(Symbol("start")) : 0,
      shape->input()->sizes().size());
  const int64_t end = AddYIfNegative<int64_t>(
      shape->hasAttribute(Symbol("end")) ? shape->i(Symbol("end"))
                                         : shape->input()->sizes().size(),
      shape->input()->sizes().size());
  return {start, end};
}

}  // namespace optimization
}  // namespace ONNX_NAMESPACE
