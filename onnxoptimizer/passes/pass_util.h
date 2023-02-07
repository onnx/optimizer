/*
 * SPDX-License-Identifier: Apache-2.0
 */

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include "onnxoptimizer/pass.h"

namespace ONNX_NAMESPACE {
namespace optimization {

namespace {
template <typename T>
struct CanonicalizeSymbolType {
  using type = const T&;
};

template <size_t N>
struct CanonicalizeSymbolType<char[N]> {
  using type = const char*;
};

template <typename S>
struct ToSymbol {
  static Symbol Call(const S& s) {
    return Symbol(s);
  }
};

template <>
struct ToSymbol<const char*> {
  static Symbol Call(const char* s) {
    return Symbol(std::string(s));
  }
};

}  // namespace

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

template <typename Sym>
bool FetchSoleIntValueOfAttr(const Node* node, const Sym& symbol,
                             int64_t& val) {
  Symbol attr_name =
      ToSymbol<typename CanonicalizeSymbolType<Sym>::type>::Call(symbol);
  if (node->kindOf(attr_name) == AttributeKind::is) {
    const auto attrs = node->is(attr_name);
    if (attrs.size() != 1) {
      return false;
    }
    val = attrs[0];
    return true;
  } else if (node->kindOf(attr_name) == AttributeKind::i) {
    val = node->i(attr_name);
    return true;
  } else {
    return false;
  }
}

template <typename Sym>
bool CheckKind(const Value* v, const Sym& symbol) {
  Symbol s = ToSymbol<typename CanonicalizeSymbolType<Sym>::type>::Call(symbol);
  return v->node()->kind() == s;
}

template <typename Sym>
bool CheckKind(const Node* n, const Sym& symbol) {
  Symbol s = ToSymbol<typename CanonicalizeSymbolType<Sym>::type>::Call(symbol);
  return n->kind() == s;
}

inline std::pair<int64_t, int64_t> FetchStartAndEndAttrOfShape(
    const Node* shape) {
  ONNX_ASSERT(CheckKind(shape, "Shape") && shape->input()->has_sizes());

  const int64_t start = AddYIfNegative<int64_t>(
      shape->hasAttribute("start"_sym) ? shape->i("start"_sym) : 0,
      shape->input()->sizes().size());
  const int64_t end = AddYIfNegative<int64_t>(
      shape->hasAttribute("end"_sym) ? shape->i("end"_sym)
                                     : shape->input()->sizes().size(),
      shape->input()->sizes().size());
  return {start, end};
}

#define Create_GetValueFromAttr(type_, method)                                 \
  template <typename Sym>                                                      \
  bool GetValueFromAttr(const Node* n, const Sym& attr_name, type_& value) {   \
    Symbol attr =                                                              \
        ToSymbol<typename CanonicalizeSymbolType<Sym>::type>::Call(attr_name); \
    if (!n->hasAttribute(attr)) {                                              \
      return false;                                                            \
    }                                                                          \
    value = n->method(attr);                                                   \
    return true;                                                               \
  }                                                                            \
  template <typename Sym>                                                      \
  type_ GetValueFromAttrWithDefault(const Node* n, const Sym& attr_name,       \
                                    const type_& default_value) {              \
    Symbol attr =                                                              \
        ToSymbol<typename CanonicalizeSymbolType<Sym>::type>::Call(attr_name); \
    return n->hasAttribute(attr) ? n->method(attr) : default_value;            \
  }

Create_GetValueFromAttr(double, f);
Create_GetValueFromAttr(float, f);
Create_GetValueFromAttr(std::vector<double>, fs);
Create_GetValueFromAttr(std::string, s);
Create_GetValueFromAttr(std::vector<std::string>, ss);
Create_GetValueFromAttr(int32_t, i);
Create_GetValueFromAttr(int64_t, i);
Create_GetValueFromAttr(std::vector<int64_t>, is);
Create_GetValueFromAttr(Tensor, t);
Create_GetValueFromAttr(std::vector<Tensor>, ts);
#undef Create_GetValueFromAttr

#define Create_GetValueFromInput(type_)                                      \
  inline bool GetValueFromInput(const Node* n, size_t which, type_& value) { \
    if (which >= n->inputs().size()) {                                       \
      return false;                                                          \
    }                                                                        \
    const Tensor* tensor = FetchConstantTensor(n->inputs()[which]);          \
    if (!tensor) {                                                           \
      return false;                                                          \
    }                                                                        \
    value = ParseData<typename type_::value_type>(tensor);                   \
    return true;                                                             \
  }

Create_GetValueFromInput(std::vector<float>);
Create_GetValueFromInput(std::vector<double>);
Create_GetValueFromInput(std::vector<int32_t>);
Create_GetValueFromInput(std::vector<int64_t>);
Create_GetValueFromInput(std::vector<uint64_t>);
#undef Create_GetValueFromInput

template <typename T, typename Sym>
bool GetValueFromAttrOrInput(const Node* n, const Sym& attr_name,
                             size_t which_input, T& value) {
  return GetValueFromAttr(n, attr_name, value) ||
         GetValueFromInput(n, which_input, value);
}

}  // namespace optimization

}  // namespace ONNX_NAMESPACE
