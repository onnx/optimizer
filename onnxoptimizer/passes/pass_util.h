/*
 * SPDX-License-Identifier: Apache-2.0
 */

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include <algorithm>
#include <type_traits>

#include "onnx/onnx_pb.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/logging.h"
#include "onnxoptimizer/passes/string_utils.h"
#include "onnxoptimizer/passes/tensor_util.h"

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

template <typename>
struct SupportedTypeOfAttr : public std::false_type {};

#define Create_SupportedTypeOfAttr(type, kind_)              \
  template <>                                                \
  struct SupportedTypeOfAttr<type> : public std::true_type { \
    static constexpr AttributeKind kind = kind_;             \
  };

Create_SupportedTypeOfAttr(double, AttributeKind::f);
Create_SupportedTypeOfAttr(float, AttributeKind::f);
Create_SupportedTypeOfAttr(std::string, AttributeKind::s);
Create_SupportedTypeOfAttr(int32_t, AttributeKind::i);
Create_SupportedTypeOfAttr(int64_t, AttributeKind::i);
Create_SupportedTypeOfAttr(std::vector<double>, AttributeKind::fs);
Create_SupportedTypeOfAttr(std::vector<float>, AttributeKind::fs);
Create_SupportedTypeOfAttr(std::vector<std::string>, AttributeKind::ss);
Create_SupportedTypeOfAttr(std::vector<int32_t>, AttributeKind::is);
Create_SupportedTypeOfAttr(std::vector<int64_t>, AttributeKind::is);
#undef Create_SupportedTypeOfAttr

template <typename>
struct SupportedTypeOfTensor : public std::false_type {};

#define Create_SupportedTypeOfTensor(cpp_type_, elem_type_)             \
  template <>                                                           \
  struct SupportedTypeOfTensor<cpp_type_> : public std::true_type {     \
    using cpp_type = cpp_type_;                                         \
    static constexpr int elem_type = TensorProto_DataType_##elem_type_; \
  };

Create_SupportedTypeOfTensor(std::vector<int32_t>, INT32);
Create_SupportedTypeOfTensor(std::vector<int64_t>, INT64);
Create_SupportedTypeOfTensor(std::vector<uint64_t>, UINT64);
Create_SupportedTypeOfTensor(std::vector<float>, FLOAT);
Create_SupportedTypeOfTensor(std::vector<double>, DOUBLE);
Create_SupportedTypeOfTensor(std::vector<uint32_t>, UINT32);
Create_SupportedTypeOfTensor(std::vector<int8_t>, INT8);
Create_SupportedTypeOfTensor(std::vector<uint8_t>, UINT8);
Create_SupportedTypeOfTensor(std::vector<int16_t>, INT16);
Create_SupportedTypeOfTensor(std::vector<uint16_t>, UINT16);
Create_SupportedTypeOfTensor(std::vector<Float16>, FLOAT16);
Create_SupportedTypeOfTensor(std::vector<BFloat16>, BFLOAT16);
Create_SupportedTypeOfTensor(std::vector<Complex64>, COMPLEX64);
Create_SupportedTypeOfTensor(std::vector<Complex128>, COMPLEX128);
Create_SupportedTypeOfTensor(std::vector<std::string>, STRING);
#undef Create_SupportedTypeOfTensor
}  // namespace

template <typename NodeOrValue, typename Sym,
          typename = std::enable_if_t<std::is_same_v<NodeOrValue, Value> ||
                                      std::is_same_v<NodeOrValue, Node>>>
bool CheckKind(const NodeOrValue* nv, const Sym& symbol) {
  Symbol s = ToSymbol<typename CanonicalizeSymbolType<Sym>::type>::Call(symbol);
  if constexpr (std::is_same_v<NodeOrValue, Value>) {
    return nv->node()->kind() == s;
  }
  if constexpr (std::is_same_v<NodeOrValue, Node>) {
    return nv->kind() == s;
  }
  return false;
}

template <typename Sym1, typename Which, typename Sym2, typename... Args>
bool CheckKind(const Node* n, const Sym1& s1, const Which& which,
               const Sym2& s2, const Args&... args) {
  return CheckKind(n, s1) && which < n->inputs().size() &&
         CheckKind(n->input(which)->node(), s2, args...);
}

template <typename T1, typename T2>
T1 AddYIfNegative(T1 x, T2 y) {
  return x < 0 ? x + y : x;
}

inline bool IsConstantTensor(const Value* v) {
  auto* graph = v->owningGraph();
  return v->node()->kind() == kConstant || graph->is_constant_initializer(v);
}

template <typename W, typename... Args>
bool IsConstantTensor(const Node* n, const W& which_input,
                      const Args&... args) {
  ONNX_ASSERT(which_input < n->inputs().size());
  if constexpr (sizeof...(args) == 0) {
    return IsConstantTensor(n->input(which_input));
  } else {
    return IsConstantTensor(n->input(which_input)->node(), args...);
  }
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

template <typename T, typename Sym,
          typename = std::enable_if_t<SupportedTypeOfAttr<T>::value>>
bool GetValueFromAttr(const Node* n, const Sym& attr_name, T& value) {
  Symbol attr =
      ToSymbol<typename CanonicalizeSymbolType<Sym>::type>::Call(attr_name);
  if (!n->hasAttribute(attr)) {
    return false;
  }
#define DEFINE_IF_CONSTEXPR_WITH_NO_CAST(cpp_type, method) \
  if constexpr (std::is_same_v<T, cpp_type>) {             \
    if (n->kindOf(attr) != SupportedTypeOfAttr<T>::kind) { \
      return false;                                        \
    }                                                      \
    value = static_cast<T>(n->method(attr));               \
    return true;                                           \
  }

  DEFINE_IF_CONSTEXPR_WITH_NO_CAST(float, f)
  DEFINE_IF_CONSTEXPR_WITH_NO_CAST(double, f)
  DEFINE_IF_CONSTEXPR_WITH_NO_CAST(std::string, s)
  DEFINE_IF_CONSTEXPR_WITH_NO_CAST(int32_t, i)
  DEFINE_IF_CONSTEXPR_WITH_NO_CAST(int64_t, i)
  DEFINE_IF_CONSTEXPR_WITH_NO_CAST(std::vector<std::string>, ss)
  DEFINE_IF_CONSTEXPR_WITH_NO_CAST(std::vector<double>, fs)
  DEFINE_IF_CONSTEXPR_WITH_NO_CAST(std::vector<int64_t>, is)
#undef DEFINE_IF_CONSTEXPR_WITH_NO_CAST

#define DEFINE_IF_CONSTEXPR_WITH_CAST(cpp_type, method)                  \
  if constexpr (std::is_same_v<T, cpp_type>) {                           \
    if (n->kindOf(attr) != SupportedTypeOfAttr<T>::kind) {               \
      return false;                                                      \
    }                                                                    \
    const auto& v = n->method(attr);                                     \
    std::transform(v.cbegin(), v.cend(), std::back_inserter(value),      \
                   [](const auto& d) { return static_cast<float>(d); }); \
    return true;                                                         \
  }

  DEFINE_IF_CONSTEXPR_WITH_CAST(std::vector<float>, fs)
  DEFINE_IF_CONSTEXPR_WITH_CAST(std::vector<int32_t>, is)
#undef DEFINE_IF_CONSTEXPR_WITH_CAST

  return false;
}

template <typename T, typename Sym>
T GetValueFromAttrWithDefault(const Node* n, const Sym& attr_name,
                              const T& default_value) {
  T temp_;
  if (GetValueFromAttr(n, attr_name, temp_)) {
    return temp_;
  } else {
    return default_value;
  }
}

template <typename Vec,
          typename = std::enable_if_t<SupportedTypeOfTensor<Vec>::value>>
bool GetValueFromInput(const Value* v, Vec& value) {
  const Tensor* tensor = FetchConstantTensor(v);
  if (!tensor || tensor->elem_type() != SupportedTypeOfTensor<Vec>::elem_type) {
    return false;
  }
  value = ParseTensorData<typename Vec::value_type>(tensor);
  return true;
}
template <typename T, typename = std::enable_if_t<
                          SupportedTypeOfTensor<std::vector<T>>::value>>
bool GetValueFromInput(const Value* v, T& value, size_t which_value = 0) {
  std::vector<T> temp_;
  if (GetValueFromInput(v, temp_) && which_value < temp_.size()) {
    value = temp_[which_value];
    return true;
  }
  return false;
}

template <typename Vec,
          typename = std::enable_if_t<SupportedTypeOfTensor<Vec>::value>>
bool GetValueFromInput(const Node* n, size_t which, Vec& value) {
  if (which >= n->inputs().size()) {
    return false;
  }
  return GetValueFromInput(n->input(which), value);
}

template <typename T, typename = std::enable_if_t<
                          SupportedTypeOfTensor<std::vector<T>>::value>>
bool GetValueFromInput(const Node* n, size_t which_input, T& value,
                       size_t which_value = 0) {
  if (which_input >= n->inputs().size()) {
    return false;
  }
  return GetValueFromInput(n->input(which_input), value, which_value);
}

template <typename Vec, typename Sym,
          typename = std::enable_if_t<
              std::is_same_v<Vec, std::vector<typename Vec::value_type>>>>
bool GetValueFromAttrOrInput(const Node* n, const Sym& attr_name,
                             size_t which_input, Vec& value) {
  return GetValueFromAttr(n, attr_name, value) ||
         GetValueFromInput(n, which_input, value);
}

template <typename T, typename Sym,
          typename = std::enable_if_t<std::is_arithmetic_v<T>>>
bool GetValueFromAttrOrInput(const Node* n, const Sym& attr_name,
                             size_t which_input, T& value,
                             size_t which_value = 0) {
  return GetValueFromAttr(n, attr_name, value) ||
         GetValueFromInput(n, which_input, value, which_value);
}

template <typename T, typename Sym>
bool FetchSoleValueOfAttr(const Node* node, const Sym& symbol, T& val) {
  Symbol attr_name =
      ToSymbol<typename CanonicalizeSymbolType<Sym>::type>::Call(symbol);
  static std::unordered_set<AttributeKind> container_type{
      AttributeKind::is, AttributeKind::fs, AttributeKind::ss};

  if (container_type.count(node->kindOf(attr_name)) == 1) {
    std::vector<T> temp_;
    if (GetValueFromAttr(node, attr_name, temp_) && temp_.size() == 1) {
      val = temp_[0];
      return true;
    }
  } else {
    return GetValueFromAttr(node, attr_name, val);
  }
  return false;
}

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

// fetch the only element when the tensor is a scalar or a tensor that only has
// a element
template <typename T>
bool FetchSoleValueOfTensor(const Value* t, T& val) {
  std::vector<T> temp_;
  if (!GetValueFromInput(t, temp_) || temp_.size() != 1) {
    return false;
  }
  val = temp_[0];
  return true;
}

// FetchSoleIntValueOfTensor is a wraper that fetchs int value(INT32 or INT64)
// easier. E.g: get axis from axes tensor
bool FetchSoleIntValueOfTensor(const Value* t, int64_t& val);

inline std::pair<int64_t, int64_t> FetchStartAndEndAttrOfShape(
    const Node* shape, const int64_t rank) {
  ONNX_ASSERT(CheckKind(shape, "Shape"));

  const int64_t start = AddYIfNegative<int64_t>(
      GetValueFromAttrWithDefault(shape, "start", 0), rank);
  const int64_t end = AddYIfNegative<int64_t>(
      GetValueFromAttrWithDefault(shape, "end", rank), rank);
  return {start, end};
}

inline std::pair<int64_t, int64_t> FetchStartAndEndAttrOfShape(
    const Node* shape) {
  ONNX_ASSERT(CheckKind(shape, "Shape") && shape->input()->has_sizes());
  return FetchStartAndEndAttrOfShape(
      shape, static_cast<int64_t>(shape->input()->sizes().size()));
}

template <typename Sym>
Node* ParentNode(Node* node, const Sym& symbol) {
  Symbol s = ToSymbol<typename CanonicalizeSymbolType<Sym>::type>::Call(symbol);

  for (auto* input : node->inputs()) {
    if (CheckKind(input, s)) {
      return input->node();
    }
  }
  return nullptr;
}

template <typename T>
Node* PrevNode(Node* n, T which) {
  ONNX_ASSERT(which < n->inputs().size());
  return n->input(which)->node();
}

template <typename T, typename U, typename... Args>
Node* PrevNode(Node* n, T which, U which1, Args... args) {
  ONNX_ASSERT(which < n->inputs().size());
  return PrevNode(n->input(which)->node(), which1, args...);
}

template <typename T>
bool HasIntersection(std::vector<T> v1, std::vector<T> v2) {
  std::vector<T> intersect;
  std::sort(v1.begin(), v1.end());
  std::sort(v2.begin(), v2.end());
  std::set_intersection(v1.begin(), v1.end(), v2.begin(), v2.end(),
                        std::back_inserter(intersect));
  return !intersect.empty();
}

template <typename T, typename... Args>
ArrayRef<Value*> GetInputsOfPreNode(Node* n, T which, Args... args) {
  return PrevNode(n, which, args...)->inputs();
}

inline bool HasDimsOfInputOfNode(Node* n, size_t which) {
  const auto vs = n->inputs();
  ONNX_ASSERT(which < vs.size());
  return vs[which]->has_sizes();
}

inline std::vector<int64_t> GetIntsFromValue(const Value* v) {
  std::vector<int64_t> is64;
  std::vector<int32_t> is32;
  if (GetValueFromInput(v, is64)) {
    return is64;
  }
  if (GetValueFromInput(v, is32)) {
    std::transform(is32.cbegin(), is32.cend(), std::back_inserter(is64),
                   [](const auto& d) { return static_cast<int64_t>(d); });
    return is64;
  }
  LOG(FATAL) << "We expect that the int32s or int64s exists in Value ("
             << Str(v->uniqueName(), "), but failed!");
  return is64;
}

inline bool isABroadcastToB(const std::vector<int64_t>& dims_a,
                            const std::vector<Dimension>& dims_b) {
  int ndim_a = dims_a.size();
  int ndim_b = dims_b.size();
  if (ndim_a > ndim_b) {
    return false;
  }

  ndim_a--;
  ndim_b--;

  for (; ndim_a >= 0; ndim_a--, ndim_b--) {
    int d_a = dims_a[ndim_a];
    auto const& d_b = dims_b[ndim_b];
    if (d_a == 1) {
      continue;
    }
    if (!d_b.is_int || (d_a != d_b.dim)) {
      return false;
    }
  }
  return true;
}

template <onnx::TensorProto_DataType>
struct ToCppType;

template <>
struct ToCppType<TensorProto_DataType_INT8> {
  using type = int8_t;
};
template <>
struct ToCppType<TensorProto_DataType_INT16> {
  using type = int16_t;
};
template <>
struct ToCppType<TensorProto_DataType_INT32> {
  using type = int32_t;
};
template <>
struct ToCppType<TensorProto_DataType_INT64> {
  using type = int64_t;
};
template <>
struct ToCppType<TensorProto_DataType_UINT8> {
  using type = uint8_t;
};
template <>
struct ToCppType<TensorProto_DataType_UINT16> {
  using type = uint16_t;
};
template <>
struct ToCppType<TensorProto_DataType_UINT32> {
  using type = uint32_t;
};
template <>
struct ToCppType<TensorProto_DataType_UINT64> {
  using type = uint64_t;
};
template <>
struct ToCppType<TensorProto_DataType_BFLOAT16> {
  using type = BFloat16;
};
template <>
struct ToCppType<TensorProto_DataType_FLOAT16> {
  using type = Float16;
};
template <>
struct ToCppType<TensorProto_DataType_FLOAT> {
  using type = float;
};
template <>
struct ToCppType<TensorProto_DataType_DOUBLE> {
  using type = double;
};
template <>
struct ToCppType<TensorProto_DataType_BOOL> {
  using type = bool;
};

}  // namespace optimization

}  // namespace ONNX_NAMESPACE
