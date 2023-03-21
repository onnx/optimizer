/*
 * SPDX-License-Identifier: Apache-2.0
 */

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include <algorithm>
#include <functional>
#include <type_traits>
#include <typeinfo>

#include "onnx/onnx_pb.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/logging.h"
#include "onnxoptimizer/passes/string_utils.h"
#include "onnxoptimizer/passes/tensor_util.h"

namespace ONNX_NAMESPACE {
namespace optimization {

/// https://stackoverflow.com/questions/2590677/how-do-i-combine-hash-values-in-c0x
inline void hash_combine(std::size_t& seed) {}

template <typename Hasher, typename T, typename... Rest>
void hash_combine(std::size_t& seed, const Hasher& hasher, const T& v,
                  Rest... rest) {
  seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  hash_combine(seed, rest...);
}

struct SymbolCompare {
  bool operator()(const Symbol& lhs, const Symbol& rhs) {
    return static_cast<uint32_t>(lhs) < static_cast<uint32_t>(rhs);
  }
};

inline bool CSETensorCompare(const Tensor* lhs, const Tensor* rhs) {
  // lhs and rhs maybe nullptr
  if (!lhs) {
    return !rhs;
  } else if (!rhs) {
    return !lhs;
  }
  ONNX_ASSERT(!lhs->is_segment() && !rhs->is_segment());
  if (lhs->elem_type() != rhs->elem_type() || lhs->sizes() != rhs->sizes()) {
    return false;
  }

#define DO_CASE(pb_type, cpp_type)                                        \
  case ONNX_NAMESPACE::TensorProto_DataType_##pb_type:                    \
    if (ParseTensorData<cpp_type>(lhs) != ParseTensorData<cpp_type>(rhs)) \
      return false;                                                       \
    break;

  switch (lhs->elem_type()) {
    DO_CASE(BOOL, bool)
    DO_CASE(INT8, int8_t)
    DO_CASE(INT16, int16_t)
    DO_CASE(INT32, int32_t)
    DO_CASE(INT64, int64_t)
    DO_CASE(UINT8, uint8_t)
    DO_CASE(UINT16, uint16_t)
    DO_CASE(UINT32, uint32_t)
    DO_CASE(UINT64, uint64_t)
    DO_CASE(FLOAT, float)
    DO_CASE(DOUBLE, double)
    DO_CASE(COMPLEX64, Complex64)
    DO_CASE(COMPLEX128, Complex128)
    DO_CASE(FLOAT16, Float16)
    DO_CASE(BFLOAT16, BFloat16)

#undef DO_CASE

    case ONNX_NAMESPACE::TensorProto_DataType_STRING:
      if (lhs->strings() != rhs->strings())
        return false;
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED:
      // tensor is empty
      break;
    default:
      return false;
  }
  return true;
}

inline bool IsSupportedByCSE(const Node* n) {
  if (!n) {
    return false;
  }
  const auto attribute_names = n->attributeNames();
  for (const auto& name : attribute_names) {
    auto kind = n->kindOf(name);
    switch (kind) {
      case AttributeKind::g:
      case AttributeKind::gs:
      case AttributeKind::tp:
      case AttributeKind::tps:
        return false;
      default:
        break;
    }
  }
  return true;
}

template <typename T>
struct CSEContainerHash {
  std::size_t operator()(const std::vector<T>& container) const {
    std::size_t seed = 0;
    hash_combine(seed, std::hash<std::string>(), std::string(typeid(T).name()),
                 std::hash<std::size_t>(), container.size());
    for (const auto& d : container) {
      hash_combine(seed, std::hash<T>(), d);
    }
    return seed;
  }
};

struct CSETensorHash {
  std::size_t operator()(const Tensor* tensor) const {
    /// https://github.com/onnx/onnx/issues/2630
    ONNX_ASSERT(tensor && !tensor->is_segment());
    std::size_t seed = 0;
    auto int32_hasher = std::hash<int32_t>();
    auto size_hasher = std::hash<std::size_t>();
    const auto elem_type = tensor->elem_type();
    /// dtype、dims、value
    hash_combine(seed, int32_hasher, elem_type);
    hash_combine(seed, CSEContainerHash<int64_t>(), tensor->sizes());

#define DO_CASE(pb_type, cpp_type)                     \
  case ONNX_NAMESPACE::TensorProto_DataType_##pb_type: \
    hash_combine(seed, CSEContainerHash<cpp_type>(),   \
                 ParseTensorData<cpp_type>(tensor));   \
    break;

    switch (elem_type) {
      DO_CASE(BOOL, bool)
      DO_CASE(INT8, int8_t)
      DO_CASE(INT16, int16_t)
      DO_CASE(INT32, int32_t)
      DO_CASE(INT64, int64_t)
      DO_CASE(UINT8, uint8_t)
      DO_CASE(UINT16, uint16_t)
      DO_CASE(UINT32, uint32_t)
      DO_CASE(UINT64, uint64_t)
      DO_CASE(FLOAT, float)
      DO_CASE(DOUBLE, double)
      DO_CASE(COMPLEX64, Complex64)
      DO_CASE(COMPLEX128, Complex128)
      DO_CASE(FLOAT16, Float16)
      DO_CASE(BFLOAT16, BFloat16)

#undef DO_CASE
      case ONNX_NAMESPACE::TensorProto_DataType_STRING:
        hash_combine(seed, CSEContainerHash<std::string>(), tensor->strings());
        break;
      case ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED:
        break;
      default:
        throw std::runtime_error(Str("no supported data type: ", elem_type));
        break;
    }
    return seed;
  }
};

template <>
struct CSEContainerHash<Tensor> {
  std::size_t operator()(const std::vector<Tensor>& container) const {
    std::size_t seed = 0;
    hash_combine(seed, std::hash<std::string>(),
                 std::string(typeid(Tensor).name()), std::hash<std::size_t>(),
                 container.size());
    for (const auto& d : container) {
      hash_combine(seed, CSETensorHash(), &d);
    }
    return seed;
  }
};

struct CSENodeHash {
  std::size_t operator()(const Node* n) const {
    ONNX_ASSERT(n);
    std::size_t seed = 0;
    const auto inputs = n->inputs();
    auto size_t_hasher = std::hash<std::size_t>();
    auto string_hasher = std::hash<std::string>();
    auto sym_hasher = std::hash<Symbol>();
    hash_combine(seed, std::hash<uint32_t>(), static_cast<uint32_t>(n->kind()),
                 size_t_hasher, inputs.size());
    for (const auto& input : inputs) {
      hash_combine(seed, string_hasher, input->uniqueName());
    }
    auto attribute_names = n->attributeNames();
    SymbolCompare cmp;
    std::sort(attribute_names.begin(), attribute_names.end(), cmp);
    for (const auto& name : attribute_names) {
      hash_combine(seed, sym_hasher, name);
      auto kind = n->kindOf(name);
      switch (kind) {
        case ONNX_NAMESPACE::AttributeKind::f:
          hash_combine(seed, std::hash<double>(), n->f(name));
          break;
        case ONNX_NAMESPACE::AttributeKind::fs:
          hash_combine(seed, CSEContainerHash<double>(), n->fs(name));
          break;
        case ONNX_NAMESPACE::AttributeKind::i:
          hash_combine(seed, std::hash<int64_t>(), n->i(name));
          break;
        case ONNX_NAMESPACE::AttributeKind::is:
          hash_combine(seed, CSEContainerHash<int64_t>(), n->is(name));
          break;
        case ONNX_NAMESPACE::AttributeKind::s:
          hash_combine(seed, string_hasher, n->s(name));
          break;
        case ONNX_NAMESPACE::AttributeKind::ss:
          hash_combine(seed, CSEContainerHash<std::string>(), n->ss(name));
          break;
        case ONNX_NAMESPACE::AttributeKind::t:
          hash_combine(seed, CSETensorHash(), &n->t(name));
          break;
        case ONNX_NAMESPACE::AttributeKind::ts:
          hash_combine(seed, CSEContainerHash<Tensor>(), n->ts(name));
          break;
        default:
          throw std::runtime_error(
              Str("no support hash type: ", ONNX_NAMESPACE::toString(kind)));
          break;
      }
    }
    hash_combine(seed, size_t_hasher, n->outputs().size());
    return seed;
  }
};

struct CSEEqual {
  bool operator()(const Node* lhs, const Node* rhs) const {
    if (!lhs) {
      return !rhs;
    } else if (!rhs) {
      return !lhs;
    }

    auto inputs_l = lhs->inputs();
    auto inputs_r = rhs->inputs();
    auto outputs_l = lhs->outputs();
    auto outputs_r = rhs->outputs();
    auto attr_names_l = lhs->attributeNames();
    auto attr_names_r = rhs->attributeNames();
    SymbolCompare cmp;
    std::sort(attr_names_l.begin(), attr_names_l.end(), cmp);
    std::sort(attr_names_r.begin(), attr_names_r.end(), cmp);
    if (lhs->kind() != rhs->kind() || inputs_l.size() != inputs_r.size() ||
        outputs_l.size() != outputs_r.size() || attr_names_l != attr_names_r) {
      return false;
    }
    for (int i = 0; i < inputs_l.size(); ++i) {
      if (inputs_l[i]->uniqueName() != inputs_r[i]->uniqueName()) {
        return false;
      }
    }

    for (int i = 0; i < attr_names_l.size(); ++i) {
      const auto attr_name = attr_names_l[i];
      if (lhs->kindOf(attr_name) != rhs->kindOf(attr_name)) {
        return false;
      }
      switch (lhs->kindOf(attr_name)) {
        case AttributeKind::f:
          if (lhs->f(attr_name) != rhs->f(attr_name))
            return false;
          break;
        case AttributeKind::fs:
          if (lhs->fs(attr_name) != rhs->fs(attr_name))
            return false;
          break;
        case AttributeKind::i:
          if (lhs->i(attr_name) != rhs->i(attr_name))
            return false;
          break;
        case AttributeKind::is:
          if (lhs->is(attr_name) != rhs->is(attr_name))
            return false;
          break;
        case AttributeKind::s:
          if (lhs->s(attr_name) != rhs->s(attr_name))
            return false;
          break;
        case AttributeKind::ss:
          if (lhs->ss(attr_name) != rhs->ss(attr_name))
            return false;
          break;
        case AttributeKind::t:
          if (!CSETensorCompare(&lhs->t(attr_name), &rhs->t(attr_name)))
            return false;
          break;
        case AttributeKind::ts: {
          const auto& lts = lhs->ts(attr_name);
          const auto& rts = rhs->ts(attr_name);
          if (lts.size() != rts.size())
            return false;
          for (std::size_t k = 0; k < lts.size(); ++k) {
            if (!CSETensorCompare(&lts[k], &rts[k])) {
              return false;
            }
          }
          break;
        }
        default:
          return false;
      }
    }
    return true;
  }
};

struct CSETensorEqual {
  bool operator()(const Tensor* lhs, const Tensor* rhs) const {
    return CSETensorCompare(lhs, rhs);
  }
};

}  // namespace optimization
}  // namespace ONNX_NAMESPACE
