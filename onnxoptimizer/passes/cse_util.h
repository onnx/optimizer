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
      case AttributeKind::t:
      case AttributeKind::ts:
      case AttributeKind::tp:
      case AttributeKind::tps:
        return false;
    }
  }
  return true;
}

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
        default:
          return false;
      }
    }
    return true;
  }
};

template <typename T>
struct CSEContainerHash {
  std::size_t operator()(const std::vector<T>& container) const {
    std::size_t seed = 0;
    ONNX_NAMESPACE::optimization::hash_combine(
        seed, std::hash<std::string>(), std::string(typeid(T).name()),
        std::hash<std::size_t>(), container.size());
    for (const auto& d : container) {
      ONNX_NAMESPACE::optimization::hash_combine(seed, std::hash<T>(), d);
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
    ONNX_NAMESPACE::optimization::hash_combine(seed, std::hash<uint32_t>(),
                                               static_cast<uint32_t>(n->kind()),
                                               size_t_hasher, inputs.size());
    for (const auto& input : inputs) {
      ONNX_NAMESPACE::optimization::hash_combine(seed, string_hasher,
                                                 input->uniqueName());
    }
    auto attribute_names = n->attributeNames();
    ONNX_NAMESPACE::optimization::SymbolCompare cmp;
    std::sort(attribute_names.begin(), attribute_names.end(), cmp);
    for (const auto& name : attribute_names) {
      ONNX_NAMESPACE::optimization::hash_combine(seed, sym_hasher, name);
      auto kind = n->kindOf(name);
      switch (kind) {
        case ONNX_NAMESPACE::AttributeKind::f:
          ONNX_NAMESPACE::optimization::hash_combine(seed, std::hash<double>(),
                                                     n->f(name));
          break;
        case ONNX_NAMESPACE::AttributeKind::fs:
          ONNX_NAMESPACE::optimization::hash_combine(
              seed, CSEContainerHash<double>(), n->fs(name));
          break;
        case ONNX_NAMESPACE::AttributeKind::i:
          ONNX_NAMESPACE::optimization::hash_combine(seed, std::hash<int64_t>(),
                                                     n->i(name));
          break;
        case ONNX_NAMESPACE::AttributeKind::is:
          ONNX_NAMESPACE::optimization::hash_combine(
              seed, CSEContainerHash<int64_t>(), n->is(name));
          break;
        case ONNX_NAMESPACE::AttributeKind::s:
          ONNX_NAMESPACE::optimization::hash_combine(seed, string_hasher,
                                                     n->s(name));
          break;
        case ONNX_NAMESPACE::AttributeKind::ss:
          ONNX_NAMESPACE::optimization::hash_combine(
              seed, CSEContainerHash<std::string>(), n->ss(name));
          break;
        default:
          throw std::runtime_error(ONNX_NAMESPACE::optimization::Str(
              "no support hash type: ", ONNX_NAMESPACE::toString(kind)));
          break;
      }
    }
    ONNX_NAMESPACE::optimization::hash_combine(seed, size_t_hasher,
                                               n->outputs().size());
    return seed;
  }
};

}  // namespace optimization
}  // namespace ONNX_NAMESPACE
