/*
 * SPDX-License-Identifier: Apache-2.0
 */

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#include "eliminate_nop_with_unit.h"
#include "onnx/defs/tensor_util.h"

using namespace ONNX_NAMESPACE;
using namespace optimization;

namespace {

#define PROTO_DTYPE_TO_CPP_DTYPE_LIST(_)             \
  _(TensorProto_DataType_FLOAT, float, 1, 0)         \
  _(TensorProto_DataType_INT32, int32_t, 1, 0)       \
  _(TensorProto_DataType_INT64, int64_t, 1, 0)       \
  _(TensorProto_DataType_DOUBLE, double, 1, 0)       \
  _(TensorProto_DataType_FLOAT16, int32_t, 15360, 0) \
  _(TensorProto_DataType_UINT8, int32_t, 1, 0)       \
  _(TensorProto_DataType_INT8, int32_t, 1, 0)        \
  _(TensorProto_DataType_UINT16, int32_t, 1, 0)      \
  _(TensorProto_DataType_INT16, int32_t, 1, 0)       \
  _(TensorProto_DataType_UINT32, uint64_t, 1, 0)     \
  _(TensorProto_DataType_UINT64, uint64_t, 1, 0)

enum Unit {
  ZERO,
  ONE,
};

template <TensorProto_DataType, Unit>
struct IdentityUnitTraits;

#define IDENTITY_UNIT_TRAITS(pb_dtype, cpp_dtype, one, zero)                 \
  template <>                                                                \
  struct IdentityUnitTraits<pb_dtype, ZERO> {                                \
    using cpp_type = cpp_dtype;                                              \
    static bool isAllValue(const std::vector<cpp_type>& data) {              \
      return std::all_of(data.cbegin(), data.cend(), [](const cpp_type& v) { \
        return v == cpp_type{zero};                                          \
      });                                                                    \
    }                                                                        \
  };                                                                         \
  template <>                                                                \
  struct IdentityUnitTraits<pb_dtype, ONE> {                                 \
    using cpp_type = cpp_dtype;                                              \
    static bool isAllValue(const std::vector<cpp_type>& data) {              \
      return std::all_of(data.cbegin(), data.cend(), [](const cpp_type& v) { \
        return v == cpp_type{one};                                           \
      });                                                                    \
    }                                                                        \
  };

PROTO_DTYPE_TO_CPP_DTYPE_LIST(IDENTITY_UNIT_TRAITS)

#undef IDENTITY_UNIT_TRAITS
#undef PROTO_DTYPE_TO_CPP_DTYPE_LIST

template <Unit unit>
struct TensorValueCheck {
  static bool all(const Tensor& tensor) {
    int elem_type = tensor.elem_type();
#define CASE_BRANCH_CONTENT(pb_dtype)                                         \
  case pb_dtype: {                                                            \
    const std::vector<typename IdentityUnitTraits<pb_dtype, unit>::cpp_type>  \
        data =                                                                \
            ParseData<typename IdentityUnitTraits<pb_dtype, unit>::cpp_type>( \
                &tensor);                                                     \
    return IdentityUnitTraits<pb_dtype, unit>::isAllValue(data);              \
  }

    switch (elem_type) {
      CASE_BRANCH_CONTENT(TensorProto_DataType_FLOAT)
      CASE_BRANCH_CONTENT(TensorProto_DataType_INT32)
      CASE_BRANCH_CONTENT(TensorProto_DataType_INT64)
      CASE_BRANCH_CONTENT(TensorProto_DataType_DOUBLE)
      CASE_BRANCH_CONTENT(TensorProto_DataType_FLOAT16)
      CASE_BRANCH_CONTENT(TensorProto_DataType_UINT8)
      CASE_BRANCH_CONTENT(TensorProto_DataType_INT8)
      CASE_BRANCH_CONTENT(TensorProto_DataType_UINT16)
      CASE_BRANCH_CONTENT(TensorProto_DataType_INT16)
      // TODO: support uint64
      //  CASE_BRANCH_CONTENT(TensorProto_DataType_UINT32)
      //  CASE_BRANCH_CONTENT(TensorProto_DataType_UINT64)
    }
#undef CASE_BRANCH_CONTENT
    return false;
  }
};

bool isConstantTensor(Graph& graph, const std::string& name) {
  auto& initializer_names = graph.initializer_names();
  return std::find(initializer_names.cbegin(), initializer_names.cend(),
                   name) != initializer_names.cend();
}

bool isABroadcastToB(const std::vector<int64_t>& dims_a,
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

#define NODE_SUPPORT_COMMUTATIVE_LAW_LIST(_) \
  _(kMul, ONE)                               \
  _(kAdd, ZERO)

#define NODE_NO_SUPPORT_COMMUTATIVE_LAW_LIST(_) \
  _(kDiv, ONE)                                  \
  _(kSub, ZERO)                                 \
  _(kPow, ONE)

template <uint32_t>
struct NodeKindTraits;

#define NODE_KIND_TRAITS_SUPPORT_COMMUTATIVE_LAW(node_kind, unit)             \
  template <>                                                                 \
  struct NodeKindTraits<node_kind> {                                          \
    static bool isPatternConstantOfshape(Value* a_value, Value* b_value) {    \
      Node* a_node = a_value->node();                                         \
      if (a_node->kind() != Symbol("ConstantOfShape")) {                      \
        return false;                                                         \
      }                                                                       \
      if (!a_node->hasAttribute(kvalue) &&                                    \
          !IdentityUnitTraits<TensorProto_DataType_FLOAT, unit>::isAllValue(  \
              {0.0f})) {                                                      \
        return false;                                                         \
      } else {                                                                \
        Tensor t = a_node->t(kvalue);                                         \
        if (!TensorValueCheck<unit>::all(t)) {                                \
          return false;                                                       \
        }                                                                     \
      }                                                                       \
      Node* parent_node = a_node->input()->node();                            \
      return parent_node->kind() == Symbol("Shape") &&                        \
             parent_node->input()->node() == b_value->node();                 \
    }                                                                         \
    static bool isPatternConstantOfshape(Node* node) {                        \
      auto& a_value = node->inputs()[0];                                      \
      auto& b_value = node->inputs()[1];                                      \
      return isPatternConstantOfshape(a_value, b_value) ||                    \
             isPatternConstantOfshape(b_value, a_value);                      \
    }                                                                         \
    static bool patternMatchPredicate(Node* node) {                           \
      return node->inputs()[0]->node()->kind() == kParam ||                   \
             node->inputs()[1]->node()->kind() == kParam ||                   \
             isPatternConstantOfshape(node);                                  \
    }                                                                         \
    static bool runTransform(Node* node, Graph& graph,                        \
                             NodeDestroyType& destroy_current) {              \
      auto& a_value = node->inputs()[0];                                      \
      auto& b_value = node->inputs()[1];                                      \
      const auto a_name = a_value->uniqueName();                              \
      const auto b_name = b_value->uniqueName();                              \
      const auto a_tensor = graph.getInitializer(a_name);                     \
      const auto b_tensor = graph.getInitializer(b_name);                     \
      bool replacing_success = false;                                         \
      if (isConstantTensor(graph, a_name) &&                                  \
          TensorValueCheck<unit>::all(*a_tensor)) {                           \
        replacing_success =                                                   \
            isABroadcastToB(a_tensor->sizes(), b_value->sizes()) &&           \
            tryReplacingAllUsesWith(node->output(), b_value);                 \
      }                                                                       \
      if (!replacing_success && isConstantTensor(graph, b_name) &&            \
          TensorValueCheck<unit>::all(*b_tensor)) {                           \
        replacing_success =                                                   \
            isABroadcastToB(b_tensor->sizes(), a_value->sizes()) &&           \
            tryReplacingAllUsesWith(node->output(), a_value);                 \
      }                                                                       \
      if (!replacing_success && isPatternConstantOfshape(a_value, b_value)) { \
        replacing_success = tryReplacingAllUsesWith(node->output(), b_value); \
      }                                                                       \
      if (!replacing_success && isPatternConstantOfshape(b_value, a_value)) { \
        replacing_success = tryReplacingAllUsesWith(node->output(), a_value); \
      }                                                                       \
      if (!replacing_success) {                                               \
        return false;                                                         \
      }                                                                       \
      destroy_current = NodeDestroyType::DestroyOne;                          \
      return true;                                                            \
    }                                                                         \
  };

#define NODE_KIND_TRAITS_NOSUPPORT_COMMUTATIVE_LAW(node_kind, unit)           \
  template <>                                                                 \
  struct NodeKindTraits<node_kind> {                                          \
    static bool isPatternConstantOfshape(Value* a_value, Value* b_value) {    \
      Node* a_node = a_value->node();                                         \
      if (a_node->kind() != Symbol("ConstantOfShape")) {                      \
        return false;                                                         \
      }                                                                       \
      if (!a_node->hasAttribute(kvalue) &&                                    \
          !IdentityUnitTraits<TensorProto_DataType_FLOAT, unit>::isAllValue(  \
              {0.0f})) {                                                      \
        return false;                                                         \
      } else {                                                                \
        Tensor t = a_node->t(kvalue);                                         \
        if (!TensorValueCheck<unit>::all(t)) {                                \
          return false;                                                       \
        }                                                                     \
      }                                                                       \
      Node* parent_node = a_node->input()->node();                            \
      return parent_node->kind() == Symbol("Shape") &&                        \
             parent_node->input()->node() == b_value->node();                 \
    }                                                                         \
    static bool isPatternConstantOfshape(Node* node) {                        \
      auto& a_value = node->inputs()[0];                                      \
      auto& b_value = node->inputs()[1];                                      \
      return isPatternConstantOfshape(b_value, a_value);                      \
    }                                                                         \
    static bool patternMatchPredicate(Node* node) {                           \
      return node->inputs()[1]->node()->kind() == kParam ||                   \
             isPatternConstantOfshape(node);                                  \
    }                                                                         \
    static bool runTransform(Node* node, Graph& graph,                        \
                             NodeDestroyType& destroy_current) {              \
      auto& a_value = node->inputs()[0];                                      \
      auto& b_value = node->inputs()[1];                                      \
      const auto b_name = b_value->uniqueName();                              \
      const auto b_tensor = graph.getInitializer(b_name);                     \
      bool replacing_success = false;                                         \
      if (isConstantTensor(graph, b_name) &&                                  \
          TensorValueCheck<unit>::all(*b_tensor)) {                           \
        replacing_success =                                                   \
            isABroadcastToB(b_tensor->sizes(), a_value->sizes()) &&           \
            tryReplacingAllUsesWith(node->output(), a_value);                 \
      }                                                                       \
      if (!replacing_success && isPatternConstantOfshape(node)) {             \
        replacing_success = tryReplacingAllUsesWith(node->output(), a_value); \
      }                                                                       \
      if (!replacing_success) {                                               \
        return false;                                                         \
      }                                                                       \
      destroy_current = NodeDestroyType::DestroyOne;                          \
      return true;                                                            \
    }                                                                         \
  };

NODE_SUPPORT_COMMUTATIVE_LAW_LIST(NODE_KIND_TRAITS_SUPPORT_COMMUTATIVE_LAW)
NODE_NO_SUPPORT_COMMUTATIVE_LAW_LIST(NODE_KIND_TRAITS_NOSUPPORT_COMMUTATIVE_LAW)

#undef NODE_KIND_TRAITS_SUPPORT_COMMUTATIVE_LAW
#undef NODE_KIND_TRAITS_NOSUPPORT_COMMUTATIVE_LAW
#undef NODE_SUPPORT_COMMUTATIVE_LAW_LIST
#undef NODE_NO_SUPPORT_COMMUTATIVE_LAW_LIST

}  // namespace

bool EliminateOpWithUnit::patternMatchPredicate(Node* node) {
  uint32_t kind = node->kind();
  switch (kind) {
    case kMul:
      return NodeKindTraits<kMul>::patternMatchPredicate(node);
    case kDiv:
      return NodeKindTraits<kDiv>::patternMatchPredicate(node);
    case kAdd:
      return NodeKindTraits<kAdd>::patternMatchPredicate(node);
    case kSub:
      return NodeKindTraits<kSub>::patternMatchPredicate(node);
    case kPow:
      return NodeKindTraits<kPow>::patternMatchPredicate(node);
    default:
      return false;
  }
}

bool EliminateOpWithUnit::runTransform(Node* node, Graph& graph,
                                       NodeDestroyType& destroy_current) {
  uint32_t kind = node->kind();
  switch (kind) {
    case kMul:
      return NodeKindTraits<kMul>::runTransform(node, graph, destroy_current);
    case kDiv:
      return NodeKindTraits<kDiv>::runTransform(node, graph, destroy_current);
    case kAdd:
      return NodeKindTraits<kAdd>::runTransform(node, graph, destroy_current);
    case kSub:
      return NodeKindTraits<kSub>::runTransform(node, graph, destroy_current);
    case kPow:
      return NodeKindTraits<kPow>::runTransform(node, graph, destroy_current);
    default:
      return false;
  }
}
