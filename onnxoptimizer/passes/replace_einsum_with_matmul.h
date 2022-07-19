/*
 * SPDX-License-Identifier: Apache-2.0
 */

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

// Before:
//   Z = einsum(X,Y)
// After:
//   Z = matmul(X,Y)
//   or  Y1 = transpose(Y), Z = matmul(X,Y1)
// the pass can handle the case when:
//     case 1: equation represents matmul, e.g: "bhij,bhjd->bhid"
//     case 2: equation represents transpose matmul, e.g: "bhid,bhjd->bhij"

#include <numeric>

#include "onnx/defs/tensor_util.h"
#include "onnxoptimizer/pass.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct ReplaceEinsumWithMatmul final : public PredicateBasedPass {
  explicit ReplaceEinsumWithMatmul()
      : PredicateBasedPass(PassType::Replace, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override {
    return "replace_einsum_with_matmul";
  }

  bool patternMatchPredicate(Node* node) override {
    return CheckKind(node, "Einsum") && node->inputs().size() == 2 &&
           std::all_of(node->inputs().begin(), node->inputs().end(),
                       [](const Value* v) {
                         switch (v->elemType()) {
                             // matmul support these dtype
                           case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
                           case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE:
                           case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
                           case ONNX_NAMESPACE::TensorProto_DataType_INT32:
                           case ONNX_NAMESPACE::TensorProto_DataType_UINT32:
                           case ONNX_NAMESPACE::TensorProto_DataType_INT64:
                           case ONNX_NAMESPACE::TensorProto_DataType_UINT64:
                             return true;
                         }
                         return false;
                       });
  }

  template <typename T>
  bool isEqual(const T& a, const T& b) {
    return a == b;
  }

  template <typename T>
  bool isEqual(const T& a, const T& b, const T& c) {
    return isEqual(a, b) && isEqual(a, c);
  }

  bool runTransform(Node* n, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    ONNX_ASSERT(n->hasAttribute(Symbol("equation")));
    std::string equation = n->s(Symbol("equation"));
    // remove space
    equation.erase(std::remove(equation.begin(), equation.end(), ' '),
                   equation.end());
    auto mid_index = equation.find("->");
    if (mid_index == std::string::npos) {
      return false;
    }
    const auto left_equation = equation.substr(0, mid_index);
    const auto right_equation = equation.substr(mid_index + 2);
    mid_index = left_equation.find(",");
    if (mid_index == std::string::npos) {
      return false;
    }
    // reference https://github.com/onnx/onnx/blob/main/docs/Operators.md#Einsum
    const auto term1 = left_equation.substr(0, mid_index);
    const auto term2 = left_equation.substr(mid_index + 1);

    if (term1.size() < 2 ||
        !isEqual(term1.size(), term2.size(), right_equation.size())) {
      return false;
    }

    auto is_lower_letter = [](char c) {
      return std::isalpha(c) && std::islower(c);
    };

    const int shape_size = term1.size();
    for (int i = 0; i < shape_size; ++i) {
      // the term should only contain lower case letters
      if (!is_lower_letter(term1[i]) || !is_lower_letter(term2[i]) ||
          !is_lower_letter(right_equation[i])) {
        return false;
      }
      // the batch dim should be equal
      if ((i < shape_size - 2) &&
          !isEqual(term1[i], term2[i], right_equation[i])) {
        return false;
      }
    }
    char term1_m = term1[shape_size - 2];
    char term2_k = term2[shape_size - 2];
    char result_m = right_equation[shape_size - 2];

    char term1_k = term1[shape_size - 1];
    char term2_n = term2[shape_size - 1];
    char result_n = right_equation[shape_size - 1];

    bool need_transpose = false;
    if (isEqual(term1_m, result_m) && isEqual(term1_k, term2_k) &&
        isEqual(term2_n, result_n)) {  // "ij,jd->id"
      need_transpose = false;
    } else if (isEqual(term1_m, result_m) && isEqual(term1_k, term2_n) &&
               isEqual(term2_k, result_n)) {  // "id,jd->ij"
      need_transpose = true;
    } else {
      return false;
    }

    Node* matmul_node = graph.create(kMatMul, 1);
    matmul_node->addInput(n->inputs()[0]);
    if (need_transpose) {
      Node* transpose_node = graph.create(kTranspose, 1);
      transpose_node->addInput(n->inputs()[1]);
      transpose_node->output()->setUniqueName(
          ONNX_NAMESPACE::to_string(graph.getNextUnique()));
      std::vector<int64_t> perm(shape_size, 0);
      for (int i = 0; i < shape_size - 2; ++i) {
        perm[i] = i;
      }
      perm[shape_size - 1] = shape_size - 2;
      perm[shape_size - 2] = shape_size - 1;
      transpose_node->is_(kperm, std::move(perm));
      matmul_node->addInput(transpose_node->output());
      transpose_node->insertBefore(n);
    } else {
      matmul_node->addInput(n->inputs()[1]);
    }
    matmul_node->insertBefore(n);

    const bool replacing_success = tryReplacingAllUsesWith(n, matmul_node);
    if (!replacing_success) {
      return false;
    }
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

}  // namespace optimization
}  // namespace ONNX_NAMESPACE
