/*
 * SPDX-License-Identifier: Apache-2.0
 */

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

// Before:
//	 conv = Conv()
//   bn = BatchNormalization()
//
// After:
//	 bn is deleted
//   new inputs/initializers to conv are added to graph
//   any no longer used inputs/initializers are erased from graph
//
//	 this pass can handle the case satisfy all following conditions:
//	   condition 1: Run in testing mode
//     condition 2: Inputs 1 - 4 of bn are all initializer_size
//     condition 3: Output of initial conv has no other uses
//     condition 3: Currently works for only DOUBLE, FLOAT32 tensor types
//
// Formula for transformation
// $$ X_{bn} = \frac{s(X - m)}{\sqrt{\sigma + \epsilon}} + b_{bn}$$
// $$ X_{conv} = X * W + b_{conv} $$
// thus, substituting $X$ with $X_{conv}$ in the BN equation we get:
// $$X_{bn} = X * \frac{sW}{\sqrt{\sigma + \epsilon}} + \frac{s(b_{conv} -
// m)}{\sqrt{\sigma + \epsilon}} + b_{bn}$$ or
// $$ W' = W\frac{s}{\sqrt{\sigma + \epsilon}}$$
// $$ b' = (b_{conv} - m)\frac{s}{\sqrt{\sigma + \epsilon}} + b_{bn}$$

#include "onnx/common/assertions.h"
#include "onnxoptimizer/pass.h"
#include "onnxoptimizer/passes/pass_util.h"

namespace ONNX_NAMESPACE {
namespace optimization {
// TODO: Currently broken for complex values and float16
struct FuseBNIntoConv final : public PredicateBasedPass {
  explicit FuseBNIntoConv()
      : PredicateBasedPass(PassType::Fuse, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "fuse_bn_into_conv";
  }

  bool modify_conv(Node* conv, Node* bn, Graph& graph) {
    const auto& bn_inputs = bn->inputs();
    const auto& conv_inputs = conv->inputs();

    auto bn_scale = *FetchConstantTensor(bn_inputs[1]);
    auto bn_bais = *FetchConstantTensor(bn_inputs[2]);
    auto bn_mean = *FetchConstantTensor(bn_inputs[3]);
    auto bn_var = *FetchConstantTensor(bn_inputs[4]);
    auto conv_W = *FetchConstantTensor(conv_inputs[1]);
    bn_scale.setName(ONNX_NAMESPACE::to_string(graph.getNextUnique()));
    bn_bais.setName(ONNX_NAMESPACE::to_string(graph.getNextUnique()));
    bn_mean.setName(ONNX_NAMESPACE::to_string(graph.getNextUnique()));
    bn_var.setName(ONNX_NAMESPACE::to_string(graph.getNextUnique()));
    conv_W.setName(ONNX_NAMESPACE::to_string(graph.getNextUnique()));

    /// scale bais mean var must be the same shape (C)
    ONNX_ASSERT(bn_scale.sizes() == bn_bais.sizes());
    ONNX_ASSERT(bn_scale.sizes() == bn_mean.sizes());
    ONNX_ASSERT(bn_scale.sizes() == bn_var.sizes());
    ONNX_ASSERT(bn_scale.sizes().size() == 1);
    int64_t C = bn_scale.sizes()[0];
    ONNX_ASSERT(conv_W.sizes().size() > 2 && conv_W.sizes()[0] == C);
    if (bn_scale.elem_type() != bn_bais.elem_type() ||
        bn_scale.elem_type() != bn_mean.elem_type() ||
        bn_scale.elem_type() != bn_var.elem_type() ||
        bn_scale.elem_type() != conv_W.elem_type()) {
      return false;
    }

    Value* conv_bias = nullptr;
    if (conv_inputs.size() == 3) {
      if (!IsConstantTensor(conv_inputs[2])) {
        return false;
      }
      auto bc_t = *FetchConstantTensor(conv_inputs[2]);
      bc_t.setName(ONNX_NAMESPACE::to_string(graph.getNextUnique()));
      ONNX_ASSERT(bc_t.sizes() == bn_scale.sizes());
      conv_bias = graph.addInitializerAndCreateValue(bc_t);
    } else {
      Tensor bc_t;
      bc_t.elem_type() = ONNX_NAMESPACE::TensorProto_DataType_FLOAT;
      bc_t.sizes().push_back(C);
      for (int i = 0; i < C; ++i) {
        bc_t.floats().push_back(float{0});
      }
      conv_bias = graph.addInitializerAndCreateValue(bc_t);
    }

    /// scalar
    Tensor eps_t;
    eps_t.elem_type() = ONNX_NAMESPACE::TensorProto_DataType_FLOAT;
    eps_t.floats().push_back(GetValueFromAttrWithDefault(bn, kepsilon, 1e-5f));
    Value* eps = graph.addInitializerAndCreateValue(eps_t);

    Node* cast = graph.create(kCast, 1);
    cast->addInput(eps);
    cast->i_(kto, bn_var.elem_type());
    cast->insertBefore(conv);

    Node* var_add = graph.create(kAdd, 1);
    var_add->insertAfter(cast);
    var_add->addInput(graph.addInitializerAndCreateValue(bn_var));
    var_add->addInput(cast->output());

    Node* sqrt = graph.create(kSqrt, 1);
    sqrt->insertAfter(var_add);
    sqrt->addInput(var_add->output());

    Node* scale = graph.create(kDiv, 1);
    scale->insertAfter(sqrt);
    scale->addInput(graph.addInitializerAndCreateValue(bn_scale));
    scale->addInput(sqrt->output());

    Node* unsqueeze = graph.create(kUnsqueeze, 1);
    unsqueeze->insertAfter(scale);
    unsqueeze->addInput(scale->output());
    std::vector<int64_t> insert_dims;
    for (int i = 1; i < conv_W.sizes().size(); ++i) {
      insert_dims.push_back(i);
    }
    if (getOpsetVersion(graph) >= 13) {
      Tensor shape_s_t;
      shape_s_t.elem_type() = ONNX_NAMESPACE::TensorProto_DataType_INT64;
      shape_s_t.sizes().push_back(insert_dims.size());
      shape_s_t.int64s() = insert_dims;
      unsqueeze->addInput(graph.addInitializerAndCreateValue(shape_s_t));
    } else {
      unsqueeze->is_(kaxes, std::move(insert_dims));
    }

    Node* mul_w = graph.create(kMul, 1);
    mul_w->insertAfter(unsqueeze);
    mul_w->addInput(graph.addInitializerAndCreateValue(conv_W));
    mul_w->addInput(unsqueeze->output());

    Node* cast1 = graph.create(kCast, 1);
    cast1->insertAfter(mul_w);
    cast1->addInput(conv_bias);
    cast1->i_(kto, bn_mean.elem_type());

    Node* sub = graph.create(kSub, 1);
    sub->insertAfter(cast1);
    sub->addInput(cast1->output());
    sub->addInput(graph.addInitializerAndCreateValue(bn_mean));

    Node* mul = graph.create(kMul, 1);
    mul->insertAfter(sub);
    mul->addInput(sub->output());
    mul->addInput(scale->output());

    Node* bias_add = graph.create(kAdd, 1);
    bias_add->insertAfter(mul);
    bias_add->addInput(mul->output());
    bias_add->addInput(graph.addInitializerAndCreateValue(bn_bais));

    Value* old_w_value = conv_inputs[1];
    conv->replaceInput(1, mul_w->output());
    if (old_w_value->uses().size() == 0) {
      graph.eraseInitializerAndInput(old_w_value);
    }

    if (conv_inputs.size() == 3) {
      Value* old_b_value = conv_inputs[2];
      conv->replaceInput(2, bias_add->output());
      if (old_b_value->uses().size() == 0) {
        graph.eraseInitializerAndInput(old_b_value);
      }
    } else {
      conv->addInput(bias_add->output());
    }
    return true;
  }

  bool patternMatchPredicate(Node* n) override {
    return CheckKind(n, kBatchNormalization, 0, kConv) &&
           GetValueFromAttrWithDefault(n, "training_mode", (int64_t)0) == 0 &&
           n->input(0)->uses().size() == 1 && n->outputs().size() == 1 &&
           IsConstantTensor(n, 1) && IsConstantTensor(n, 2) &&
           IsConstantTensor(n, 3) && IsConstantTensor(n, 4) &&
           IsConstantTensor(PrevNode(n, 0), 1);
  }
  bool runTransform(Node* n, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    Node* bn = n;
    Node* conv = PrevNode(n, 0);
    auto origInput = bn->inputs()[0];
    if (!modify_conv(conv, bn, graph)) {
      destroy_current = NodeDestroyType::DestroyZero;
      return false;
    }
    // clean
    for (int i = 4; i >= 1; --i) {
      if (bn->inputs()[i]->uses().size() == 1) {
        auto input = bn->inputs()[i];
        bn->removeInput(i);
        graph.eraseInitializerAndInput(input);
      }
    }
    const bool replacing_success =
        tryReplacingAllUsesWith(bn->output(), origInput);
    if (!replacing_success) {
      return false;
    }
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

}  // namespace optimization
}  // namespace ONNX_NAMESPACE
