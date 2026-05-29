// SPDX-FileCopyrightText: ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <onnxoptimizer/optimize.h>
#include <onnx/defs/parser.h>
#include <onnx/checker.h>

TEST(OptimizerTest, NopReshape) {
    const char* graph_str = R"(
        <
            ir_version: 7,
            opset_import: [ "": 10]
        >
        agraph (float[5, 7] X) => (float[5, 7] Z)
        {
            Shape = Constant<value=int64[2]{5, -1}> ()
            Y = Reshape (X, Shape)
            Z = Identity(Y)
        }
    )";
    onnx::ModelProto model;
    const onnx::Status status = onnx::OnnxParser::Parse(model, graph_str);
    EXPECT_TRUE(status.IsOK());
    auto optimized_model = onnx::optimization::Optimize(model, {"eliminate_nop_reshape", "eliminate_deadend"});

    ASSERT_EQ(optimized_model.graph().node().size(), 1);
    ASSERT_EQ(optimized_model.graph().node()[0].op_type(), "Identity");
}

TEST(OptimizerTest, SplitPredictPreservesElemType) {
    // Test that split_predict preserves elem_type for intermediate values
    // This reproduces the bug where intermediate values without value_info
    // cause split_predict to generate invalid models with missing elem_type
    const char* graph_str = R"(
        <
            ir_version: 7,
            opset_import: [ "": 13]
        >
        agraph (float[2] X) => (float[2] Y)
        {
            # Pure operation that can go to init net
            one = Constant<value: tensor = float[2] {1.0, 1.0}>()
            added = Add(X, one)
            
            # Impure operation that must stay in predict net
            # This uses 'added', making it a boundary value
            random = RandomUniform<dtype: int = 1, shape: ints = [2]>()
            Y = Add(random, added)
        }
    )";
    
    onnx::ModelProto model;
    const onnx::Status status = onnx::OnnxParser::Parse(model, graph_str);
    EXPECT_TRUE(status.IsOK());
    
    // Run split_predict optimization
    auto optimized_model = onnx::optimization::Optimize(model, {"split_predict"});
    
    // Verify the model is valid - this will catch missing elem_type
    try {
        onnx::checker::check_model(optimized_model);
    } catch (const std::exception& e) {
        FAIL() << "Optimized model failed validation: " << e.what();
    }
    
    // Verify all inputs have valid elem_type (not UNDEFINED)
    for (const auto& input : optimized_model.graph().input()) {
        ASSERT_TRUE(input.has_type()) << "Input " << input.name() << " missing type";
        ASSERT_TRUE(input.type().has_tensor_type()) << "Input " << input.name() << " missing tensor_type";
        // Check elem_type is not UNDEFINED
        ASSERT_NE(input.type().tensor_type().elem_type(), 
                  ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED) 
            << "Input " << input.name() << " has UNDEFINED elem_type";
    }
}
