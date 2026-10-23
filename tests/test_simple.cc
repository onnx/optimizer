// SPDX-FileCopyrightText: ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <onnxoptimizer/optimize.h>
#include <onnx/defs/parser.h>

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
    const auto status = onnx::OnnxParser::Parse(model, graph_str);
    EXPECT_TRUE(status.IsOK());
    auto optimized_model = onnx::optimization::Optimize(model, {"eliminate_nop_reshape", "eliminate_deadend"});

    ASSERT_EQ(optimized_model.graph().node().size(), 1);
    ASSERT_EQ(optimized_model.graph().node()[0].op_type(), "Identity");
}

// Exercises the in-place C++ IR (Graph) entry point: import a ModelProto into
// the IR, optimize the Graph directly (no ModelProto round-trip inside the
// optimizer), then export to confirm the passes mutated the graph in place.
TEST(OptimizerTest, OptimizeGraphInPlace) {
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
    const auto status = onnx::OnnxParser::Parse(model, graph_str);
    EXPECT_TRUE(status.IsOK());

    // ModelProto -> C++ IR.
    std::shared_ptr<onnx::Graph> g(onnx::ImportModelProto(model));
    ASSERT_NE(g.get(), nullptr);

    // Optimize the IR in place.
    onnx::optimization::OptimizeGraph(
        *g, {"eliminate_nop_reshape", "eliminate_deadend"});

    // Export the mutated IR to confirm the passes ran on the graph directly.
    onnx::ModelProto optimized_model = onnx::PrepareOutput(model);
    onnx::ExportModelProto(&optimized_model, g);

    ASSERT_EQ(optimized_model.graph().node().size(), 1);
    ASSERT_EQ(optimized_model.graph().node()[0].op_type(), "Identity");
}
