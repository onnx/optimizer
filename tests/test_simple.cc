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
    const onnx::Status status = onnx::OnnxParser::Parse(model, graph_str);
    EXPECT_TRUE(status.IsOK());
    auto optimized_model = onnx::optimization::Optimize(model, {"eliminate_nop_reshape", "eliminate_deadend"});

    ASSERT_EQ(optimized_model.graph().node().size(), 1);
    ASSERT_EQ(optimized_model.graph().node()[0].op_type(), "Identity");
}
