# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import onnx

import onnxoptimizer


def test_seq_type():
    sequence = onnx.helper.make_tensor_sequence_value_info(
        "sequence", onnx.TensorProto.INT64, None
    )
    tensor = onnx.helper.make_tensor_value_info(
        "tensor", onnx.TensorProto.INT64, [3]
    )
    output = onnx.helper.make_tensor_sequence_value_info(
        "output_sequence", onnx.TensorProto.INT64, None
    )
    orig = onnx.helper.make_model(
        onnx.helper.make_graph(
            [
                onnx.helper.make_node(
                    "SequenceInsert", ["sequence", "tensor"], ["output_sequence"]
                )
            ],
            "sequence_insert",
            [sequence, tensor],
            [output],
        ),
        opset_imports=[onnx.helper.make_opsetid("", 11)],
    )
    optimized = onnxoptimizer.optimize(orig)
    assert optimized.graph.input[0].type == orig.graph.input[0].type
