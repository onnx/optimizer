# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import onnx
import onnxoptimizer
import pytest
from onnx import TensorProto, checker, helper

try:
    import onnxruntime as ort
except ImportError:
    ort = None


def _run(model):
    if ort is None:
        return None
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    ).run(None, {})


@pytest.mark.parametrize(
    "attribute_names", [("out_a", "out_a"), ("out_b", "out_b"), ("tensor_a", "tensor_b")]
)
def test_extract_constant_graph_outputs(attribute_names):
    # Issue #350: TensorProto names inside attributes are not value names.
    nodes = [
        helper.make_node(
            "Constant",
            [],
            [output],
            value=helper.make_tensor(name, TensorProto.FLOAT, [1], [value]),
        )
        for output, name, value in zip(("out_a", "out_b"), attribute_names, (1.0, 2.0))
    ]
    model = helper.make_model(
        helper.make_graph(
            nodes,
            "constant_outputs",
            [],
            [
                helper.make_tensor_value_info(name, TensorProto.FLOAT, [1])
                for name in ("out_a", "out_b")
            ],
        ),
        opset_imports=[helper.make_opsetid("", 13)],
        ir_version=10,
    )
    checker.check_model(model, full_check=True)
    expected = _run(model)

    optimized = onnxoptimizer.optimize(model, ["extract_constant_to_initializer"])
    checker.check_model(optimized, full_check=True)
    names = [tensor.name for tensor in optimized.graph.initializer]
    assert len(names) == len(set(names))
    assert set(names) == {"out_a", "out_b"}
    if expected is not None:
        for actual, original in zip(_run(optimized), expected):
            np.testing.assert_array_equal(actual, original)


def test_extract_constant_shared_uses():
    constant = helper.make_node(
        "Constant",
        [],
        ["constant"],
        value=helper.make_tensor("left", TensorProto.FLOAT, [1], [3.0]),
    )
    model = helper.make_model(
        helper.make_graph(
            [
                constant,
                helper.make_node("Identity", ["constant"], ["left"]),
                helper.make_node("Identity", ["constant"], ["right"]),
            ],
            "shared_constant",
            [],
            [
                helper.make_tensor_value_info(name, TensorProto.FLOAT, [1])
                for name in ("left", "right")
            ],
        ),
        opset_imports=[helper.make_opsetid("", 13)],
        ir_version=10,
    )
    checker.check_model(model, full_check=True)
    expected = _run(model)
    optimized = onnxoptimizer.optimize(model, ["extract_constant_to_initializer"])
    checker.check_model(optimized, full_check=True)
    assert [tensor.name for tensor in optimized.graph.initializer] == ["constant"]
    assert [node.input[0] for node in optimized.graph.node] == ["constant", "constant"]
    if expected is not None:
        for actual, original in zip(_run(optimized), expected):
            np.testing.assert_array_equal(actual, original)
