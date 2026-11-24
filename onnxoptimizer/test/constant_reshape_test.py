# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest
from onnx import TensorProto, checker, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

import onnxoptimizer


def make_model(input_shape: list[int], target_shape: list[int], allowzero: int | None):
    values = np.arange(np.prod(input_shape, dtype=np.int64), dtype=np.float32).reshape(
        input_shape
    )
    attributes = {} if allowzero is None else {"allowzero": allowzero}
    graph = helper.make_graph(
        [
            helper.make_node(
                "Constant", [], ["data"], value=numpy_helper.from_array(values)
            ),
            helper.make_node(
                "Constant",
                [],
                ["shape"],
                value=numpy_helper.from_array(np.array(target_shape, dtype=np.int64)),
            ),
            helper.make_node("Reshape", ["data", "shape"], ["Y"], **attributes),
        ],
        "constant_reshape",
        [],
        [
            helper.make_tensor_value_info(
                "Y", TensorProto.FLOAT, [None] * len(target_shape)
            )
        ],
    )
    return helper.make_model(
        graph, ir_version=7, opset_imports=[helper.make_opsetid("", 14)]
    )


@pytest.mark.parametrize(
    "input_shape,target_shape,allowzero,expected_shape",
    [
        ([2, 3], [0, 3], None, [2, 3]),
        ([2, 3], [0, -1], 0, [2, 3]),
        ([2, 3], [3, -1], None, [3, 2]),
        ([2, 3], [-1], None, [6]),
        ([0, 3], [0, 3], 1, [0, 3]),
        ([0, 3], [-1, 3], None, [0, 3]),
        ([], [], None, []),
        ([1], [], None, []),
    ],
)
def test_constant_reshape_preserves_values_and_shape(
    input_shape, target_shape, allowzero, expected_shape
) -> None:
    model = make_model(input_shape, target_shape, allowzero)
    checker.check_model(model)
    expected = ReferenceEvaluator(model).run(None, {})[0]
    optimized = onnxoptimizer.optimize(model, ["fuse_constant_reshape"])
    checker.check_model(optimized)
    assert all(node.op_type != "Reshape" for node in optimized.graph.node)
    actual = ReferenceEvaluator(optimized).run(None, {})[0]
    assert list(actual.shape) == expected_shape
    assert [
        dimension.dim_value
        for dimension in optimized.graph.output[0].type.tensor_type.shape.dim
    ] == expected_shape
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "input_shape,target_shape,allowzero",
    [
        ([0, 3], [0, -1], 0),
        ([0, 3], [0, -1], 1),
        ([2, 3], [-1, -1], None),
        ([2, 3], [-2, 3], None),
        ([2, 3], [4, -1], None),
        ([2, 3], [5], None),
        ([2, 3], [1, 1, 0], None),
        ([2, 3], [2**32, 2**32], None),
    ],
)
def test_constant_reshape_invalid_shape_is_not_folded(
    input_shape, target_shape, allowzero
) -> None:
    model = make_model(input_shape, target_shape, allowzero)
    optimized = onnxoptimizer.optimize(model, ["fuse_constant_reshape"])
    assert any(node.op_type == "Reshape" for node in optimized.graph.node)
