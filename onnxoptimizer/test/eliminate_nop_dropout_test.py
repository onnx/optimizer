# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest
from onnx import TensorProto, checker, helper
from onnx.reference import ReferenceEvaluator

import onnxoptimizer


@pytest.mark.parametrize("opset", [10, 11])
@pytest.mark.parametrize("ratio", [0.0, 0.5])
@pytest.mark.parametrize(
    "mask_use", ["omitted", "empty", "unused", "exported", "consumed", "mask_only"]
)
def test_eliminate_nop_dropout_preserves_mask(opset: int, ratio: float, mask_use: str) -> None:
    shape = [2, 3]
    dropout_outputs = ["Y"]
    if mask_use == "empty":
        dropout_outputs.append("")
    elif mask_use != "omitted":
        dropout_outputs.append("mask")
    nodes = [
        helper.make_node("Identity", ["X"], ["data"]),
        helper.make_node("Dropout", ["data"], dropout_outputs, ratio=ratio),
    ]
    outputs = []
    if mask_use != "mask_only":
        nodes.append(helper.make_node("Identity", ["Y"], ["Z"]))
        outputs.append(helper.make_tensor_value_info("Z", TensorProto.FLOAT, shape))
    if mask_use in ("exported", "mask_only"):
        outputs.append(helper.make_tensor_value_info("mask", TensorProto.BOOL, shape))
    elif mask_use == "consumed":
        nodes.append(helper.make_node("Identity", ["mask"], ["mask_result"]))
        outputs.append(helper.make_tensor_value_info("mask_result", TensorProto.BOOL, shape))
    model = helper.make_model(
        helper.make_graph(
            nodes,
            "dropout_mask",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, shape)],
            outputs,
        ),
        ir_version=7,
        opset_imports=[helper.make_opsetid("", opset)],
    )
    checker.check_model(model, full_check=True)
    optimized = onnxoptimizer.optimize(model, ["eliminate_nop_dropout"])
    checker.check_model(optimized, full_check=True)
    mask_is_used = mask_use in ("exported", "consumed", "mask_only")
    assert any(node.op_type == "Dropout" for node in optimized.graph.node) == (
        ratio != 0.0 or mask_is_used
    )
    inputs = {"X": np.array([[-2, 0, 3], [4, -5, 6]], dtype=np.float32)}
    expected = ReferenceEvaluator(model).run(None, inputs)
    actual = ReferenceEvaluator(optimized).run(None, inputs)
    assert len(actual) == len(expected)
    for result, original in zip(actual, expected):
        assert result.dtype == original.dtype
        assert result.shape == original.shape
        np.testing.assert_array_equal(result, original)
