# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import onnx
import onnx.backend.test.loader

import onnxoptimizer


def test_seq_type():
    orig = onnx.load(
        onnx.backend.test.loader.DATA_DIR + "/node/test_sequence_insert_at_back/model.onnx"
    )
    optimized = onnxoptimizer.optimize(orig)
    assert optimized.graph.input[0].type == orig.graph.input[0].type
