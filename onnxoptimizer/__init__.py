# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

# ATTENTION: The code in this file is highly EXPERIMENTAL.
# Adventurous users should note that the APIs will probably change.

"""ONNX optimizer."""

from __future__ import annotations

__all__ = [
    "optimize",
    "get_available_passes",
    "get_fuse_and_elimination_passes",
    "main",
]


import os
import tempfile

import onnx

import onnxoptimizer.onnx_opt_cpp2py_export as _c
from onnxoptimizer.onnx_opt_cpp2py_export import (
    get_available_passes,
    get_fuse_and_elimination_passes,
)
from onnxoptimizer.onnxoptimizer_main import main

from .version import version as __version__  # noqa


def optimize(
    model: onnx.ModelProto,
    passes: list[str] | None = None,
    fixed_point: bool = False,
    return_report: bool = False,
) -> onnx.ModelProto | tuple[onnx.ModelProto, dict[str, int]]:
    """Apply the optimization on the serialized ModelProto.

    Arguments:
        model: ONNX model.
        passes: Optimization names.
        fixed_point: Whether to repeatedly apply the passes until the graph
            reaches a fixed point.
        return_report: When ``True``, also return a report describing how many
            times each pass modified the graph.

    Return:
        The optimized model. When ``return_report`` is ``True``, a tuple of
        ``(optimized_model, report)`` is returned instead, where ``report`` is a
        ``dict`` mapping each pass name to the total number of positive
        (successful) transforms it applied. A pass that ran without changing the
        graph maps to ``0``; passes absent from the map did not report a count.
    """

    if passes is None:
        passes = get_fuse_and_elimination_passes()
    if not isinstance(model, onnx.ModelProto):
        raise TypeError(f"Optimizer only accepts ModelProto, incorrect type: {type(model)}")
    try:
        model_str = model.SerializeToString()
        if return_report:
            if fixed_point:
                optimized_model_str, report = _c.optimize_fixedpoint_report(
                    model_str, passes
                )
            else:
                optimized_model_str, report = _c.optimize_report(model_str, passes)
            return onnx.load_from_string(optimized_model_str), report
        if fixed_point:
            optimized_model_str = _c.optimize_fixedpoint(model_str, passes)
        else:
            optimized_model_str = _c.optimize(model_str, passes)

        return onnx.load_from_string(optimized_model_str)
    except ValueError:
        with (
            tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as file_src,
            tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as file_dest,
            tempfile.NamedTemporaryFile(delete=False) as data_file_src,
            tempfile.NamedTemporaryFile(delete=False) as data_file_dest,
        ):
            data_src_rel_filename = os.path.relpath(
                data_file_src.name, os.path.dirname(file_src.name)
            )
            data_dest_rel_filename = os.path.relpath(
                data_file_dest.name, os.path.dirname(file_dest.name)
            )
            try:
                onnx.save(
                    model,
                    file_src.name,
                    save_as_external_data=True,
                    location=data_src_rel_filename,
                    convert_attribute=True,
                )
                report = None
                if return_report:
                    if fixed_point:
                        report = _c.optimize_fixedpoint_from_path_report(
                            file_src.name, file_dest.name, passes, data_dest_rel_filename
                        )
                    else:
                        report = _c.optimize_from_path_report(
                            file_src.name, file_dest.name, passes, data_dest_rel_filename
                        )
                elif fixed_point:
                    _c.optimize_fixedpoint_from_path(
                        file_src.name, file_dest.name, passes, data_dest_rel_filename
                    )
                else:
                    _c.optimize_from_path(
                        file_src.name, file_dest.name, passes, data_dest_rel_filename
                    )
                optimized_model = onnx.load(file_dest.name, load_external_data=True)
                if return_report:
                    return optimized_model, report
                return optimized_model
            finally:
                os.remove(file_src.name)
                os.remove(file_dest.name)
                os.remove(data_file_src.name)
                os.remove(data_file_dest.name)
