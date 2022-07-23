# SPDX-License-Identifier: Apache-2.0

# ATTENTION: The code in this file is highly EXPERIMENTAL.
# Adventurous users should note that the APIs will probably change.

"""onnx optimizer

This enables users to optimize their models.
"""

import onnx
import onnxoptimizer.onnx_opt_cpp2py_export as C
from .version import version as __version__  # noqa
from onnx import ModelProto
from typing import Text, Sequence, Optional

import tempfile
import os

get_available_passes = C.get_available_passes

get_fuse_and_elimination_passes = C.get_fuse_and_elimination_passes


def optimize(model, passes=None, fixed_point=False):  # type: (ModelProto, Optional[Sequence[Text]], bool) -> ModelProto
    """Apply the optimization on the serialized ModelProto.

    Arguments:
        model (ModelProto): model
        passes (list of string): list of optimization names

    Return:
        return (ModelProto) optimized model
    """

    if passes is None:
        print('WARNING: defualt optimization passes will be enlarged to all fuse and elimination passes in the next version')
        passes = ['eliminate_nop_transpose',
                  'eliminate_nop_pad',
                  'fuse_consecutive_transposes',
                  'fuse_transpose_into_gemm']
    if not isinstance(model, ModelProto):
        raise ValueError(
            'Optimizer only accepts ModelProto, incorrect type: {}'.format(type(model)))
    try:
        model_str = model.SerializeToString()
        if fixed_point:
            optimized_model_str = C.optimize_fixedpoint(model_str, passes)
        else:
            optimized_model_str = C.optimize(model_str, passes)

        return onnx.load_from_string(optimized_model_str)
    except ValueError:
        file_src = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
        file_dest = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
        data_file_src = tempfile.NamedTemporaryFile(delete=False)
        data_file_dest = tempfile.NamedTemporaryFile(delete=False)
        data_src_rel_filename = os.path.relpath(data_file_src.name, os.path.dirname(file_src.name))
        data_dest_rel_filename = os.path.relpath(data_file_dest.name, os.path.dirname(file_dest.name))
        try:
            onnx.save(model, file_src.name, save_as_external_data=True, location=data_src_rel_filename, convert_attribute=True,)
            if fixed_point:
                C.optimize_fixedpoint_from_path(file_src.name, file_dest.name, passes, data_dest_rel_filename)
            else:
                C.optimize_from_path(file_src.name, file_dest.name, passes, data_dest_rel_filename)
            return onnx.load(file_dest, load_external_data=True)
        finally:
            os.remove(file_src.name)
            os.remove(file_dest.name)
            os.remove(data_file_src.name)
            os.remove(data_file_dest.name)


__all__ = ['optimize', 'get_available_passes', 'get_fuse_and_elimination_passes']
