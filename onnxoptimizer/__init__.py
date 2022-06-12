# SPDX-License-Identifier: Apache-2.0

# ATTENTION: The code in this file is highly EXPERIMENTAL.
# Adventurous users should note that the APIs will probably change.

"""onnx optimizer

This enables users to optimize their models.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import onnx
import onnxoptimizer.onnx_opt_cpp2py_export as C
from onnx import ModelProto
from typing import Text, Sequence, Optional

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

    model_str = model.SerializeToString()
    if fixed_point:
        optimized_model_str = C.optimize_fixedpoint(model_str, passes)
    else:
        optimized_model_str = C.optimize(model_str, passes)

    return onnx.load_from_string(optimized_model_str)


__all__ = ['optimize', 'get_available_passes', 'get_fuse_and_elimination_passes']
