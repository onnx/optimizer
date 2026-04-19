import io
import onnx
import onnxoptimizer
import pytest
import unittest

try:
    import torch
    import torchvision as tv

    has_tv = True
except:
    has_tv = False


@pytest.mark.skipif(not has_tv, reason="This test needs torchvision")
def test_function_preserved():
    with io.BytesIO() as f:
        module = tv.models.resnet18()
        torch.onnx.export(
            module,
            (torch.ones([1, 3, 224, 224], dtype=torch.float32), ),
            f,
            opset_version=15,
            export_modules_as_functions={
                torch.nn.BatchNorm2d,
                torch.nn.Conv2d,
            }
        )

        model = onnx.load_model_from_string(f.getvalue())
        opt_model = onnxoptimizer.optimize(model)
        assert len(model.functions) > 0
        assert len(model.functions) == len(opt_model.functions)
