<!--
SPDX-FileCopyrightText: ONNX Project Contributors

SPDX-License-Identifier: Apache-2.0
-->

# ONNX Optimizer

[![PyPI version](https://img.shields.io/pypi/v/onnxoptimizer.svg)](https://pypi.org/project/onnxoptimizer/)
[![PyPI license](https://img.shields.io/pypi/l/onnxoptimizer.svg)](https://pypi.org/project/onnxoptimizer/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/onnx/optimizer/pulls)

ONNX Optimizer provides reusable, backend-independent optimization passes for
ONNX models. Its C++ core works on the ONNX intermediate representation (IR)
and is exposed through Python, C++, and C APIs.

The optimizer applies an ordered list of caller-selected passes. It is intended
for transformations that can be expressed using ONNX graph information alone;
optimizations that depend on a particular runtime or hardware target belong in
the corresponding backend.

The optimizer APIs are currently experimental and may change.

## Installation

ONNX Optimizer requires Python 3.10 or newer. Install the released package from
PyPI:

```bash
python -m pip install onnxoptimizer
```

## Python API

Load a model, select passes, and optimize it:

```python
import onnx
import onnxoptimizer

model = onnx.load("input.onnx")
passes = ["eliminate_identity", "eliminate_deadend"]
optimized = onnxoptimizer.optimize(model, passes)

onnx.checker.check_model(optimized)
onnx.save(optimized, "optimized.onnx")
```

If `passes` is omitted, `optimize()` uses the fuse and elimination passes
returned by `get_fuse_and_elimination_passes()`. Passes run in the supplied
order. Set `fixed_point=True` to repeat partial passes while they continue to
change the graph.

Inspect the available pass names with:

```python
import onnxoptimizer

print(onnxoptimizer.get_available_passes())
print(onnxoptimizer.get_fuse_and_elimination_passes())
```

## Command-line interface

The installed package provides the `onnxoptimizer` command. The module form is
equivalent:

```bash
onnxoptimizer input.onnx optimized.onnx
python -m onnxoptimizer input.onnx optimized.onnx
```

By default, the command runs the fuse and elimination passes and performs ONNX
shape inference after optimization. Common options include:

```text
--print_all_passes
--print_fuse_elimination_passes
-p [PASS ...], --passes [PASS ...]
--fixed_point
--skip_infer_shapes
```

Run `onnxoptimizer --help` for the complete command-line help.

## Build from source

Clone the repository with its pinned ONNX submodule, then install it in editable
mode:

```bash
git clone --recursive https://github.com/onnx/optimizer.git onnxoptimizer
cd onnxoptimizer
python -m pip install -e .
```

The build requires a C++17 compiler. The Python build configuration installs
its CMake and protobuf build dependencies automatically.

## Development

Install the test and lint tools, then run the Python checks:

```bash
python -m pip install pytest pytest-xdist ruff mypy
python -m pytest onnxoptimizer/test
ruff check .
python setup.py typecheck
```

To build and test the standalone C++ targets:

```bash
cmake -DONNX_BUILD_TESTS=ON -B build -S .
cmake --build build
ctest --test-dir build --output-on-failure
```

The main implementation areas are:

- `onnxoptimizer/passes/`: optimization passes and pass utilities;
- `onnxoptimizer/pass_registry.*`: built-in pass registration;
- `onnxoptimizer/pass_manager.*`: ordered and fixed-point scheduling;
- `onnxoptimizer/optimize.*`: public C++ optimization entry points;
- `onnxoptimizer/c_api/`: C interface;
- `onnxoptimizer/test/` and `tests/`: Python and C++ tests.

Feature proposals and bugs are tracked in the
[GitHub issue tracker](https://github.com/onnx/optimizer/issues). Contributions
should preserve ONNX model semantics, public graph interfaces, and
compatibility across Linux, macOS, and Windows.

## Related projects

- [ONNX Simplifier](https://github.com/onnxsim/onnxsim) combines graph
  optimization with shape inference, constant folding, and correctness checks.

## Code of Conduct

Participation in this project is governed by the
[ONNX Open Source Code of Conduct](https://onnx.ai/codeofconduct.html).
