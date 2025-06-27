<!--- SPDX-License-Identifier: Apache-2.0 -->

# ONNX Optimizer

[![PyPI version](https://img.shields.io/pypi/v/onnxoptimizer.svg)](https://pypi.python.org/pypi/onnxoptimizer/)
[![PyPI license](https://img.shields.io/pypi/l/onnxoptimizer.svg)](https://pypi.python.org/pypi/onnxoptimizer/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/onnx/optimizer/pulls)

## 🛠 Maintainer Wanted

We are currently **looking for a new maintainer** to help support and evolve the `onnxoptimizer` project.

If you're passionate about ONNX, graph optimizations, or contributing to the open source machine learning ecosystem, we'd love to hear from you! This is a great opportunity to contribute to a widely used project and collaborate with the ONNX community.

**To express interest:**  
Please open an issue or comment on [this thread](https://github.com/onnx/optimizer/issues) and let us know about your interest and background.

## Introduction

ONNX provides a C++ library for performing arbitrary optimizations on ONNX models, as well as a growing list of prepackaged optimization passes.

The primary motivation is to share work between the many ONNX backend implementations. Not all possible optimizations can be directly implemented on ONNX graphs - some will need additional backend-specific information - but many can, and our aim is to provide all such passes along with ONNX so that they can be re-used with a single function call.

You may be interested in invoking the provided passes, or in implementing new ones (or both).

## Installation

You can install onnxoptimizer from PyPI:

```bash
pip3 install onnxoptimizer
```

Note that you may need to upgrade your pip first if you have trouble:

```bash
pip3 install -U pip
```

If you want to build from source:

```bash
git clone --recursive https://github.com/onnx/optimizer onnxoptimizer
cd onnxoptimizer
pip3 install -e .
```

Note that you need to install protobuf before building from source.


## Command-line API
Now you can use command-line api in terminal instead of  python script.

```
python -m onnxoptimizer input_model.onnx output_model.onnx
```

Arguments list is following:
```
# python3 -m onnxoptimizer -h                                 
usage: python -m onnxoptimizer input_model.onnx output_model.onnx 

onnxoptimizer command-line api

optional arguments:
  -h, --help            show this help message and exit
  --print_all_passes    print all available passes
  --print_fuse_elimination_passes
                        print all fuse and elimination passes
  -p [PASSES ...], --passes [PASSES ...]
                        list of optimization passes name, if no set, fuse_and_elimination_passes will be used
  --fixed_point         fixed point
```
## Roadmap

* More built-in pass
* Separate graph rewriting and constant folding (or a pure graph rewriting mode, see [issue #9](https://github.com/onnx/optimizer/issues/9) for the details)

## Relevant tools

* [onnx-simplifier](https://github.com/daquexian/onnx-simplifier): A handy and popular tool based on onnxoptimizer

## Code of Conduct

[ONNX Open Source Code of Conduct](https://onnx.ai/codeofconduct.html)
