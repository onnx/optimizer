#!/usr/bin/env python

# SPDX-FileCopyrightText: ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pathlib
import subprocess
import sys


def main() -> int:
    root_folder = pathlib.Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "-m", "mypy", "."], cwd=root_folder, check=False)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
