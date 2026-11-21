# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import subprocess
import sys


def test_metrics_import_without_torchscript_deprecation():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import torch, warnings\n"
            'warnings.filterwarnings("error", message=r"`torch\\.jit\\.script` is deprecated")\n'
            "import torcheval.metrics",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
