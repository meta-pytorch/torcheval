# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from typing import TypeVar

import torch
from typing_extensions import ParamSpec

_Params = ParamSpec("_Params")
_Result = TypeVar("_Result")


@lru_cache(maxsize=None)
def _script_on_use(function: Callable[_Params, _Result]) -> Callable[_Params, _Result]:
    """Cache a metric helper's compilation at its eager call site."""
    return torch.jit.script(function)
