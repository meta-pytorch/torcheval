# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import functools
from typing import Union

import torch


@functools.lru_cache()
def largest_float(device: Union[torch.device, str, None]) -> torch.dtype:
    """Determines whether the largest representable floating-point type on
    a given device is 64-bit or 32-bit.

    Args:
        device (Union[torch.device, str, None])

    Returns:
        torch.dtype: either torch.float64 or torch.float32"""
    try:
        torch.zeros(1, dtype=torch.float64, device=device)
        return torch.float64
    except TypeError:
        return torch.float32
