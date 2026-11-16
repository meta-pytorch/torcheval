# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch


def _riemann_integral(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Riemann integral approximates the area of each cell with a rectangle positioned at the egde.
    It is conventionally used rather than trapezoid approximation, which uses a rectangle positioned in the
    center"""
    return -torch.sum((x[1:] - x[:-1]) * y[:-1])


def _create_threshold_tensor(
    threshold: int | list[float] | torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Creates a threshold tensor from an integer, a list or a tensor.
    If `threshold` is an integer n, returns a Tensor with values [0, 1/(n-1), 2/(n-1), ..., (n-2)/(n-1), 1].
    If `threshold` is a list, returns the list converted to a Tensor.
    Otherwise, returns the tensor itself.
    """
    if isinstance(threshold, int):
        threshold = torch.linspace(0, 1.0, threshold, device=device)
    elif isinstance(threshold, list):
        threshold = torch.tensor(threshold, device=device)
    return threshold
