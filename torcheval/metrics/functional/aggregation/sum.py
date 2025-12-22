# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import torch


@torch.inference_mode()
def sum(
    input: torch.Tensor,
    weight: float | torch.Tensor = 1.0,
) -> torch.Tensor:
    """
    Compute weighted sum. When weight is not provided, it calculates the unweighted sum.
    Its class version is :obj:`torcheval.metrics.Sum`.

    Args:
        input (Tensor): Tensor of input values.
        weight(optional): Float or Int or Tensor of input weights. It is default to 1.0. If weight is a Tensor, its size should match the input tensor size.
    Raises:
        ValueError: If value of weight is neither a ``float`` nor an ``int`` nor a ``torch.Tensor`` that matches the input tensor size.

    Examples::

        >>> import torch
        >>> from torcheval.metrics.functional import sum
        >>> sum(torch.tensor([2, 3]))
        tensor(5.)
        >>> sum(torch.tensor([2, 3]), torch.tensor([0.1, 0.6]))
        tensor(2.)
        >>> sum(torch.tensor([2, 3]), 0.5)
        tensor(2.5)
        >>> sum(torch.tensor([2, 3]), 2)
        tensor(10.)
    """
    return _sum_update(input, weight)


def _sum_update(
    input: torch.Tensor, weight: float | int | torch.Tensor
) -> torch.Tensor:
    if (
        isinstance(weight, float)
        or isinstance(weight, int)
        or (isinstance(weight, torch.Tensor) and input.size() == weight.size())
    ):
        return (input * weight).sum()
    else:
        raise ValueError(
            "Weight must be either a float value or an int value or a tensor that matches the input tensor size. "
            f"Got {weight} instead."
        )
