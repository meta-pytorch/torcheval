# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import torch


@torch.inference_mode()
def mean(
    input: torch.Tensor,
    weight: float | int | torch.Tensor = 1.0,
) -> torch.Tensor:
    """
    Compute weighted mean. When weight is not provided, it calculates the unweighted mean.
    Its class version is :obj:`torcheval.metrics.Mean`.

    weighted_mean = sum(weight * input) / sum(weight)

    Args:
        input (Tensor): Tensor of input values.
        weight(optional): Float or Int or Tensor of input weights. It is default to 1.0. If weight is a Tensor, its size should match the input tensor size.
    Raises:
        ValueError: If value of weight is neither a ``float`` nor a ``int`` nor a ``torch.Tensor`` that matches the input tensor size.

    Examples::

        >>> import torch
        >>> from torcheval.metrics.functional import mean
        >>> mean(torch.tensor([2, 3]))
        tensor(2.5)
        >>> mean(torch.tensor([2, 3]), torch.tensor([0.2, 0.8]))
        tensor(2.8)
        >>> mean(torch.tensor([2, 3]), 0.5)
        tensor(2.5)
        >>> mean(torch.tensor([2, 3]), 1)
        tensor(2.5)
    """
    return _mean_compute(input, weight)


def _mean_update(
    input: torch.Tensor, weight: float | int | torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(weight, float) or isinstance(weight, int):
        weighted_sum = weight * torch.sum(input)
        weights = torch.tensor(float(weight) * torch.numel(input))
        return weighted_sum, weights
    elif isinstance(weight, torch.Tensor) and input.size() == weight.size():
        return torch.sum(weight * input), torch.sum(weight)
    else:
        raise ValueError(
            "Weight must be either a float value or a tensor that matches the input tensor size. "
            f"Got {weight} instead."
        )


def _mean_compute(
    input: torch.Tensor, weight: float | int | torch.Tensor
) -> torch.Tensor:
    weighted_sum, weights = _mean_update(input, weight)
    return weighted_sum / weights
