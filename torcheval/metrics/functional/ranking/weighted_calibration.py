# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch


@torch.inference_mode()
def weighted_calibration(
    input: torch.Tensor,
    target: torch.Tensor,
    weight: float | int | torch.Tensor = 1.0,
    *,
    num_tasks: int = 1,
) -> torch.Tensor:
    """
    Compute weighted calibration metric. When weight is not provided, it calculates the unweighted calibration.
    Its class version is :obj:`torcheval.metrics.WeightedCalibration`.

    weighted_calibration = sum(input * weight) / sum(target * weight)

    Args:
        input (Tensor): Tensor of input values.
        target (Tensor): Tensor of binary targets
        weight(optional): Float or Int or Tensor of input weights. It is default to 1.0. If weight is a Tensor, its size should match the input tensor size.
        num_tasks (int): Number of tasks that need weighted_calibration calculation. Default value
                    is 1.

    Returns:
            Tensor: The return value of weighted calibration for each task (num_tasks,).

    Raises:
        ValueError: If value of weight is neither a ``float`` nor a ``int`` nor a ``torch.Tensor`` that matches the input tensor size.

    Examples::

        >>> import torch
        >>> from torcheval.metrics.functional import weighted_calibration
        >>> weighted_calibration(torch.tensor([0.8, 0.4, 0.3, 0.8, 0.7, 0.6]),torch.tensor([1, 1, 0, 0, 1, 0]))
        tensor([1.2])

        >>> weighted_calibration(torch.tensor([0.8, 0.4, 0.3, 0.8, 0.7, 0.6]),torch.tensor([1, 1, 0, 0, 1, 0]), torch.tensor([0.5, 1., 2., 0.4, 1.3, 0.9]))
        tensor([1.1321])

        >>> weighted_calibration(
                torch.tensor([[0.8, 0.4], [0.8, 0.7]]),
                torch.tensor([[1, 1], [0, 1]]),
                num_tasks=2,
            ),
        >>> tensor([0.6000, 1.5000])

    """
    return _weighted_calibration_compute(input, target, weight, num_tasks=num_tasks)


def _weighted_calibration_update(
    input: torch.Tensor,
    target: torch.Tensor,
    weight: float | int | torch.Tensor,
    *,
    num_tasks: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    _weighted_calibration_input_check(input, target, weight, num_tasks=num_tasks)
    if isinstance(weight, float) or isinstance(weight, int):
        weighted_input_sum = weight * torch.sum(input, dim=-1)
        weighted_target_sum = weight * torch.sum(target, dim=-1)
        return weighted_input_sum, weighted_target_sum
    elif isinstance(weight, torch.Tensor) and input.size() == weight.size():
        return torch.sum(weight * input, dim=-1), torch.sum(weight * target, dim=-1)
    else:
        raise ValueError(
            "Weight must be either a float value or a tensor that matches the input tensor size. "
            f"Got {weight} instead."
        )


def _weighted_calibration_compute(
    input: torch.Tensor,
    target: torch.Tensor,
    weight: float | int | torch.Tensor,
    *,
    num_tasks: int,
) -> torch.Tensor:
    weighted_input_sum, weighted_target_sum = _weighted_calibration_update(
        input, target, weight, num_tasks=num_tasks
    )
    return weighted_input_sum / weighted_target_sum


def _weighted_calibration_input_check(
    input: torch.Tensor,
    target: torch.Tensor,
    weight: float | int | torch.Tensor,
    num_tasks: int,
) -> None:
    if input.shape != target.shape:
        raise ValueError(
            f"`input` shape ({input.shape}) is different from `target` shape ({target.shape})"
        )
    if num_tasks == 1:
        if len(input.shape) > 1:
            raise ValueError(
                f"`num_tasks = 1`, `input` is expected to be one-dimensional tensor, but got shape ({input.shape})."
            )
    elif len(input.shape) == 1 or input.shape[0] != num_tasks:
        raise ValueError(
            f"`num_tasks = {num_tasks}`, `input`'s shape is expected to be ({num_tasks}, num_samples), but got shape ({input.shape})."
        )
