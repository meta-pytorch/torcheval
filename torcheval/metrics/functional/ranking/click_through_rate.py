# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import torch


@torch.inference_mode()
def click_through_rate(
    input: torch.Tensor,
    weights: torch.Tensor | None = None,
    *,
    num_tasks: int = 1,
) -> torch.Tensor:
    """
    Compute the click through rate given a click events.
    Its class version is ``torcheval.metrics.ClickThroughRate``.

    Args:
        input (Tensor): Series of values representing user click (1) or skip (0)
                        of shape (num_events) or (num_objectives, num_events).
        weights (Tensor, Optional): Weights for each event, tensor with the same shape as input.
        num_tasks (int): Number of tasks that need weighted_calibration calculation. Default value
                    is 1.

    Examples::

        >>> import torch
        >>> from torcheval.metrics.functional import click_through_rate
        >>> input = torch.tensor([0, 1, 0, 1, 1, 0, 0, 1])
        >>> click_through_rate(input)
        tensor(0.5)
        >>> weights = torch.tensor([1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0])
        >>> click_through_rate(input, weights)
        tensor(0.58333)
        >>> input = torch.tensor([[0, 1, 0, 1], [1, 0, 0, 1]])
        >>> weights = torch.tensor([[1.0, 2.0, 1.0, 2.0],[1.0, 2.0, 1.0, 1.0]])
        >>> click_through_rate(input, weights, num_tasks=2)
        tensor([0.6667, 0.4])
    """
    if weights is None:
        # pyrefly: ignore [bad-assignment]
        weights = 1.0
    click_total, weight_total = _click_through_rate_update(
        # pyrefly: ignore [bad-argument-type]
        input,
        # pyrefly: ignore [bad-argument-type]
        weights,
        num_tasks=num_tasks,
    )
    return _click_through_rate_compute(click_total, weight_total)


def _click_through_rate_update(
    input: torch.Tensor,
    weights: torch.Tensor | float | int = 1.0,
    *,
    num_tasks: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    _click_through_rate_input_check(input, weights, num_tasks=num_tasks)
    if isinstance(weights, torch.Tensor):
        weights = weights.type(torch.float)
        click_total = (input * weights).sum(-1)
        weight_total = weights.sum(-1)
    else:
        click_total = weights * input.sum(-1).type(torch.float)
        weight_total = weights * input.size(-1) * torch.ones_like(click_total)

    return click_total, weight_total


def _click_through_rate_compute(
    click_total: torch.Tensor,
    weight_total: torch.Tensor,
) -> torch.Tensor:
    # epsilon is a performant solution to divide by zero errors when weight_total = 0.0
    # Since click_total = input*weights, weights = 0.0 implies 0.0/(0.0 + eps) = 0.0
    eps = torch.finfo(weight_total.dtype).tiny
    return click_total / (weight_total + eps)


def _click_through_rate_input_check(
    input: torch.Tensor,
    weights: torch.Tensor | float | int,
    *,
    num_tasks: int,
) -> None:
    if input.ndim != 1 and input.ndim != 2:
        raise ValueError(
            f"`input` should be a one or two dimensional tensor, got shape {input.shape}."
        )
    if isinstance(weights, torch.Tensor) and weights.shape != input.shape:
        raise ValueError(
            f"tensor `weights` should have the same shape as tensor `input`, got shapes {weights.shape} and {input.shape}, respectively."
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
