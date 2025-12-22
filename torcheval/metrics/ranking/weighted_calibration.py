# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# pyre-ignore-all-errors[16]: Undefined attribute of metric states.

from collections.abc import Iterable
from typing import TypeVar

import torch

from torcheval.metrics.functional.ranking.weighted_calibration import (
    _weighted_calibration_update,
)
from torcheval.metrics.metric import Metric
from torcheval.utils.device import largest_float

TWeightedCalibration = TypeVar("TWeightedCalibration")


class WeightedCalibration(Metric[torch.Tensor]):
    """
    Compute weighted calibration metric. When weight is not provided, it calculates the unweighted calibration.
    Its functional version is :func:`torcheval.metrics.functional.weighted_calibration`.

    weighted_calibration = sum(input * weight) / sum(target * weight)

    Args:
        num_tasks (int): Number of tasks that need WeightedCalibration calculations. Default value
                    is 1.
    Raises:
        ValueError: If value of weight is neither a ``float`` nor a ``int`` nor a ``torch.Tensor`` that matches the input tensor size.

    Examples::

        >>> import torch
        >>> from torcheval.metrics import WeightedCalibration
        >>> metric = WeightedCalibration()
        >>> metric.update(torch.tensor([0.8, 0.4, 0.3, 0.8, 0.7, 0.6]),torch.tensor([1, 1, 0, 0, 1, 0]))
        >>> metric.compute()
        tensor([1.2], dtype=torch.float64)

        >>> metric = WeightedCalibration()
        >>> metric.update(torch.tensor([0.8, 0.4, 0.3, 0.8, 0.7, 0.6]),torch.tensor([1, 1, 0, 0, 1, 0]), torch.tensor([0.5, 1., 2., 0.4, 1.3, 0.9]))
        >>> metric.compute()
        tensor([1.1321], dtype=torch.float64)

        >>> metric = WeightedCalibration(num_tasks=2)
        >>> metric.update(torch.tensor([[0.8, 0.4], [0.8, 0.7]]),torch.tensor([[1, 1], [0, 1]]),)
        >>> metric.compute()
        tensor([0.6000, 1.5000], dtype=torch.float64)


    """

    def __init__(
        self: TWeightedCalibration,
        *,
        num_tasks: int = 1,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(device=device)
        if num_tasks < 1:
            raise ValueError(
                "`num_tasks` value should be greater than and equal to 1, but received {num_tasks}. "
            )
        self.num_tasks = num_tasks
        dtype = largest_float(device)
        self._add_state(
            "weighted_input_sum",
            torch.zeros(self.num_tasks, dtype=dtype, device=self.device),
        )
        self._add_state(
            "weighted_target_sum",
            torch.zeros(self.num_tasks, dtype=dtype, device=self.device),
        )

    @torch.inference_mode()
    # pyre-ignore[14]: `update` overrides method defined in `Metric` inconsistently.
    def update(
        self: TWeightedCalibration,
        input: torch.Tensor,
        target: torch.Tensor,
        weight: float | int | torch.Tensor = 1.0,
    ) -> TWeightedCalibration:
        """
        Update the metric state with the total sum of weighted inputs and the total sum of weighted labels.

        Args:
            input (Tensor): Predicted unnormalized scores (often referred to as logits) or
                binary class probabilities (num_tasks, num_samples).
            target (Tensor): Ground truth binary class indices (num_tasks, num_samples).
            weight (Optional): Float or Int or Tensor of input weights. It is default to 1.0. If weight is a Tensor, its size should match the input tensor size.
        """

        weighted_input_sum, weighted_target_sum = _weighted_calibration_update(
            input, target, weight, num_tasks=self.num_tasks
        )
        self.weighted_input_sum += weighted_input_sum
        self.weighted_target_sum += weighted_target_sum
        return self

    @torch.inference_mode()
    def compute(self: TWeightedCalibration) -> torch.Tensor:
        """
        Return the weighted calibration.  If no ``update()`` calls are made before
        ``compute()`` is called, return an empty tensor.

        Returns:
            Tensor: The return value of weighted calibration for each task (num_tasks,).
        """
        if torch.any(self.weighted_target_sum == 0.0):
            return torch.empty(0, device=self.device)

        weighted_calibration = self.weighted_input_sum / self.weighted_target_sum
        return weighted_calibration

    @torch.inference_mode()
    def merge_state(
        self: TWeightedCalibration, metrics: Iterable[TWeightedCalibration]
    ) -> TWeightedCalibration:
        """
        Merge the metric state with its counterparts from other metric instances.

        Args:
            metrics (Iterable[Metric]): metric instances whose states are to be merged.
        """
        for metric in metrics:
            self.weighted_input_sum += metric.weighted_input_sum.to(self.device)
            self.weighted_target_sum += metric.weighted_target_sum.to(self.device)
        return self
