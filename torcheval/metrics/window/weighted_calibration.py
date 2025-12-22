# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# pyre-ignore-all-errors[16]: Undefined attribute of metric states.

from collections.abc import Iterable
from typing import TypeVar, Union

import torch

from torcheval.metrics.functional.ranking.weighted_calibration import (
    _weighted_calibration_update,
)
from torcheval.metrics.metric import Metric
from torcheval.utils.device import largest_float

TWindowedWeightedCalibration = TypeVar("TWindowedWeightedCalibration")


class WindowedWeightedCalibration(
    Metric[Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]]
):
    """
    Compute weighted calibration metric. When weight is not provided, it calculates the unweighted calibration.
    Its functional version is :func:`torcheval.metrics.functional.weighted_calibration`.

    weighted_calibration = sum(input * weight) / sum(target * weight)

    Args:
        num_tasks (int): Number of tasks that need WeightedCalibration calculations. Default value
                    is 1.
        max_num_updates (int): The max window size that can accommodate the number of updates.
        enable_lifetime (bool): A boolean indicator whether to calculate lifetime values.
    Raises:
        ValueError: If value of weight is neither a ``float`` nor a ``int`` nor a ``torch.Tensor`` that matches the input tensor size.

    Examples::

        >>> import torch
        >>> from torcheval.metrics import WindowedWeightedCalibration
        >>> metric = WindowedWeightedCalibration(max_num_updates=2, enable_lifetime=False)
        >>> metric.update(torch.tensor([0.8, 0.4]),torch.tensor([1, 1]))
        >>> metric.update(torch.tensor([0.3, 0.8]),torch.tensor([0, 0]))
        >>> metric.update(torch.tensor([0.7, 0.6]),torch.tensor([1, 0]))
        >>> metric.compute()
        tensor([2.4], dtype=torch.float64)

        >>> metric = WindowedWeightedCalibration(max_num_updates=2, enable_lifetime=True)
        >>> metric.update(torch.tensor([0.8, 0.4]),torch.tensor([1, 1]))
        >>> metric.update(torch.tensor([0.3, 0.8]),torch.tensor([0, 0]))
        >>> metric.update(torch.tensor([0.7, 0.6]),torch.tensor([1, 0]))
        >>> metric.compute()
        (
        tensor([1.2], dtype=torch.float64)
        tensor([2.4], dtype=torch.float64)
        )

        >>> metric = WindowedWeightedCalibration(num_tasks=2)
        >>> metric.update(torch.tensor([[0.8, 0.4], [0.8, 0.7]]),torch.tensor([[1, 1], [0, 1]]),)
        >>> metric.compute()
        tensor([0.6000, 1.5000], dtype=torch.float64)


    """

    def __init__(
        self: TWindowedWeightedCalibration,
        *,
        num_tasks: int = 1,
        max_num_updates: int = 100,
        enable_lifetime: bool = True,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(device=device)
        if num_tasks < 1:
            raise ValueError(
                "`num_tasks` value should be greater than and equal to 1, but received {num_tasks}. "
            )
        self.num_tasks = num_tasks
        self._add_state("max_num_updates", max_num_updates)
        self.enable_lifetime = enable_lifetime
        self.next_inserted = 0
        self._add_state("total_updates", 0)
        dtype = largest_float(device)
        if self.enable_lifetime:
            self._add_state(
                "weighted_input_sum",
                torch.zeros(self.num_tasks, dtype=dtype, device=self.device),
            )
            self._add_state(
                "weighted_target_sum",
                torch.zeros(self.num_tasks, dtype=dtype, device=self.device),
            )
        self._add_state(
            "windowed_weighted_input_sum",
            torch.zeros(
                self.num_tasks,
                self.max_num_updates,
                dtype=dtype,
                device=self.device,
            ),
        )
        self._add_state(
            "windowed_weighted_target_sum",
            torch.zeros(
                self.num_tasks,
                self.max_num_updates,
                dtype=dtype,
                device=self.device,
            ),
        )

    @torch.inference_mode()
    # pyre-ignore[14]: `update` overrides method defined in `Metric` inconsistently.
    def update(
        self: TWindowedWeightedCalibration,
        input: torch.Tensor,
        target: torch.Tensor,
        weight: float | int | torch.Tensor = 1.0,
    ) -> TWindowedWeightedCalibration:
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
        if self.enable_lifetime:
            self.weighted_input_sum += weighted_input_sum
            self.weighted_target_sum += weighted_target_sum
        self.windowed_weighted_input_sum[:, self.next_inserted] = weighted_input_sum
        self.windowed_weighted_target_sum[:, self.next_inserted] = weighted_target_sum
        self.next_inserted += 1
        self.next_inserted %= self.max_num_updates
        self.total_updates += 1
        return self

    @torch.inference_mode()
    def compute(
        self: TWindowedWeightedCalibration,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Return the weighted calibration.  If no ``update()`` calls are made before
        ``compute()`` is called, return an empty tensor.

        Returns:
            Tensor: The return value of weighted calibration for each task (num_tasks,).
        """
        if self.total_updates == 0:
            naught = torch.empty(0, device=self.device)
            if self.enable_lifetime:
                return naught, naught
            else:
                return naught
        # for the case the winodw has been filled more than once
        dtype = largest_float(self.device)
        if self.total_updates >= self.max_num_updates:
            windowed_weighted_calibration = self.windowed_weighted_input_sum.sum(
                dim=-1
            ) / torch.clamp(
                self.windowed_weighted_target_sum.sum(dim=-1),
                min=torch.finfo(dtype).eps,
            )
        else:
            # for the situation when window array hasn't been filled up
            windowed_weighted_calibration = self.windowed_weighted_input_sum[
                :, : self.next_inserted
            ].sum(dim=-1) / torch.clamp(
                self.windowed_weighted_target_sum[:, : self.next_inserted].sum(dim=-1),
                min=torch.finfo(dtype).eps,
            )
        if self.enable_lifetime:
            self.weighted_target_sum = torch.clamp(
                self.weighted_target_sum, min=torch.finfo(dtype).eps
            )
            weighted_calibration = self.weighted_input_sum / self.weighted_target_sum
            return weighted_calibration, windowed_weighted_calibration
        return windowed_weighted_calibration

    @torch.inference_mode()
    def merge_state(
        self: TWindowedWeightedCalibration,
        metrics: Iterable[TWindowedWeightedCalibration],
    ) -> TWindowedWeightedCalibration:
        """
        Merge the metric state with its counterparts from other metric instances.

        Args:
            metrics (Iterable[Metric]): metric instances whose states are to be merged.
        """
        merge_max_num_updates = self.max_num_updates
        for metric in metrics:
            merge_max_num_updates += metric.max_num_updates
        cur_windowed_weighted_input_sum = self.windowed_weighted_input_sum
        cur_windowed_weighted_target_sum = self.windowed_weighted_target_sum
        idx = min(self.total_updates, self.max_num_updates)
        dtype = largest_float(self.device)
        self.windowed_weighted_input_sum = torch.zeros(
            self.num_tasks,
            merge_max_num_updates,
            dtype=dtype,
            device=self.device,
        )
        self.windowed_weighted_target_sum = torch.zeros(
            self.num_tasks,
            merge_max_num_updates,
            dtype=dtype,
            device=self.device,
        )
        self.windowed_weighted_input_sum[:, :idx] = cur_windowed_weighted_input_sum[
            :, :idx
        ]
        self.windowed_weighted_target_sum[:, :idx] = cur_windowed_weighted_target_sum[
            :, :idx
        ]

        for metric in metrics:
            if self.enable_lifetime:
                self.weighted_input_sum += metric.weighted_input_sum.to(self.device)
                self.weighted_target_sum += metric.weighted_target_sum.to(self.device)
            cur_size = min(metric.total_updates, metric.max_num_updates)
            self.windowed_weighted_input_sum[:, idx : idx + cur_size] = (
                metric.windowed_weighted_input_sum[:, :cur_size]
            )
            self.windowed_weighted_target_sum[:, idx : idx + cur_size] = (
                metric.windowed_weighted_target_sum[:, :cur_size]
            )
            idx += cur_size
            self.total_updates += metric.total_updates
        self.next_inserted = idx
        self.next_inserted %= self.max_num_updates
        return self
