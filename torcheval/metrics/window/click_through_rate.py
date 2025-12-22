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

from torcheval.metrics.functional.ranking.click_through_rate import (
    _click_through_rate_compute,
    _click_through_rate_update,
)
from torcheval.metrics.metric import Metric
from torcheval.utils.device import largest_float


TWindowedClickThroughRate = TypeVar("TWindowedClickThroughRate")


class WindowedClickThroughRate(
    Metric[Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]]
):
    """
    The windowed version of ClickThroughRate that provides both windowed and lifetime values.
    Windowed value is calculated from the input and target of the last window_size number of `update()` calls.
    Lifetime value is calculated from all past input and target of `update()` calls.

    Compute the click through rate given click events.
    Its functional version is :func:`torcheval.metrics.functional.click_through_rate`.

    Args:
        num_tasks (int): Number of tasks that need click through rate calculation. Default value
                    is 1.
        max_num_updates (int): The max window size that can accommodate the number of updates.
        enable_lifetime (bool): A boolean indicator whether to calculate lifetime values.


    Examples::

        >>> import torch
        >>> from torcheval.metrics import WindowedClickThroughRate
        >>> metric = WindowedClickThroughRate(max_num_updates=2)
        >>> metric.update(torch.tensor([0, 1, 0, 1, 1, 0, 0, 1]))
        >>> metric.update(torch.tensor([0, 1, 0, 1, 1, 1, 1, 1]))
        >>> metric.update(torch.tensor([0, 1, 0, 1, 0, 0, 0, 1]))
        >>> metric.compute()
        (tensor([0.5417], dtype=torch.float64), tensor([0.5625], dtype=torch.float64))

        >>> metric = WindowedClickThroughRate(max_num_updates=2, enable_lifetime=False)
        >>> metric.update(torch.tensor([0, 1, 0, 1, 1, 0, 0, 1]))
        >>> metric.update(torch.tensor([0, 1, 0, 1, 1, 1, 1, 1]))
        >>> metric.update(torch.tensor([0, 1, 0, 1, 0, 0, 0, 1]))
        >>> metric.compute()
        tensor([0.5625], dtype=torch.float64)

    """

    def __init__(
        self: TWindowedClickThroughRate,
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
        if max_num_updates < 1:
            raise ValueError(
                "`max_num_updates` value should be greater than and equal to 1, but received {max_num_updates}. "
            )
        self.num_tasks = num_tasks
        self._add_state("max_num_updates", max_num_updates)
        self.next_inserted = 0
        self.enable_lifetime = enable_lifetime
        self._add_state("total_updates", 0)
        dtype = largest_float(device)
        if self.enable_lifetime:
            self._add_state(
                "click_total",
                torch.zeros(self.num_tasks, dtype=dtype, device=self.device),
            )
            self._add_state(
                "weight_total",
                torch.zeros(self.num_tasks, dtype=dtype, device=self.device),
            )
        self._add_state(
            "windowed_click_total",
            torch.zeros(
                self.num_tasks,
                self.max_num_updates,
                dtype=dtype,
                device=self.device,
            ),
        )
        self._add_state(
            "windowed_weight_total",
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
        self: TWindowedClickThroughRate,
        input: torch.Tensor,
        weights: torch.Tensor | float | int = 1.0,
    ) -> TWindowedClickThroughRate:
        """
        Update the metric state with new inputs.

        Args:
            input (Tensor): Series of values representing user click (1) or skip (0)
                            of shape (num_events) or (num_objectives, num_events).
            weights (Tensor, float, int): Weights for each event, single weight or tensor with the same shape as input.
        """
        click_total, weight_total = _click_through_rate_update(
            input, weights, num_tasks=self.num_tasks
        )
        if self.enable_lifetime:
            self.click_total += click_total
            self.weight_total += weight_total
        self.windowed_click_total[:, self.next_inserted] = click_total
        self.windowed_weight_total[:, self.next_inserted] = weight_total
        self.next_inserted += 1
        self.next_inserted %= self.max_num_updates
        self.total_updates += 1
        return self

    @torch.inference_mode()
    def compute(
        self: TWindowedClickThroughRate,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Return the stacked click through rank scores. If no ``update()`` calls are made before
        ``compute()`` is called, return tensor(0.0).
        """
        if self.total_updates == 0:
            naught = torch.empty(0, device=self.device)
            if self.enable_lifetime:
                return naught, naught
            else:
                return naught

        # For the case that window has been filled more than once
        if self.total_updates >= self.max_num_updates:
            click_total = self.windowed_click_total.sum(dim=-1)
            weight_total = self.windowed_weight_total.sum(dim=-1)
        else:  # For the case that window hasn't been filled
            click_total = self.windowed_click_total[:, : self.next_inserted].sum(dim=-1)
            weight_total = self.windowed_weight_total[:, : self.next_inserted].sum(
                dim=-1
            )
        windowed_click_through_rate = _click_through_rate_compute(
            click_total, weight_total
        )
        if self.enable_lifetime:
            lifetime_click_through_rate = _click_through_rate_compute(
                self.click_total, self.weight_total
            )
            return (
                lifetime_click_through_rate,
                windowed_click_through_rate,
            )
        else:
            return windowed_click_through_rate

    @torch.inference_mode()
    def merge_state(
        self: TWindowedClickThroughRate, metrics: Iterable[TWindowedClickThroughRate]
    ) -> TWindowedClickThroughRate:
        """
        Merge the metric state with its counterparts from other metric instances.
        First create tensors of size equal to the sum of all metrics' window sizes.
        Then, put all tensors to the front and leave the remaining indices zeros.

        Args:
            metrics (Iterable[Metric]): metric instances whose states are to be merged.
        """
        merge_max_num_updates = self.max_num_updates
        for metric in metrics:
            merge_max_num_updates += metric.max_num_updates
        cur_click_total = self.windowed_click_total
        cur_weight_total = self.windowed_weight_total

        dtype = largest_float(self.device)
        self.windowed_click_total = torch.zeros(
            self.num_tasks,
            merge_max_num_updates,
            dtype=dtype,
            device=self.device,
        )
        self.windowed_weight_total = torch.zeros(
            self.num_tasks,
            merge_max_num_updates,
            dtype=dtype,
            device=self.device,
        )

        cur_size = min(self.total_updates, self.max_num_updates)
        self.windowed_click_total[:, :cur_size] = cur_click_total[:, :cur_size]
        self.windowed_weight_total[:, :cur_size] = cur_weight_total[:, :cur_size]
        idx = cur_size

        for metric in metrics:
            if self.enable_lifetime:
                self.click_total += metric.click_total.to(self.device)
                self.weight_total += metric.weight_total.to(self.device)
            cur_size = min(metric.total_updates, metric.max_num_updates)
            self.windowed_click_total[:, idx : idx + cur_size] = (
                metric.windowed_click_total[:, :cur_size]
            )
            self.windowed_weight_total[:, idx : idx + cur_size] = (
                metric.windowed_weight_total[:, :cur_size]
            )
            idx += cur_size
            self.total_updates += metric.total_updates

        self.next_inserted = idx
        self.next_inserted %= self.max_num_updates
        return self
