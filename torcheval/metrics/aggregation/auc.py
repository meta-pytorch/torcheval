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
from torcheval.metrics.functional.aggregation.auc import (
    _auc_compute,
    _auc_update_input_check,
)
from torcheval.metrics.metric import Metric


TAUC = TypeVar("TAUC")


class AUC(Metric[torch.Tensor]):
    r"""
    Computes Area Under the Curve (AUC) using the trapezoidal rule. Supports x and y being two dimensional tensors,
    each row is treated as its own list of x and y coordinates returning one dimensional tensor should be
    returned with the AUC for each row calculated.

    Args:
    reorder (bool): Reorder the input tensor for auc computation. Default value is True.
    num_tasks (int):  Number of tasks that need AUC calculation. Default value is 1.

    >>> from torcheval.metrics.aggregation.auc import AUC

    >>> metric = AUC()
    >>> metric.update(torch.tensor([0,.2,.3,.1]), torch.tensor([1,1,1,1]))
    >>> metric.compute()
    tensor([0.3000])
    >>> metric.reset()
    >>> metric.update(torch.tensor([0,.1,.13,.2]), torch.tensor([1,1,2,4]))
    >>> metric.update(torch.tensor([1.,2.,.1, 3.]), torch.tensor([1,2,3,2]))
    >>> metric.compute()
    tensor([5.8850])
    >>> metric = AUC(n_tasks=2) # n_tasks should be equal to first dimension of x, y in update()
    >>> x = torch.tensor([[0.3941, 0.2980, 0.3080],
                          [0.1448, 0.6090, 0.2462]])
    >>> y = torch.tensor([[1, 0, 4],
                          [0, 4, 2]])
    >>> metric.update(x, y)
    >>> x1 = torch.tensor([[0.4562, 0.1200, 0.4238],
                           [0.4076, 0.4448, 0.1476]])
    >>> y1 = torch.tensor([[3, 4, 3],
                           [2, 0, 4]])
    >>> metric.update(x1, y1)
    >>> metric.compute()
    tensor([0.7479, 0.9898])
    """

    def __init__(
        self: TAUC,
        *,
        reorder: bool = True,
        n_tasks: int = 1,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(device=device)
        self._add_state("x", [])
        self._add_state("y", [])
        self.n_tasks = n_tasks
        self.reorder = reorder

    @torch.inference_mode()
    # pyre-ignore[14]: inconsistent override on *_:Any, **__:Any
    def update(self: TAUC, x: torch.Tensor, y: torch.Tensor) -> TAUC:
        r"""
        Updates and returns variables required to compute area under the curve.
        Args:
            x: x-coordinates,
            y: y-coordinates
        """
        _auc_update_input_check(x, y, n_tasks=self.n_tasks)

        if x.ndim == 1:
            x = x.unsqueeze(0)
        if y.ndim == 1:
            y = y.unsqueeze(0)

        self.x.append(x)
        self.y.append(y)

        return self

    @torch.inference_mode()
    def compute(self: TAUC) -> torch.Tensor:
        """Computes AUC based on inputs passed in to ``update`` previously."""

        if not self.x or not self.y:
            return torch.tensor([])

        return _auc_compute(
            torch.cat(self.x, dim=1), torch.cat(self.y, dim=1), reorder=self.reorder
        )

    @torch.inference_mode()
    def merge_state(self: TAUC, metrics: Iterable[TAUC]) -> TAUC:
        self._prepare_for_merge_state()
        for metric in metrics:
            if metric.x:
                metric_x = torch.cat(metric.x, dim=1).to(self.device)
                metric_y = torch.cat(metric.y, dim=1).to(self.device)
                self.x.append(metric_x)
                self.y.append(metric_y)
        return self

    @torch.inference_mode()
    def _prepare_for_merge_state(self: TAUC) -> None:
        if self.x and self.y:
            self.x = [torch.cat(self.x, dim=1)]
            self.y = [torch.cat(self.y, dim=1)]
