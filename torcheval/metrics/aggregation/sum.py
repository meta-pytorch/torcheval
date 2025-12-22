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

from torcheval.metrics.functional.aggregation.sum import _sum_update
from torcheval.metrics.metric import Metric
from torcheval.utils.device import largest_float

TSum = TypeVar("TSum")


class Sum(Metric[torch.Tensor]):
    """
    Calculate the weighted sum value of all elements in all the input tensors.
    When weight is not provided, it calculates the unweighted sum.
    Its functional version is :func:`torcheval.metrics.functional.sum`.

    Examples::

        >>> import torch
        >>> from torcheval.metrics import Sum
        >>> metric = Sum()
        >>> metric.update(1)
        >>> metric.update(torch.tensor([2, 3]))
        >>> metric.compute()
        tensor(6.)
        >>> metric.update(torch.tensor(-1)).compute()
        tensor(5.)
        >>> metric.reset()
        >>> metric.update(torch.tensor(-1)).compute()
        tensor(-1.)

        >>> metric = Sum()
        >>> metric.update(torch.tensor([2, 3]), torch.tensor([0.1, 0.6])).compute()
        tensor(2.)
        >>> metric.update(torch.tensor([2, 3]), 0.5).compute()
        tensor(4.5)
        >>> metric.update(torch.tensor([4, 6]), 1).compute()
        tensor(14.5)
    """

    def __init__(
        self: TSum,
        *,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(device=device)
        self._add_state(
            "weighted_sum",
            torch.tensor(0.0, device=self.device, dtype=largest_float(device)),
        )

    @torch.inference_mode()
    # pyre-ignore[14]: inconsistent override on *_:Any, **__:Any
    def update(
        self: TSum,
        input: torch.Tensor,
        *,
        weight: float | int | torch.Tensor = 1.0,
    ) -> TSum:
        """
        Update states with the values and weights.

        Args:
            input (Tensor): Tensor of input values.
            weight(optional): Float or Int or Tensor of input weights. It is default to 1.0. If weight is a Tensor, its size should match the input tensor size.
        Raises:
            ValueError: If value of weight is neither a ``float`` nor ``int`` nor a ``torch.Tensor`` that matches the input tensor size.
        """

        self.weighted_sum += _sum_update(input, weight)
        return self

    @torch.inference_mode()
    def compute(self: TSum) -> torch.Tensor:
        return self.weighted_sum

    @torch.inference_mode()
    def merge_state(self: TSum, metrics: Iterable[TSum]) -> TSum:
        for metric in metrics:
            self.weighted_sum += metric.weighted_sum.to(self.device)
        return self
