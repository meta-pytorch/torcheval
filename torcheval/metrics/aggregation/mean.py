# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# pyre-ignore-all-errors[16]: Undefined attribute of metric states.

import logging
from collections.abc import Iterable
from typing import TypeVar

import torch

from torcheval.metrics.functional.aggregation.mean import _mean_update
from torcheval.metrics.metric import Metric
from torcheval.utils.device import largest_float

TMean = TypeVar("TMean")


class Mean(Metric[torch.Tensor]):
    """
    Calculate the weighted mean value of all elements in all the input tensors.
    When weight is not provided, it calculates the unweighted mean.
    Its functional version is :func:`torcheval.metrics.functional.mean`.

    Examples::

        >>> import torch
        >>> from torcheval.metrics import Mean
        >>> metric = Mean()
        >>> metric.update(1)
        >>> metric.update(torch.tensor([2, 3]))
        >>> metric.compute()
        tensor(2.)

        >>> metric.update(torch.tensor(-1)).compute()
        tensor(1.25)

        >>> metric.reset()
        >>> metric.update(torch.tensor(-1)).compute()
        tensor(-1.)

        >>> metric = Mean()
        >>> metric.update(torch.tensor([2, 3]), torch.tensor([0.2, 0.8])).compute()
        tensor(2.8)
        >>> metric.update(torch.tensor([4, 5]), 0.5).compute()
        tensor(3.65)
        >>> metric.update(torch.tensor([6]), 2).compute()
        tensor(4.825)
    """

    def __init__(
        self: TMean,
        *,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(device=device)
        dtype = largest_float(device)
        # weighted sum of values over the entire state
        self._add_state(
            "weighted_sum", torch.tensor(0.0, device=self.device, dtype=dtype)
        )
        # sum total of weights over the entire state
        self._add_state("weights", torch.tensor(0.0, device=self.device, dtype=dtype))

    @torch.inference_mode()
    # pyre-ignore[14]: inconsistent override on *_:Any, **__:Any
    def update(
        self: TMean,
        input: torch.Tensor,
        *,
        weight: float | int | torch.Tensor = 1.0,
    ) -> TMean:
        """
        Compute weighted mean. When weight is not provided, it calculates the unweighted mean.

        weighted_mean = sum(weight * input) / sum(weight)

        Args:
            input (Tensor): Tensor of input values.
            weight(optional): Float or Int or Tensor of input weights. It is default to 1.0. If weight is a Tensor, its size should match the input tensor size.
        Raises:
            ValueError: If value of weight is neither a ``float`` nor a ``int'' nor a ``torch.Tensor`` that matches the input tensor size.
        """

        weighted_sum, net_weight = _mean_update(input, weight)
        self.weighted_sum += weighted_sum
        self.weights += net_weight
        return self

    @torch.inference_mode()
    def compute(self: TMean) -> torch.Tensor:
        """
        If no calls to ``update()`` are made before ``compute()`` is called,
        the function throws a warning and returns 0.0.
        """
        if not torch.is_nonzero(self.weights):
            logging.warning(
                "There is no weight for the average, no samples with weight have been added (did you ever run update()?)- returning 0.0"
            )
            return torch.tensor(0.0, dtype=largest_float(self.device))
        return self.weighted_sum / self.weights

    @torch.inference_mode()
    def merge_state(self: TMean, metrics: Iterable[TMean]) -> TMean:
        for metric in metrics:
            self.weighted_sum += metric.weighted_sum.to(self.device)
            self.weights += metric.weights.to(self.device)
        return self
