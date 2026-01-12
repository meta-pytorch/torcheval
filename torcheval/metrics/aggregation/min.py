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
from torcheval.metrics.metric import Metric


TMin = TypeVar("TMin")


class Min(Metric[torch.Tensor]):
    """
    Calculate the minimum value of all elements in all the input tensors.
    Its functional version is ``torch.min(input)``.

    Examples::

        >>> import torch
        >>> from torcheval.metrics import Min
        >>> metric = Min()
        >>> metric.update(torch.tensor([[1, 2], [3, 4]]))
        >>> metric.compute()
        tensor(1.)

        >>> metric.update(torch.tensor(-1)).compute()
        tensor(-1.)

        >>> metric.reset()
        >>> metric.update(torch.tensor(5)).compute()
        tensor(5.)
    """

    def __init__(
        self: TMin,
        *,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(device=device)
        self._add_state("min", torch.tensor(float("inf"), device=self.device))

    @torch.inference_mode()
    # pyre-ignore[14]: inconsistent override on *_:Any, **__:Any
    def update(self: TMin, input: torch.Tensor) -> TMin:
        self.min = torch.min(self.min, torch.min(input))
        return self

    @torch.inference_mode()
    def compute(self: TMin) -> torch.Tensor:
        return self.min

    @torch.inference_mode()
    def merge_state(self: TMin, metrics: Iterable[TMin]) -> TMin:
        for metric in metrics:
            self.min = torch.min(self.min, metric.min.to(self.device))
        return self
