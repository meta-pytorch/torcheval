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


TMax = TypeVar("TMax")


class Max(Metric[torch.Tensor]):
    """
    Calculate the maximum value of all elements in all the input tensors.
    Its functional version is ``torch.max(input)``.

    Examples::

        >>> import torch
        >>> from torcheval.metrics import Max
        >>> metric = Max()
        >>> metric.update(torch.tensor([[1, 2], [3, 4]]))
        >>> metric.compute()
        tensor(4.)

        >>> metric.update(torch.tensor(-1)).compute()
        tensor(4.)

        >>> metric.reset()
        >>> metric.update(torch.tensor(-1)).compute()
        tensor(-1.)
    """

    def __init__(
        self: TMax,
        *,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(device=device)
        self._add_state("max", torch.tensor(float("-inf"), device=self.device))

    @torch.inference_mode()
    # pyre-ignore[14]: inconsistent override on *_:Any, **__:Any
    def update(self: TMax, input: torch.Tensor) -> TMax:
        self.max = torch.max(self.max, torch.max(input))
        return self

    @torch.inference_mode()
    def compute(self: TMax) -> torch.Tensor:
        return self.max

    @torch.inference_mode()
    def merge_state(self: TMax, metrics: Iterable[TMax]) -> TMax:
        for metric in metrics:
            self.max = torch.max(self.max, metric.max.to(self.device))
        return self
