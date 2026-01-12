# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# pyre-ignore-all-errors[16]: Undefined attribute of metric states.

from collections import defaultdict
from collections.abc import Iterable
from typing import TypeVar

import torch
from torcheval.metrics import Metric

TDummySumMetric = TypeVar("TDummySumMetric")


class DummySumMetric(Metric[torch.Tensor]):
    def __init__(self: TDummySumMetric, *, device: torch.device | None = None) -> None:
        super().__init__(device=device)
        self._add_state("sum", torch.tensor(0.0, device=self.device))

    @torch.inference_mode()
    # pyre-ignore[14]: inconsistent override on *_:Any, **__:Any
    def update(self: TDummySumMetric, x: torch.Tensor) -> TDummySumMetric:
        self.sum += x
        return self

    @torch.inference_mode()
    def compute(self: TDummySumMetric) -> torch.Tensor:
        return self.sum

    @torch.inference_mode()
    def merge_state(
        self: TDummySumMetric, metrics: Iterable[TDummySumMetric]
    ) -> TDummySumMetric:
        for metric in metrics:
            self.sum += metric.sum.to(self.device)
        return self


TDummySumListStateMetric = TypeVar("TDummySumListStateMetric")


class DummySumListStateMetric(Metric[torch.Tensor]):
    def __init__(
        self: TDummySumListStateMetric, *, device: torch.device | None = None
    ) -> None:
        super().__init__(device=device)
        self._add_state("x", [])

    @torch.inference_mode()
    # pyre-ignore[14]: inconsistent override on *_:Any, **__:Any
    def update(
        self: TDummySumListStateMetric, x: torch.Tensor
    ) -> TDummySumListStateMetric:
        self.x.append(x.to(self.device))
        return self

    @torch.inference_mode()
    def compute(self: TDummySumListStateMetric) -> torch.Tensor:
        # pyre-fixme[7]: Expected `Tensor` but got `int`.
        return sum(tensor.sum() for tensor in self.x)

    @torch.inference_mode()
    def merge_state(
        self: TDummySumListStateMetric, metrics: Iterable[TDummySumListStateMetric]
    ) -> TDummySumListStateMetric:
        for metric in metrics:
            self.x.extend(element.to(self.device) for element in metric.x)
        return self


TDummySumDictStateMetric = TypeVar("TDummySumDictStateMetric")


class DummySumDictStateMetric(Metric[torch.Tensor]):
    def __init__(
        self: TDummySumDictStateMetric, *, device: torch.device | None = None
    ) -> None:
        super().__init__(device=device)
        self._add_state("x", defaultdict(lambda: torch.tensor(0.0, device=self.device)))

    @torch.inference_mode()
    # pyre-ignore[14]: inconsistent override on *_:Any, **__:Any
    def update(
        self: TDummySumDictStateMetric,
        k: str,
        v: torch.Tensor,
    ) -> TDummySumDictStateMetric:
        self.x[k] += v
        return self

    @torch.inference_mode()
    def compute(self: TDummySumDictStateMetric) -> torch.Tensor:
        return self.x

    @torch.inference_mode()
    def merge_state(
        self: TDummySumDictStateMetric, metrics: Iterable[TDummySumDictStateMetric]
    ) -> TDummySumDictStateMetric:
        for metric in metrics:
            for k in metric.keys():
                self.x[k] += metric.x[k].to(self.device)

        return self
