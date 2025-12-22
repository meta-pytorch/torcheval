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

from torcheval.metrics.functional.text.word_error_rate import (
    _word_error_rate_compute,
    _word_error_rate_update,
)
from torcheval.metrics.metric import Metric
from torcheval.utils.device import largest_float

TWordErrorRate = TypeVar("TWordErrorRate")


class WordErrorRate(Metric[torch.Tensor]):
    """
    Compute the word error rate of the predicted word sequence(s) with the reference word sequence(s).
    Its functional version is :func:`torcheval.metrics.functional.word_error_rate`.

    Examples:

        >>> import torch
        >>> from torcheval.metrics import WordErrorRate

        >>> metric = WordErrorRate()
        >>> metric.update(["this is the prediction", "there is an other sample"],
        ["this is the reference", "there is another one"])
        >>> metric.compute()
        tensor(0.5)

        >>> metric = WordErrorRate()
        >>> metric.update(["this is the prediction", "there is an other sample"],
        ["this is the reference", "there is another one"])
        >>> metric.update(["hello world", "welcome to the facebook"],
        ["hello metaverse", "welcome to meta"])
        >>> metric.compute()
        tensor(0.53846)
    """

    def __init__(
        self: TWordErrorRate,
        *,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(device=device)
        dtype = largest_float(device)
        self._add_state("errors", torch.tensor(0, dtype=dtype, device=self.device))
        self._add_state("total", torch.tensor(0, dtype=dtype, device=self.device))

    @torch.inference_mode()
    # pyre-ignore[14]: `update` overrides method defined in `Metric` inconsistently.
    def update(
        self: TWordErrorRate,
        input: str | list[str],
        target: str | list[str],
    ) -> TWordErrorRate:
        """
        Update the metric state with edit distance and the length of the reference sequence.

        Args:
            input (str, List[str]): Predicted word sequence(s) to score as a string or list of strings.
            target (str, List[str]): Reference word sequence(s) as a string or list of strings.
        """
        errors, total = _word_error_rate_update(input, target, self.device)
        self.errors += errors
        self.total += total
        return self

    @torch.inference_mode()
    def compute(self: TWordErrorRate) -> torch.Tensor:
        """
        Return the word error rate score
        """
        return _word_error_rate_compute(self.errors, self.total)

    @torch.inference_mode()
    def merge_state(
        self: TWordErrorRate,
        metrics: Iterable[TWordErrorRate],
    ) -> TWordErrorRate:
        """
        Merge the metric state with its counterparts from other metric instances.

        Args:
            metrics (Iterable[Metric]): metric instances whose states are to be merged.
        """
        for metric in metrics:
            self.errors += metric.errors.to(self.device)
            self.total += metric.total.to(self.device)
        return self
