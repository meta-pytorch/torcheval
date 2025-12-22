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

from torcheval.metrics.functional.text.word_information_lost import (
    _wil_compute,
    _wil_update,
)

from torcheval.metrics.metric import Metric
from torcheval.utils.device import largest_float

TWordInformationLost = TypeVar("TWordInformationLost")


class WordInformationLost(Metric[torch.Tensor]):
    r"""Word Information Lost (WIL) is a metric of the performance of an automatic speech recognition system. This
    value indicates the percentage of words that were incorrectly predicted between a set of ground-truth sentences
    and a set of hypothesis sentences. The lower the value, the better the performance of the ASR system with a
    WordInformationLost of 0 being a perfect score. Word Information Lost rate can then be computed as:

    .. math:: wil = 1 - \frac{C}{N} * \frac{C}{P}

    where:
        - :math:`C` is the number of correct words,
        - :math:`N` is the number of words in the reference
        - :math:`P` is the number of words in the prediction

    Its functional version is :func:`torcheval.metrics.functional.word_information_lost`.

    Examples:
        >>> from torcheval.metrics import WordInformationLost
        >>> preds = ["this is the prediction", "there is an other sample"]
        >>> target = ["this is the reference", "there is another one"]
        >>> metric = WordInformationLost()
        >>> metric.update(preds, target)
        >>> metric.compute()
        tensor(0.6528, dtype=torch.float64)
    """

    def __init__(
        self: TWordInformationLost,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(device=device)
        self._add_state(
            "correct_total",
            torch.tensor(0.0, dtype=largest_float(device), device=self.device),
        )
        self._add_state(
            "target_total",
            torch.tensor(0.0, dtype=largest_float(device), device=self.device),
        )
        self._add_state(
            "preds_total",
            torch.tensor(0.0, dtype=largest_float(device), device=self.device),
        )

    @torch.inference_mode()
    # pyre-ignore[14]: `update` overrides method defined in `Metric` inconsistently.
    def update(
        self: TWordInformationLost,
        input: str | list[str],
        target: str | list[str],
    ) -> TWordInformationLost:
        """Store predictions/references for computing Word Information Lost scores.
        Args:
            input: Transcription(s) to score as a string or list of strings
            target: Reference(s) for each speech input as a string or list of strings
        """
        correct_total, target_total, preds_total = _wil_update(
            input, target, self.device
        )
        self.correct_total += correct_total
        self.target_total += target_total
        self.preds_total += preds_total
        return self

    @torch.inference_mode()
    def compute(self: TWordInformationLost) -> torch.Tensor:
        """Calculate the Word Information Lost.
        Returns:
            Word Information Lost score
        """
        return _wil_compute(self.correct_total, self.target_total, self.preds_total)

    @torch.inference_mode()
    def merge_state(
        self: TWordInformationLost,
        metrics: Iterable[TWordInformationLost],
    ) -> TWordInformationLost:
        """
        Merge the metric state with its counterparts from other metric instances.

        Args:
            metrics (Iterable[Metric]): metric instances whose states are to be merged.
        """
        for metric in metrics:
            self.correct_total += metric.correct_total.to(self.device)
            self.target_total += metric.target_total.to(self.device)
            self.preds_total += metric.preds_total.to(self.device)
        return self
