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

from torcheval.metrics.functional.text.word_information_preserved import (
    _word_information_preserved_compute,
    _word_information_preserved_update,
)
from torcheval.metrics.metric import Metric
from torcheval.utils.device import largest_float

TWordInformationPreserved = TypeVar("TWordInformationPreserved")


class WordInformationPreserved(Metric[torch.Tensor]):
    """
    Compute the word information preserved of the predicted word sequence(s) with the reference word sequence(s).
    Its functional version is :func:`torcheval.metrics.functional.word_information_preserved`.

    Examples:

        >>> import torch
        >>> from torcheval.metrics import WordInformationPreserved

        >>> metric = WordInformationPreserved()
        >>> metric.update(["this is the prediction", "there is an other sample"],
        ["this is the reference", "there is another one"])
        >>> metric.compute()
        tensor(0.3472)

        >>> metric = WordInformationPreserved()
        >>> metric.update(["hello world", "welcome to the facebook"],
        ["hello metaverse", "welcome to meta"])
        >>> metric.compute()
        tensor(0.3)
    """

    def __init__(
        self: TWordInformationPreserved,
        *,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(device=device)
        self._add_state(
            "correct_total",
            torch.tensor(0, dtype=largest_float(device), device=self.device),
        )
        self._add_state(
            "input_total",
            torch.tensor(0, dtype=largest_float(device), device=self.device),
        )
        self._add_state(
            "target_total",
            torch.tensor(0, dtype=largest_float(device), device=self.device),
        )

    @torch.inference_mode()
    # pyre-ignore[14]: `update` overrides method defined in `Metric` inconsistently.
    def update(
        self: TWordInformationPreserved,
        input: str | list[str],
        target: str | list[str],
    ) -> TWordInformationPreserved:
        """
        Update the metric state with correct_total, predicted length and reference length.

        Args:
            input (str, List[str]): Predicted word sequence(s) to score as a string or list of strings.
            target (str, List[str]): Reference word sequence(s) as a string or list of strings.
        """
        correct_total, target_total, input_total = _word_information_preserved_update(
            input, target, self.device
        )
        self.correct_total += correct_total
        self.target_total += target_total
        self.input_total += input_total
        return self

    @torch.inference_mode()
    def compute(self: TWordInformationPreserved) -> torch.Tensor:
        """
        Return the word information preserved score.
        """
        return _word_information_preserved_compute(
            self.correct_total, self.target_total, self.input_total
        )

    @torch.inference_mode()
    def merge_state(
        self: TWordInformationPreserved,
        metrics: Iterable[TWordInformationPreserved],
    ) -> TWordInformationPreserved:
        """
        Merge the metric state with its counterparts from other metric instances.

        Args:
            metrics (Iterable[Metric]): metric instances whose states are to be merged.
        """
        for metric in metrics:
            self.correct_total += metric.correct_total.to(self.device)
            self.target_total += metric.target_total.to(self.device)
            self.input_total += metric.input_total.to(self.device)
        return self
