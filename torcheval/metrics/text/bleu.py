# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# pyre-ignore-all-errors[16]: Undefined attribute of metric states.

from collections.abc import Iterable, Sequence
from typing import TypeVar

import torch
from torcheval.metrics.functional.text.bleu import (
    _bleu_score_compute,
    _bleu_score_update,
)

from torcheval.metrics.metric import Metric
from torcheval.utils.device import largest_float

TBLEUScore = TypeVar("TBLEUScore")


class BLEUScore(Metric[torch.Tensor]):
    """
    Compute BLEU score (https://en.wikipedia.org/wiki/BLEU) given translations and references.
    Its functional version is :func:`torcheval.metrics.functional.bleu_score`.

    Args:
        n_gram: Maximum n-gram to use when computing BLEU score. Can be 1, 2, 3, or 4.
        weights: Optional weight distribution of n-grams. Requires len(weights) = n_gram. If unspecified, will use uniform weights.

    Examples:
        >>> import torch
        >>> from torcheval.metrics import BLEUScore
        >>> metric = BLEUScore(n_gram=4)
        >>> candidates = ["the squirrel is eating the nut", "the cat is on the mat"]
        >>> references = [["a squirrel is eating a nut", "the squirrel is eating a tasty nut"], ["there is a cat on the mat", "a cat is on the mat"]]
        >>> metric.update(candidates, references)
        >>> metric.compute()
        tensor(0.65341892)
        >>> candidates = ["i like ice cream and apple pie"]
        >>> references = [["i like apple pie with ice cream on top", "i like ice cream with my apple pie", "i enjoy my apple pie with ice cream"]]
        >>> metric.update(candidates, references)
        >>> metric.compute()
        tensor(0.56377503)
    """

    def __init__(
        self: TBLEUScore,
        *,
        n_gram: int,
        weights: torch.Tensor | None = None,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(device=device)

        if n_gram not in [1, 2, 3, 4]:
            raise ValueError(f"n_gram should be 1, 2, 3, or 4, got {n_gram}.")
        if weights is not None and n_gram != len(weights):
            raise ValueError(
                f"the length of weights should equal n_gram, got len(weights)={len(weights)}, n_gram={n_gram}"
            )

        self.weights = weights
        self.n_gram = n_gram
        dtype = largest_float(device)
        self._add_state("input_len", torch.tensor(0.0, dtype=dtype, device=device))
        self._add_state("target_len", torch.tensor(0.0, dtype=dtype, device=device))
        self._add_state(
            "matches_by_order",
            torch.zeros(n_gram, dtype=dtype, device=device),
        )
        self._add_state(
            "possible_matches_by_order",
            torch.zeros(n_gram, dtype=dtype, device=device),
        )

    @torch.inference_mode()
    # pyre-ignore[14]: `update` overrides method defined in `Metric` inconsistently.
    def update(
        self: TBLEUScore,
        input: str | Sequence[str],
        target: Sequence[str | Sequence[str]],
    ) -> TBLEUScore:
        """
        Update the metric state with new inputs.

        Args:
            input: Translations to score.
            target: List of references for each translation.
        """
        (
            input_len,
            target_len,
            matches_by_order,
            possible_matches_by_order,
        ) = _bleu_score_update(input, target, self.n_gram, self.device)
        self.input_len += input_len
        self.target_len += target_len
        self.matches_by_order += matches_by_order
        self.possible_matches_by_order += possible_matches_by_order
        return self

    @torch.inference_mode()
    def compute(self: TBLEUScore) -> torch.Tensor:
        """
        Returns the running BLEUScore. If no ``update()`` calls are made before
        ``compute()`` is called, return tensor(0.0).
        """
        if torch.sum(self.matches_by_order) == 0:
            return torch.tensor(
                0.0, dtype=largest_float(self.device), device=self.device
            )
        return _bleu_score_compute(
            self.input_len,
            self.target_len,
            self.matches_by_order,
            self.possible_matches_by_order,
            self.n_gram,
            self.weights,
        )

    @torch.inference_mode()
    def merge_state(self: TBLEUScore, metrics: Iterable[TBLEUScore]) -> TBLEUScore:
        """
        Merge the metric state with its counterparts from other metric instances.

        Args:
            metrics (Iterable[Metric]): metric instances whose states are to be merged.
        """
        for metric in metrics:
            self.input_len += metric.input_len.to(self.device)
            self.target_len += metric.target_len.to(self.device)
            self.matches_by_order += metric.matches_by_order.to(self.device)
            self.possible_matches_by_order += metric.possible_matches_by_order.to(
                self.device
            )
        return self
