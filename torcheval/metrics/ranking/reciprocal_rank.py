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
from torcheval.metrics.functional import reciprocal_rank
from torcheval.metrics.metric import Metric


TReciprocalRank = TypeVar("TReciprocalRank")


class ReciprocalRank(Metric[torch.Tensor]):
    """
    Compute the reciprocal rank of the correct class among the top predicted classes.
    Its functional version is :func:`torcheval.metrics.functional.reciprocal_rank`.

    Args:
        k (int, optional): Number of top class probabilities to be considered.

    Examples::

        >>> import torch
        >>> from torcheval.metrics import ReciprocalRank

        >>> metric = ReciprocalRank()
        >>> metric.update(torch.tensor([[0.3, 0.1, 0.6], [0.5, 0.2, 0.3]]), torch.tensor([2, 1]))
        >>> metric.update(torch.tensor([[0.2, 0.1, 0.7], [0.3, 0.3, 0.4]]), torch.tensor([1, 0]))
        >>> metric.compute()
        tensor([1.0000, 0.3333, 0.3333, 0.5000])

        >>> metric = ReciprocalRank(k=2)
        >>> metric.update(torch.tensor([[0.3, 0.1, 0.6], [0.5, 0.2, 0.3]]), torch.tensor([2, 1]))
        >>> metric.update(torch.tensor([[0.2, 0.1, 0.7], [0.3, 0.3, 0.4]]), torch.tensor([1, 0]))
        >>> metric.compute()
        tensor([1.0000, 0.0000, 0.0000, 0.5000])
    """

    def __init__(
        self: TReciprocalRank,
        *,
        k: int | None = None,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(device=device)
        self.k = k
        self._add_state("scores", [])

    @torch.inference_mode()
    # pyre-ignore[14]: `update` overrides method defined in `Metric` inconsistently.
    def update(
        self: TReciprocalRank, input: torch.Tensor, target: torch.Tensor
    ) -> TReciprocalRank:
        """
        Update the metric state with the ground truth labels and predictions.

        Args:
            input (Tensor): Predicted unnormalized scores (often referred to as logits) or
                class probabilities of shape (num_samples, num_classes).
            target (Tensor): Ground truth class indices of shape (num_samples,).
        """
        self.scores.append(reciprocal_rank(input, target, k=self.k))
        return self

    @torch.inference_mode()
    def compute(self: TReciprocalRank) -> torch.Tensor:
        """
        Return the concatenated reciprocal rank scores. If no ``update()`` calls are made before
        ``compute()`` is called, return an empty tensor.
        """
        if not self.scores:
            return torch.empty(0)
        return torch.cat(self.scores, dim=0)

    @torch.inference_mode()
    def merge_state(
        self: TReciprocalRank, metrics: Iterable[TReciprocalRank]
    ) -> TReciprocalRank:
        """
        Merge the metric state with its counterparts from other metric instances.

        Args:
            metrics (Iterable[Metric]): metric instances whose states are to be merged.
        """
        for metric in metrics:
            if metric.scores:
                self.scores.append(torch.cat(metric.scores).to(self.device))
        return self

    @torch.inference_mode()
    def _prepare_for_merge_state(self: TReciprocalRank) -> None:
        if self.scores:
            self.scores = [torch.cat(self.scores)]
