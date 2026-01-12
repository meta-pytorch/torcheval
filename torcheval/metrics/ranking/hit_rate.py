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
from torcheval.metrics.functional import hit_rate
from torcheval.metrics.metric import Metric

THitRate = TypeVar("THitRate")


class HitRate(Metric[torch.Tensor]):
    """
    Compute the hit rate of the correct class among the top predicted classes.
    Its functional version is :func:`torcheval.metrics.functional.hit_rate`.

    Args:
        k (int, optional): Number of top class probabilities to be considered.
            If k is None, all classes are considered and a hit rate of 1.0 is returned.

    Examples::

        >>> import torch
        >>> from torcheval.metrics import HitRate

        >>> metric = HitRate()
        >>> metric.update(torch.tensor([[0.3, 0.1, 0.6], [0.5, 0.2, 0.3]]), torch.tensor([2, 1]))
        >>> metric.update(torch.tensor([[0.2, 0.1, 0.7], [0.3, 0.3, 0.4]]), torch.tensor([1, 0]))
        >>> metric.compute()
        tensor([1., 1., 1., 1.])

        >>> metric = HitRate(k=2)
        >>> metric.update(torch.tensor([[0.3, 0.1, 0.6], [0.5, 0.2, 0.3]]), torch.tensor([2, 1]))
        >>> metric.update(torch.tensor([[0.2, 0.1, 0.7], [0.3, 0.3, 0.4]]), torch.tensor([1, 0]))
        >>> metric.compute()
        tensor([1., 0., 0., 1.])
    """

    def __init__(
        self: THitRate,
        *,
        k: int | None = None,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(device=device)
        self.k = k
        self._add_state("scores", [])

    @torch.inference_mode()
    # pyre-ignore[14]: `update` overrides method defined in `Metric` inconsistently.
    def update(self: THitRate, input: torch.Tensor, target: torch.Tensor) -> THitRate:
        """
        Update the metric state with the ground truth labels and predictions.

        Args:
            input (Tensor): Predicted unnormalized scores (often referred to as logits) or
                class probabilities of shape (num_samples, num_classes).
            target (Tensor): Ground truth class indices of shape (num_samples,).
        """
        self.scores.append(hit_rate(input, target, k=self.k))
        return self

    @torch.inference_mode()
    def compute(self: THitRate) -> torch.Tensor:
        """
        Return the concatenated hite rate scores. If no ``update()`` calls are made before
        ``compute()`` is called, return an empty tensor.
        """
        if not self.scores:
            return torch.empty(0)
        return torch.cat(self.scores, dim=0)

    @torch.inference_mode()
    def merge_state(self: THitRate, metrics: Iterable[THitRate]) -> THitRate:
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
    def _prepare_for_merge_state(self: THitRate) -> None:
        if self.scores:
            self.scores = [torch.cat(self.scores)]
