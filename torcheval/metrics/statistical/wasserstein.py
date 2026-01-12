# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Iterable
from typing import TypeVar

import torch
from torcheval.metrics.functional.statistical.wasserstein import (
    _wasserstein_compute,
    _wasserstein_update_input_check,
)
from torcheval.metrics.metric import Metric

TWasserstein = TypeVar("TWasserstein")


class Wasserstein1D(Metric[torch.Tensor]):
    r"""
    The Wasserstein distance, also called the Earth Mover's Distance, is a
    measure of the similarity between two distributions.

    The Wasserstein distance between two distributions is intuitively the
    minimum weight of soil (times distance moved) that would need to be moved
    if the two distributions were represented by two piles of soil.

    Its functional version is :func:'torcheval.metrics.functional.statistical.wasserstein'.

    Examples
    --------
    >>> from torcheval.metrics import Wasserstein1D
    >>> metric = Wasserstein1D()
    >>> metric.update(torch.tensor([0,1,2,2]), torch.tensor([0,1]))
    >>> metric.compute()
    0.75
    >>> metric = Wasserstein1D()
    >>> metric.update(torch.tensor([0,1,2]), torch.tensor([0,1,1]), torch.tensor([1,2,0]), torch.tensor([1,1,1]))
    >>> metric.compute()
    0
    >>> metric = Wasserstein1D()
    >>> metric.update(torch.tensor([0,1,2]), torch.tensor([0,1,1]))
    >>> metric.compute()
    0.33333333333333337
    >>> metric.update(torch.tensor([1,1,1]), torch.tensor([1,1,1]))
    >>> metric.compute()
    0.16666666666666663
    """

    def __init__(self: TWasserstein, *, device: torch.device | None = None) -> None:
        super().__init__(device=device)
        # Keeping record of samples
        self._add_state("dist_1_samples", [])
        self._add_state("dist_2_samples", [])
        self._add_state("dist_1_weights", [])
        self._add_state("dist_2_weights", [])

    @torch.inference_mode()
    def update(
        self,
        new_samples_dist_1: torch.Tensor,
        new_samples_dist_2: torch.Tensor,
        new_weights_dist_1: torch.Tensor | None = None,
        new_weights_dist_2: torch.Tensor | None = None,
    ) -> None:
        r"""
        Update states with distribution values and corresponding weights.

        Args:
        new_samples_dist_1, new_samples_dist_2 (Tensor) : 1D Tensor values observed in the distribution.
        new_weights_dist_1, new_weights_dist_2 (Tensor): Optional tensor weights for each value.
            If unspecified, each value is assigned the same value (1.0).
        """
        _wasserstein_update_input_check(
            new_samples_dist_1,
            new_samples_dist_2,
            new_weights_dist_1,
            new_weights_dist_2,
        )

        new_samples_dist_1 = new_samples_dist_1.to(self.device)
        new_samples_dist_2 = new_samples_dist_2.to(self.device)

        if new_weights_dist_1 is None:
            new_weights_dist_1 = torch.ones_like(new_samples_dist_1, dtype=torch.float)
        else:
            new_weights_dist_1 = new_weights_dist_1.to(self.device)

        if new_weights_dist_2 is None:
            new_weights_dist_2 = torch.ones_like(new_samples_dist_2, dtype=torch.float)
        else:
            new_weights_dist_2 = new_weights_dist_2.to(self.device)

        # When new data comes in, just add them to the list of samples
        self.dist_1_samples.append(new_samples_dist_1)
        self.dist_2_samples.append(new_samples_dist_2)
        self.dist_1_weights.append(new_weights_dist_1)
        self.dist_2_weights.append(new_weights_dist_2)

        return self

    @torch.inference_mode()
    def compute(self):
        r"""
        Return Wasserstein distance.  If no ``update()`` calls are made before
        ``compute()`` is called, return an empty tensor.

        Returns:
            Tensor: The return value of Wasserstein value.
        """
        return _wasserstein_compute(
            torch.cat(self.dist_1_samples, -1),
            torch.cat(self.dist_2_samples, -1),
            torch.cat(self.dist_1_weights, -1),
            torch.cat(self.dist_2_weights, -1),
        )

    @torch.inference_mode()
    def merge_state(
        self: TWasserstein, metrics: Iterable[TWasserstein]
    ) -> TWasserstein:
        for metric in metrics:
            if metric.dist_1_samples != []:
                metric_dist_1_samples = torch.cat(metric.dist_1_samples, -1).to(
                    self.device
                )
                self.dist_1_samples.append(metric_dist_1_samples)

                metric_dist_2_samples = torch.cat(metric.dist_2_samples, -1).to(
                    self.device
                )
                self.dist_2_samples.append(metric_dist_2_samples)

                metric_dist_1_weights = torch.cat(metric.dist_1_weights, -1).to(
                    self.device
                )
                self.dist_1_weights.append(metric_dist_1_weights)

                metric_dist_2_weights = torch.cat(metric.dist_2_weights, -1).to(
                    self.device
                )
                self.dist_2_weights.append(metric_dist_2_weights)

        return self
