# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# pyre-ignore-all-errors[16]: Undefined attribute of metric states.

from collections.abc import Iterable
from typing import TypeVar, Union

import torch

from torcheval.metrics.functional.classification.binary_normalized_entropy import (
    _baseline_update,
    _binary_normalized_entropy_update,
)
from torcheval.metrics.metric import Metric
from torcheval.utils.device import largest_float

TWindowedNormalizedEntropy = TypeVar("TWindowedNormalizedEntropy")


class WindowedBinaryNormalizedEntropy(
    Metric[Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]]
):
    """
    The windowed version of BinaryNormalizedEntropy that provides both windowed and liftime values.
    Windowed value is calculated from the input and target of the last window_size number of `update()` calls.
    Lifetime value is calculated from all past input and target of `update()` calls.

    Compute the normalized binary cross entropy between predicted input and
    ground-truth binary target.

    Args:
        from_logits (bool): A boolean indicator whether the predicted value `y_pred` is
                    a floating-point logit value (i.e., value in [-inf, inf] when `from_logits=True`)
                    or a probablity value (i.e., value in [0., 1.] when `from_logits=False`)
                    Default value is False.
        num_tasks (int): Number of tasks that need BinaryNormalizedEntropy calculation. Default value
                    is 1. BinaryNormalizedEntropy for each task will be calculated independently.
        max_num_updates (int): The max window size that can accommodate the number of updates.
        enable_lifetime (bool): A boolean indicator whether to calculate lifetime values.

    Examples::

        >>> import torch
        >>> from torcheval.metrics import WindowedBinaryNormalizedEntropy

        >>> metric = WindowedBinaryNormalizedEntropy(max_num_updates=2)
        >>> metric.update(torch.tensor([0.2, 0.3]), torch.tensor([1.0, 0.0]))
        >>> metric.update(torch.tensor([0.5, 0.6]), torch.tensor([1.0, 1.0]))
        >>> metric.update(torch.tensor([0.6, 0.2]), torch.tensor([0.0, 1.0]))
        >>> metric.num_examples, metric.windowed_num_examples
        (tensor([6.], dtype=torch.float64), tensor([[2., 2.]], dtype=torch.float64))
        >>> metric.compute()
        (tensor([1.4914], dtype=torch.float64), tensor([1.6581], dtype=torch.float64))

        >>> metric = WindowedBinaryNormalizedEntropy(max_num_updates=2, enable_lifetime=False)
        >>> metric.update(torch.tensor([0.2, 0.3]), torch.tensor([1.0, 0.0]))
        >>> metric.update(torch.tensor([0.5, 0.6]), torch.tensor([1.0, 1.0]))
        >>> metric.update(torch.tensor([0.6, 0.2]), torch.tensor([0.0, 1.0]))
        >>> metric.windowed_num_examples
        tensor([[2., 2.]], dtype=torch.float64)
        >>> metric.compute()
        tensor([1.6581], dtype=torch.float64)

        >>> metric = WindowedBinaryNormalizedEntropy(max_num_updates=2, num_tasks=2)
        >>> metric.update(torch.tensor([[0.2, 0.3], [0.5, 0.1]]), torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
        >>> metric.update(torch.tensor([[0.8, 0.3], [0.6, 0.1]]), torch.tensor([[1.0, 1.0], [1.0, 0.0]]))
        >>> metric.update(torch.tensor([[0.5, 0.1], [0.3, 0.9]]), torch.tensor([[0.0, 1.0], [0.0, 0.0]]))
        >>> metric.num_examples, metric.windowed_num_examples
        (tensor([6., 6.], dtype=torch.float64),
        tensor([[2., 2.],
                [2., 2.]], dtype=torch.float64))
        >>> metric.compute()
        (tensor([1.6729, 1.6421], dtype=torch.float64),
        tensor([1.9663, 1.4562], dtype=torch.float64))
    """

    def __init__(
        self: TWindowedNormalizedEntropy,
        *,
        from_logits: bool = False,
        num_tasks: int = 1,
        max_num_updates: int = 100,
        enable_lifetime: bool = True,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(device=device)
        self.from_logits = from_logits
        if num_tasks < 1:
            raise ValueError(
                "`num_tasks` value should be greater than and equal to 1, but received {num_tasks}. "
            )
        if max_num_updates < 1:
            raise ValueError(
                "`max_num_updates` value should be greater than and equal to 1, but received {max_num_updates}. "
            )
        self.num_tasks = num_tasks
        self._add_state("max_num_updates", max_num_updates)
        self.next_inserted = 0
        self.enable_lifetime = enable_lifetime
        self._add_state("total_updates", 0)

        dtype = largest_float(device)
        if self.enable_lifetime:
            self._add_state(
                "total_entropy",
                torch.zeros(self.num_tasks, dtype=dtype, device=self.device),
            )
            self._add_state(
                "num_examples",
                torch.zeros(self.num_tasks, dtype=dtype, device=self.device),
            )
            self._add_state(
                "num_positive",
                torch.zeros(self.num_tasks, dtype=dtype, device=self.device),
            )

        self._add_state(
            "windowed_total_entropy",
            torch.zeros(
                self.num_tasks,
                self.max_num_updates,
                dtype=dtype,
                device=self.device,
            ),
        )
        self._add_state(
            "windowed_num_examples",
            torch.zeros(
                self.num_tasks,
                self.max_num_updates,
                dtype=dtype,
                device=self.device,
            ),
        )
        self._add_state(
            "windowed_num_positive",
            torch.zeros(
                self.num_tasks,
                self.max_num_updates,
                dtype=dtype,
                device=self.device,
            ),
        )

    @torch.inference_mode()
    # pyre-ignore[14]: `update` overrides method defined in `Metric` inconsistently.
    def update(
        self: TWindowedNormalizedEntropy,
        input: torch.Tensor,
        target: torch.Tensor,
        *,
        weight: torch.Tensor | None = None,
    ) -> TWindowedNormalizedEntropy:
        """
        Update the metric state with the total entropy, total number of examples and total number of
        positive targets.

        Args:
            input (Tensor): Predicted unnormalized scores (often referred to as logits) or
                binary class probabilities (num_tasks, num_samples).
            target (Tensor): Ground truth binary class indices (num_tasks, num_samples).
            weight (Tensor, optional): A manual rescaling weight to match input tensor shape (num_tasks, num_samples).
        """

        cross_entropy, num_positive, num_examples = _binary_normalized_entropy_update(
            input, target, self.from_logits, self.num_tasks, weight
        )
        if self.enable_lifetime:
            self.total_entropy += cross_entropy
            self.num_examples += num_examples
            self.num_positive += num_positive
        self.windowed_total_entropy[:, self.next_inserted] = cross_entropy
        self.windowed_num_examples[:, self.next_inserted] = num_examples
        self.windowed_num_positive[:, self.next_inserted] = num_positive
        self.next_inserted += 1
        self.next_inserted %= self.max_num_updates
        self.total_updates += 1
        return self

    @torch.inference_mode()
    def compute(
        self: TWindowedNormalizedEntropy,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Return the normalized binary cross entropy.  If no ``update()`` calls are made before
        ``compute()`` is called, return an empty tensor.

        Returns:
            Tensor or Tuple[Tensor, Tensor]: If `enable_lifetime` is equal to True, return Tuple[Tensor, Tensor],
            whose first position represents lifetime value, and the second represents windowed value. Otherwise,
            return a Tensor that represents windowed value.
            The return tensors is binary normalized entropy for each task (num_tasks,).
        """
        if self.total_updates == 0:
            naught = torch.empty(0, device=self.device)
            if self.enable_lifetime:
                return naught, naught
            else:
                return naught

        # For the case that window has been filled more than once
        if self.total_updates >= self.max_num_updates:
            windowed_baseline_entropy = _baseline_update(
                self.windowed_num_positive.sum(dim=1),
                self.windowed_num_examples.sum(dim=1),
            )
            windowed_cross_entropy = self.windowed_total_entropy.sum(
                dim=1
            ) / self.windowed_num_examples.sum(dim=1)
        else:  # For the case that window hasn't been filled
            windowed_baseline_entropy = _baseline_update(
                self.windowed_num_positive[:, : self.next_inserted].sum(dim=1),
                self.windowed_num_examples[:, : self.next_inserted].sum(dim=1),
            )
            windowed_cross_entropy = self.windowed_total_entropy[
                :, : self.next_inserted
            ].sum(dim=1) / self.windowed_num_examples[:, : self.next_inserted].sum(
                dim=1
            )
        windowed_normalized_entropy = windowed_cross_entropy / windowed_baseline_entropy

        if self.enable_lifetime:
            baseline_entropy = _baseline_update(self.num_positive, self.num_examples)
            cross_entropy = self.total_entropy / self.num_examples
            return (
                cross_entropy / baseline_entropy,
                windowed_normalized_entropy,
            )
        else:
            return windowed_normalized_entropy

    @torch.inference_mode()
    def merge_state(
        self: TWindowedNormalizedEntropy, metrics: Iterable[TWindowedNormalizedEntropy]
    ) -> TWindowedNormalizedEntropy:
        """
        Merge the metric state with its counterparts from other metric instances.
        First create tensors of size equal to the sum of all metrics' window sizes.
        Then, put all tensors to the front and leave the remaining indices zeros.

        Args:
            metrics (Iterable[Metric]): metric instances whose states are to be merged.
        """

        merge_max_num_updates = self.max_num_updates
        for metric in metrics:
            merge_max_num_updates += metric.max_num_updates
        cur_total_entropy = self.windowed_total_entropy
        cur_num_examples = self.windowed_num_examples
        cur_num_positive = self.windowed_num_positive
        dtype = largest_float(self.device)
        self.windowed_total_entropy = torch.zeros(
            self.num_tasks,
            merge_max_num_updates,
            dtype=dtype,
            device=self.device,
        )
        self.windowed_num_examples = torch.zeros(
            self.num_tasks,
            merge_max_num_updates,
            dtype=dtype,
            device=self.device,
        )
        self.windowed_num_positive = torch.zeros(
            self.num_tasks,
            merge_max_num_updates,
            dtype=dtype,
            device=self.device,
        )

        cur_size = min(self.total_updates, self.max_num_updates)
        self.windowed_total_entropy[:, :cur_size] = cur_total_entropy[:, :cur_size]
        self.windowed_num_examples[:, :cur_size] = cur_num_examples[:, :cur_size]
        self.windowed_num_positive[:, :cur_size] = cur_num_positive[:, :cur_size]
        idx = cur_size

        for metric in metrics:
            if self.enable_lifetime:
                self.total_entropy += metric.total_entropy.to(self.device)
                self.num_examples += metric.num_examples.to(self.device)
                self.num_positive += metric.num_positive.to(self.device)
            cur_size = min(metric.total_updates, metric.max_num_updates)
            self.windowed_total_entropy[:, idx : idx + cur_size] = (
                metric.windowed_total_entropy[:, :cur_size]
            )
            self.windowed_num_examples[:, idx : idx + cur_size] = (
                metric.windowed_num_examples[:, :cur_size]
            )
            self.windowed_num_positive[:, idx : idx + cur_size] = (
                metric.windowed_num_positive[:, :cur_size]
            )
            idx += cur_size
            self.total_updates += metric.total_updates

        self.next_inserted = idx
        self.next_inserted %= self.max_num_updates
        return self
