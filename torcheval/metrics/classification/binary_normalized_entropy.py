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

from torcheval.metrics.functional.classification.binary_normalized_entropy import (
    _baseline_update,
    _binary_normalized_entropy_update,
)
from torcheval.metrics.metric import Metric
from torcheval.utils.device import largest_float

TNormalizedEntropy = TypeVar("TNormalizedEntropy")


class BinaryNormalizedEntropy(Metric[torch.Tensor]):
    """
    Compute the normalized binary cross entropy between predicted input and
    ground-truth binary target.
    Its functional version is :func:`torcheval.metrics.functional.binary_normalized_entropy`

    Args:
        from_logits (bool): A boolean indicator whether the predicted value `y_pred` is
                    a floating-point logit value (i.e., value in [-inf, inf] when `from_logits=True`)
                    or a probablity value (i.e., value in [0., 1.] when `from_logits=False`)
                    Default value is False.
        num_tasks (int): Number of tasks that need BinaryNormalizedEntropy calculation. Default value
                    is 1. BinaryNormalizedEntropy for each task will be calculated independently.

    Examples::

        >>> import torch
        >>> from torcheval.metrics import BinaryNormalizedEntropy

        >>> metric = BinaryNormalizedEntropy()
        >>> metric.update(torch.tensor([0.2, 0.3]), torch.tensor([1.0, 0.0]))
        >>> metric.compute()
        tensor([1.4183], dtype=torch.float64)

        >>> metric = BinaryNormalizedEntropy()
        >>> metric.update(torch.tensor([0.2, 0.3]), torch.tensor([1.0, 0.0]), torch.tensor([5.0, 1.0]))
        >>> metric.compute()
        tensor([3.1087], dtype=torch.float64)

        >>> metric = BinaryNormalizedEntropy(from_logits = True)
        >>> metric.update(tensor([-1.3863, -0.8473]), torch.tensor([1.0, 0.0]))
        >>> metric.compute()
        tensor([1.4183], dtype=torch.float64)

        >>> metric = BinaryNormalizedEntropy(num_tasks=2)
        >>> metric.update(torch.tensor([[0.2, 0.3], [0.5, 0.1]]), torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
        >>> metric.compute()
        tensor([1.4183, 2.1610], dtype=torch.float64)
    """

    def __init__(
        self: TNormalizedEntropy,
        *,
        from_logits: bool = False,
        num_tasks: int = 1,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(device=device)
        self.from_logits = from_logits
        if num_tasks < 1:
            raise ValueError(
                "`num_tasks` value should be greater than and equal to 1, but received {num_tasks}. "
            )
        self.num_tasks = num_tasks
        dtype = largest_float(device)
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

    @torch.inference_mode()
    # pyre-ignore[14]: `update` overrides method defined in `Metric` inconsistently.
    def update(
        self: TNormalizedEntropy,
        input: torch.Tensor,
        target: torch.Tensor,
        *,
        weight: torch.Tensor | None = None,
    ) -> TNormalizedEntropy:
        """
        Update the metric state with the total entropy, total number of examples and total number of
        positive targets.

        Args:
            input (Tensor): Predicted unnormalized scores (often referred to as logits) or
                binary class probabilities (num_tasks, num_samples).
            target (Tensor): Ground truth binary class indices (num_tasks, num_samples).
            weight (Tensor, optional): A manual rescaling weight to match input tensor shape (num_tasks, num_samples).
        """
        input = input.to(self.device)
        target = target.to(self.device)

        cross_entropy, num_positive, num_examples = _binary_normalized_entropy_update(
            input, target, self.from_logits, self.num_tasks, weight
        )
        self.total_entropy += cross_entropy
        self.num_examples += num_examples
        self.num_positive += num_positive
        return self

    @torch.inference_mode()
    def compute(self: TNormalizedEntropy) -> torch.Tensor:
        """
        Return the normalized binary cross entropy.  If no ``update()`` calls are made before
        ``compute()`` is called, return an empty tensor.

        Returns:
            Tensor: The return value of binary normalized entropy for each task (num_tasks,).
        """
        if torch.any(self.num_examples == 0.0):
            return torch.empty(0)

        baseline_entropy = _baseline_update(self.num_positive, self.num_examples)
        cross_entropy = self.total_entropy / self.num_examples
        return cross_entropy / baseline_entropy

    @torch.inference_mode()
    def merge_state(
        self: TNormalizedEntropy, metrics: Iterable[TNormalizedEntropy]
    ) -> TNormalizedEntropy:
        """
        Merge the metric state with its counterparts from other metric instances.

        Args:
            metrics (Iterable[Metric]): metric instances whose states are to be merged.
        """
        for metric in metrics:
            self.total_entropy += metric.total_entropy.to(self.device)
            self.num_examples += metric.num_examples.to(self.device)
            self.num_positive += metric.num_positive.to(self.device)
        return self
