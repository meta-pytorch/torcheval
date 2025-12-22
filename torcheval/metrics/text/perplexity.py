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

from torcheval.metrics.functional.text.perplexity import (
    _perplexity_compute,
    _perplexity_update,
)
from torcheval.metrics.metric import Metric
from torcheval.utils.device import largest_float

TPerplexity = TypeVar("TPerplexity")


class Perplexity(Metric[torch.Tensor]):
    """
    Perplexity measures how well a model predicts sample data. It is calculated by:

    ppl = exp (sum of negative log likelihood / number of tokens)

    Its functional version is :func:`torcheval.metrics.functional.perplexity`.

    Args:
        ignore_index (Tensor):
            if specified, the target class with 'ignore_index' will be ignored when
            calculating perplexity. The default value is None.

    Examples:
        >>> import torch
        >>> from torcheval.metrics.text import Perplexity

        >>> metric=Perplexity()
        >>> input = torch.tensor([[[0.3659, 0.7025, 0.3104]], [[0.0097, 0.6577, 0.1947]],[[0.5659, 0.0025, 0.0104]], [[0.9097, 0.0577, 0.7947]]])
        >>> target = torch.tensor([[2],  [1], [2],  [1]])
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor(3.5257, dtype=torch.float64)

        >>> metric=Perplexity(ignore_index=1)
        >>> input = torch.tensor([[[0.3659, 0.7025, 0.3104]], [[0.0097, 0.6577, 0.1947]],[[0.5659, 0.0025, 0.0104]], [[0.9097, 0.0577, 0.7947]]])
        >>> target = torch.tensor([[2],  [1], [2],  [1]])
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor(3.6347, dtype=torch.float64)

        >>> metric1=Perplexity()
        >>> input = torch.tensor([[[0.5659, 0.0025, 0.0104]], [[0.9097, 0.0577, 0.7947]]])
        >>> target = torch.tensor([[2],  [1], ])
        >>> metric1.update(input, target)
        >>> metric1.compute()
        tensor(4.5051, dtype=torch.float64)

        >>> metric2=Perplexity()
        >>> input = torch.tensor([[[0.3659, 0.7025, 0.3104]], [[0.0097, 0.6577, 0.1947]]])
        >>> target = torch.tensor([[2],  [1]])
        >>> metric2.update(input, target)
        >>> metric2.compute())
        tensor(2.7593, dtype=torch.float64)

        >>> metric1.merge_state([metric2])
        >>> metric1.compute())
        tensor(3.5257, dtype=torch.float64)
    """

    def __init__(
        self: TPerplexity,
        ignore_index: int | None = None,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(device=device)

        self.ignore_index = ignore_index
        dtype = largest_float(device)
        self._add_state(
            "sum_log_probs", torch.tensor(0.0, dtype=dtype, device=self.device)
        )
        self._add_state("num_total", torch.tensor(0.0, dtype=dtype, device=self.device))

    @torch.inference_mode()
    # pyre-ignore[14]: `update` overrides method defined in `Metric` inconsistently.
    def update(
        self: TPerplexity,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> TPerplexity:
        """
        Update the metric state with new inputs.

        Args:
            input (Tensor):
                Predicted unnormalized scores (i.e., logits) for each token with shape
                of (n_samples, seq_len, vocab_size).
            target (Tensor):
                Tensor of ground truth vocab index with shape of (n_samples, seq_len).

        """
        sum_log_probs, num_total = _perplexity_update(input, target, self.ignore_index)
        self.sum_log_probs += sum_log_probs
        self.num_total += num_total

        return self

    @torch.inference_mode()
    def compute(self: TPerplexity) -> torch.Tensor:
        """
        Calculates perplexity based on `sum_log_probs` and `num_total`.
        If no `update()` calls are made before `compute()`  is called, return an empty tensor.
        """
        if self.num_total == 0.0:
            return torch.empty(0, device=self.device)
        return _perplexity_compute(self.sum_log_probs, self.num_total)

    @torch.inference_mode()
    def merge_state(self: TPerplexity, metrics: Iterable[TPerplexity]) -> TPerplexity:
        """
        Merge the metric state with its counterparts from other metric instances.

        Args:
            metrics (Iterable[Metric]): metric instances whose states are to be merged.
        """
        for metric in metrics:
            self.sum_log_probs += metric.sum_log_probs.to(self.device)
            self.num_total += metric.num_total.to(self.device)
        return self
