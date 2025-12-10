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

from torcheval.metrics.functional.classification.recall_at_fixed_precision import (
    _binary_recall_at_fixed_precision_compute,
    _binary_recall_at_fixed_precision_update_input_check,
    _multilabel_recall_at_fixed_precision_compute,
    _multilabel_recall_at_fixed_precision_update_input_check,
)
from torcheval.metrics.metric import Metric

"""
This file contains BinaryRecallAtFixedPrecision and MultilabelRecallAtFixedPrecision classes.
"""

TBinaryRecallAtFixedPrecision = TypeVar("TBinaryRecallAtFixedPrecision")
TMultilabelRecallAtFixedPrecision = TypeVar("TMultilabelRecallAtFixedPrecision")


class BinaryRecallAtFixedPrecision(Metric[tuple[torch.Tensor, torch.Tensor]]):
    """
    Returns the highest possible recall value give the minimum precision
    for binary classification tasks.

    Its functional version is :func:`torcheval.metrics.functional.binary_recall_at_fixed_precision`.
    See also :class:`MultilabelRecallAtFixedPrecision <MultilabelRecallAtFixedPrecision>`

    Args:
        min_precision (float): Minimum precision threshold
        use_weights (bool): If True, enables weighted recall computation.
            When enabled, the `update()` method requires a `weight` parameter.
            Default: False

    Examples::

        >>> import torch
        >>> from torcheval.metrics import BinaryRecallAtFixedPrecision
        >>> metric = BinaryRecallAtFixedPrecision(min_precision=0.5)
        >>> input = torch.tensor([0.1, 0.4, 0.6, 0.6, 0.6, 0.35, 0.8])
        >>> target = torch.tensor([0, 0, 1, 1, 1, 1, 1])
        >>> metric.update(input, target)
        >>> metric.compute()
        (torch.tensor(1.0), torch.tensor(0.35))

        >>> # With weights
        >>> metric = BinaryRecallAtFixedPrecision(min_precision=0.5, use_weights=True)
        >>> input = torch.tensor([0.1, 0.4, 0.6, 0.6, 0.6, 0.35, 0.8])
        >>> target = torch.tensor([0, 0, 1, 1, 1, 1, 1])
        >>> weight = torch.tensor([1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0])
        >>> metric.update(input, target, weight)
        >>> metric.compute()
        (torch.tensor(1.0), torch.tensor(0.35))
    """

    def __init__(
        self: TBinaryRecallAtFixedPrecision,
        *,
        min_precision: float,
        device: torch.device | None = None,
        use_weights: bool = False,
    ) -> None:
        super().__init__(device=device)
        self.min_precision = min_precision
        self.use_weights = use_weights
        self._add_state("inputs", [])
        self._add_state("targets", [])
        if self.use_weights:
            self._add_state("weights", [])

    @torch.inference_mode()
    # pyre-ignore[14]: inconsistent override on *_:Any, **__:Any
    def update(
        self: TBinaryRecallAtFixedPrecision,
        input: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor | None = None,
    ) -> TBinaryRecallAtFixedPrecision:
        input = input.to(self.device)
        target = target.to(self.device)

        _binary_recall_at_fixed_precision_update_input_check(
            input, target, self.min_precision
        )
        self.inputs.append(input)
        self.targets.append(target)

        if self.use_weights:
            if weight is None:
                raise ValueError("Weights must be provided when use_weights=True")
            weight = weight.to(self.device)
            self.weights.append(weight)
        return self

    @torch.inference_mode()
    def compute(
        self: TBinaryRecallAtFixedPrecision,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = torch.cat(self.inputs)
        targets = torch.cat(self.targets)
        if self.use_weights:
            weights = torch.cat(self.weights)
            return _binary_recall_at_fixed_precision_compute(
                inputs, targets, self.min_precision, weights
            )
        return _binary_recall_at_fixed_precision_compute(
            inputs, targets, self.min_precision
        )

    @torch.inference_mode()
    def merge_state(
        self: TBinaryRecallAtFixedPrecision,
        metrics: Iterable[TBinaryRecallAtFixedPrecision],
    ) -> TBinaryRecallAtFixedPrecision:
        for metric in metrics:
            if metric.inputs:
                metric_inputs = torch.cat(metric.inputs).to(self.device)
                metric_targets = torch.cat(metric.targets).to(self.device)
                self.inputs.append(metric_inputs)
                self.targets.append(metric_targets)
                if self.use_weights:
                    metric_weights = torch.cat(metric.weights).to(self.device)
                    self.weights.append(metric_weights)
        return self

    @torch.inference_mode()
    def _prepare_for_merge_state(self: TBinaryRecallAtFixedPrecision) -> None:
        if self.inputs and self.targets:
            self.inputs = [torch.cat(self.inputs)]
            self.targets = [torch.cat(self.targets)]
            if self.use_weights:
                self.weights = [torch.cat(self.weights)]


class MultilabelRecallAtFixedPrecision(
    Metric[tuple[list[torch.Tensor], list[torch.Tensor]]]
):
    """
    Returns the highest possible recall value given the minimum precision
    for each label and their corresponding thresholds for multi-label
    classification tasks. The maximum recall computation for each label is
    equivalent to _binary_recall_at_fixed_precision_compute in BinaryRecallAtFixedPrecision.

    Its functional version is :func:`torcheval.metrics.functional.multilabel_recall_at_fixed_precision`.
    See also :class:`BinaryRecallAtFixedPrecision <BinaryRecallAtFixedPrecision>`

    Args:
        num_labels (int): Number of labels
        min_precision (float): Minimum precision threshold

    Examples::

        >>> import torch
        >>> from torcheval.metrics import MultilabelRecallAtFixedPrecision
        >>> metric = MultilabelRecallAtFixedPrecision(num_labels=3, min_precision=0.5)
        >>> input = torch.tensor([[0.75, 0.05, 0.35], [0.45, 0.75, 0.05], [0.05, 0.55, 0.75], [0.05, 0.65, 0.05]])
        >>> target = torch.tensor([[1, 0, 1], [0, 0, 0], [0, 1, 1], [1, 1, 1]])
        >>> metric.update(input, target)
        >>> metric.compute()
        ([torch.tensor(1.0), torch.tensor(1.0), torch.tensor(1.0)],
        [torch.tensor(0.05), torch.tensor(0.55), torch.tensor(0.05)])
    """

    def __init__(
        self: TMultilabelRecallAtFixedPrecision,
        *,
        num_labels: int,
        min_precision: float,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(device=device)
        self.num_labels = num_labels
        self.min_precision = min_precision
        self._add_state("inputs", [])
        self._add_state("targets", [])

    @torch.inference_mode()
    # pyre-ignore[14]: inconsistent override on *_:Any, **__:Any
    def update(
        self: TMultilabelRecallAtFixedPrecision,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> TMultilabelRecallAtFixedPrecision:
        input = input.to(self.device)
        target = target.to(self.device)

        _multilabel_recall_at_fixed_precision_update_input_check(
            input, target, self.num_labels, self.min_precision
        )
        self.inputs.append(input)
        self.targets.append(target)
        return self

    @torch.inference_mode()
    def compute(
        self: TMultilabelRecallAtFixedPrecision,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        return _multilabel_recall_at_fixed_precision_compute(
            torch.cat(self.inputs),
            torch.cat(self.targets),
            self.num_labels,
            self.min_precision,
        )

    @torch.inference_mode()
    def merge_state(
        self: TMultilabelRecallAtFixedPrecision,
        metrics: Iterable[TMultilabelRecallAtFixedPrecision],
    ) -> TMultilabelRecallAtFixedPrecision:
        for metric in metrics:
            if metric.inputs:
                metric_inputs = torch.cat(metric.inputs).to(self.device)
                metric_targets = torch.cat(metric.targets).to(self.device)
                self.inputs.append(metric_inputs)
                self.targets.append(metric_targets)
        return self

    @torch.inference_mode()
    def _prepare_for_merge_state(self: TMultilabelRecallAtFixedPrecision) -> None:
        if self.inputs and self.targets:
            self.inputs = [torch.cat(self.inputs)]
            self.targets = [torch.cat(self.targets)]
