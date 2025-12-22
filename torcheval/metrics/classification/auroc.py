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

from torcheval.metrics.functional.classification.auroc import (
    _binary_auroc_compute,
    _binary_auroc_update_input_check,
    _multiclass_auroc_compute,
    _multiclass_auroc_param_check,
    _multiclass_auroc_update_input_check,
)
from torcheval.metrics.metric import Metric
from torcheval.utils.device import largest_float

try:
    import fbgemm_gpu.metrics  # noqa

    has_fbgemm = True
except ImportError:
    has_fbgemm = False


TAUROC = TypeVar("TAUROC")
TMulticlasslAUROC = TypeVar("TMulticlassAUROC")


class BinaryAUROC(Metric[torch.Tensor]):
    """
    Compute AUROC, which is the area under the ROC Curve, for binary classification.
    AUROC is defined as the area under the Receiver Operating Curve, a plot with x=false positive rate y=true positive rate.
    The points on the curve are sampled from the data given and the area is computed using the trapezoid method.

    Multiple tasks are supported for Binary AUROC. A two-dimensional vector can given for the predicted values (inputs) and targets. This gives equivalent results to having one BinaryAUROC object for each row.

    Its functional version is :func:`torcheval.metrics.functional.binary_auroc`.
    See also :class:`MulticlassAUROC <MulticlassAUROC>`

    Examples::

        >>> import torch
        >>> from torcheval.metrics import BinaryAUROC
        >>> metric = BinaryAUROC()
        >>> input = torch.tensor([0.1, 0.5, 0.7, 0.8])
        >>> target = torch.tensor([1, 0, 1, 1])
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor([0.6667])
        >>> input = torch.tensor([1, 1, 1, 0])
        >>> target = torch.tensor([1, 1, 1, 0])
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor([1.0])
        >>> metric = BinaryAUROC(num_tasks=2)
        >>> input = torch.tensor([[1, 1, 1, 0], [0.1, 0.5, 0.7, 0.8]])
        >>> target = torch.tensor([[1, 0, 1, 0], [1, 0, 1, 1]])
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor([0.7500, 0.6667])
    """

    def __init__(
        self: TAUROC,
        *,
        num_tasks: int = 1,
        device: torch.device | None = None,
        use_fbgemm: bool | None = False,
    ) -> None:
        super().__init__(device=device)
        if num_tasks < 1:
            raise ValueError(
                "`num_tasks` value should be greater than or equal to 1, but received {num_tasks}. "
            )
        if not has_fbgemm and use_fbgemm:
            raise ValueError(
                "`use_fbgemm` is enabled but `fbgemm_gpu` is not found. Please "
                "install `fbgemm_gpu` to use this option."
            )

        self.num_tasks = num_tasks
        self._add_state("inputs", [])
        self._add_state("targets", [])
        self._add_state("weights", [])
        self.use_fbgemm = use_fbgemm

    @torch.inference_mode()
    # pyre-ignore[14]: inconsistent override on *_:Any, **__:Any
    def update(
        self: TAUROC,
        input: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor | None = None,
    ) -> TAUROC:
        """
        Update states with the ground truth labels and predictions.

        Args:
            input (Tensor): Tensor of label predictions
                It should be predicted label, probabilities or logits with shape of (num_tasks, n_sample) or (n_sample, ).
            target (Tensor): Tensor of ground truth labels with shape of (num_tasks, n_sample) or (n_sample, ).
            weight (Tensor): Optional. A manual rescaling weight to match input tensor shape (num_tasks, num_samples) or (n_sample, ).
        """
        input = input.to(self.device)
        target = target.to(self.device)

        if weight is None:
            weight = torch.ones_like(input, dtype=largest_float(self.device))
        _binary_auroc_update_input_check(input, target, self.num_tasks, weight)
        self.inputs.append(input)
        self.targets.append(target)
        self.weights.append(weight)
        return self

    @torch.inference_mode()
    def compute(
        self: TAUROC,
    ) -> torch.Tensor:
        """
        Return AUROC.  If no ``update()`` calls are made before
        ``compute()`` is called, a runtime error is thrown as
        computing AUROC over 0 samples is undefined.

        Returns:
            Tensor: The return value of AUROC for each task (num_tasks,).
        """
        if len(self.inputs) == 0:
            raise AssertionError(
                "No samples to compute auroc with. Did you call update?"
            )

        return _binary_auroc_compute(
            torch.cat(self.inputs, -1),
            torch.cat(self.targets, -1),
            torch.cat(self.weights, -1),
            self.use_fbgemm,
        )

    @torch.inference_mode()
    def merge_state(self: TAUROC, metrics: Iterable[TAUROC]) -> TAUROC:
        for metric in metrics:
            if metric.inputs:
                metric_inputs = torch.cat(metric.inputs, -1).to(self.device)
                metric_targets = torch.cat(metric.targets, -1).to(self.device)
                metric_weights = torch.cat(metric.weights, -1).to(self.device)
                self.inputs.append(metric_inputs)
                self.targets.append(metric_targets)
                self.weights.append(metric_weights)
        return self

    @torch.inference_mode()
    def _prepare_for_merge_state(self: TAUROC) -> None:
        if self.inputs and self.targets:
            self.inputs = [torch.cat(self.inputs, -1)]
            self.targets = [torch.cat(self.targets, -1)]
            self.weights = [torch.cat(self.weights, -1)]


class MulticlassAUROC(Metric[torch.Tensor]):
    """
    Compute AUROC, which is the area under the ROC Curve, for multiclass classification in a one vs rest fashion.
    One vs. rest Multiclass AUROC is equivalent to running a BinaryAUROC with `num_classes` tasks where

    1. The `input` is transposed
    2. The `target` is translated from a 1 dimensional tensor of the correct classes to a 2 dimensional tensor where each row is a list containing which examples belong to that class.

    See examples below for more details on the connection between Multiclass and Binary AUROC.

    The functional version of this metric is :func:`torcheval.metrics.functional.multiclass_auroc`.
    See also :class:`BinaryAUROC <BinaryAUROC>`

    Args:
        num_classes (int): Number of classes.
        average (str, optional):

          - ``'macro'`` [default]:
                Calculate metrics for each class separately, and return their unweighted mean.
          - ``None``:
                Calculate the metric for each class separately, and return
                the metric for every class.

    Examples::

        >>> import torch
        >>> from torcheval.metrics import MulticlassAUROC
        >>> metric = MulticlassAUROC(num_classes=4)
        >>> input = torch.tensor([[0.1, 0.1, 0.1, 0.1], [0.5, 0.5, 0.5, 0.5], [0.7, 0.7, 0.7, 0.7], [0.8, 0.8, 0.8, 0.8]])
        >>> target = torch.tensor([0, 1, 2, 3])
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor(0.5000)

        >>> metric = MulticlassAUROC(num_classes=3, average=None)
        >>> input = torch.tensor([[0.1, 0, 0], [0, 1, 0], [0.1, 0.2, 0.7], [0, 0, 1]])
        >>> target = torch.tensor([0, 1, 2, 2])
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor([0.8333, 1.0000, 1.0000])

        the above is equivalent to
        >>> from torcheval.metrics import BinaryAUROC
        >>> metric = BinaryAUROC(num_tasks=3)
        >>> input = torch.tensor([[0.1, 0, 0.1, 0], [0, 1, 0.2, 0], [0, 0, 0.7, 1]])
        >>> target = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1]])
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor([0.8333, 1.0000, 1.0000])
    """

    def __init__(
        self: TMulticlasslAUROC,
        *,
        num_classes: int,
        average: str | None = "macro",
        device: torch.device | None = None,
    ) -> None:
        super().__init__(device=device)
        _multiclass_auroc_param_check(num_classes, average)
        self.num_classes = num_classes
        self.average = average
        self._add_state("inputs", [])
        self._add_state("targets", [])

    @torch.inference_mode()
    # pyre-ignore[14]: inconsistent override on *_:Any, **__:Any
    def update(
        self: TMulticlasslAUROC,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> TMulticlasslAUROC:
        """
        Update states with the ground truth labels and predictions.

        Args:
            input (Tensor): Tensor of label predictions
                It should be probabilities or logits with shape of (n_sample, n_class).
            target (Tensor): Tensor of ground truth labels with shape of (n_samples, ).
        """
        input = input.to(self.device)
        target = target.to(self.device)

        _multiclass_auroc_update_input_check(input, target, self.num_classes)
        self.inputs.append(input)
        self.targets.append(target)
        return self

    @torch.inference_mode()
    def compute(
        self: TMulticlasslAUROC,
    ) -> torch.Tensor:
        if len(self.inputs) == 0:
            raise AssertionError(
                "No samples to compute auroc with. Did you call update?"
            )

        return _multiclass_auroc_compute(
            torch.cat(self.inputs),
            torch.cat(self.targets),
            self.num_classes,
            self.average,
        )

    @torch.inference_mode()
    def merge_state(
        self: TMulticlasslAUROC, metrics: Iterable[TMulticlasslAUROC]
    ) -> TMulticlasslAUROC:
        for metric in metrics:
            if metric.inputs:
                metric_inputs = torch.cat(metric.inputs).to(self.device)
                metric_targets = torch.cat(metric.targets).to(self.device)
                self.inputs.append(metric_inputs)
                self.targets.append(metric_targets)
        return self

    @torch.inference_mode()
    def _prepare_for_merge_state(self: TMulticlasslAUROC) -> None:
        if self.inputs and self.targets:
            self.inputs = [torch.cat(self.inputs)]
            self.targets = [torch.cat(self.targets)]
