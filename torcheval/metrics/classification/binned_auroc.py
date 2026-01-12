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
from torcheval.metrics.functional.classification.binned_auroc import (
    _binary_binned_auroc_compute,
    _binary_binned_auroc_param_check,
    _binary_binned_auroc_update_input_check,
    _multiclass_binned_auroc_compute,
    _multiclass_binned_auroc_param_check,
    _multiclass_binned_auroc_update_input_check,
    DEFAULT_NUM_THRESHOLD,
)
from torcheval.metrics.functional.classification.binned_precision_recall_curve import (
    _create_threshold_tensor,
)
from torcheval.metrics.metric import Metric

TBinaryBinnedAUROC = TypeVar("TBinaryBinnedAUROC")
TMulticlassBinnedAUROC = TypeVar("TMulticlassBinnedAUROC")


class BinaryBinnedAUROC(Metric[tuple[torch.Tensor, torch.Tensor]]):
    """
    Compute AUROC, which is the area under the ROC Curve, for binary classification.
    Its functional version is :func:`torcheval.metrics.functional.binary_binned_auroc`.

    Args:
        num_tasks (int):  Number of tasks that need binary_binned_auroc calculation. Default value
                    is 1. binary_binned_auroc for each task will be calculated independently.
        threshold: A integer representing number of bins, a list of thresholds, or a tensor of thresholds.
    See also :class:`MulticlassBinnedAUROC <MulticlassBinnedAUROC>`


    Examples::

        >>> import torch
        >>> from torcheval.metrics import BinaryBinnedAUROC
        >>> input = torch.tensor([0.1, 0.5, 0.7, 0.8])
        >>> target = torch.tensor([1, 0, 1, 1])
        >>> metric = BinaryBinnedAUROC(threshold=5)
        >>> metric.update(input, target)
        >>> metric.compute()
        (tensor([0.5000]),
        tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000])
        )
        >>> input = torch.tensor([1, 1, 1, 0])
        >>> target = torch.tensor([1, 1, 1, 0])
        >>> metric.update(input, target)
        >>> metric.compute()
        (tensor([1.0]),
        tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000])
        )
        >>> metric = BinaryBinnedAUROC(num_tasks=2, threshold=5)
        >>> input = torch.tensor([[1, 1, 1, 0], [0.1, 0.5, 0.7, 0.8]])
        >>> target = torch.tensor([[1, 0, 1, 0], [1, 0, 1, 1]])
        >>> metric.update(input, target)
        >>> metric.compute()
        (tensor([0.7500, 0.5000]),
        tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000]
        )
        )
    """

    def __init__(
        self: TBinaryBinnedAUROC,
        *,
        num_tasks: int = 1,
        threshold: int | list[float] | torch.Tensor = DEFAULT_NUM_THRESHOLD,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(device=device)
        # TODO: @ningli move `_create_threshold_tensor()` to utils
        threshold = _create_threshold_tensor(
            threshold,
            self.device,
        )
        _binary_binned_auroc_param_check(num_tasks, threshold)
        self.num_tasks = num_tasks
        self.threshold = threshold
        self._add_state("inputs", [])
        self._add_state("targets", [])

    @torch.inference_mode()
    # pyre-ignore[14]: inconsistent override on *_:Any, **__:Any
    def update(
        self: TBinaryBinnedAUROC,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> TBinaryBinnedAUROC:
        """
        Update states with the ground truth labels and predictions.

        Args:
            input (Tensor): Tensor of label predictions
                It should be predicted label, probabilities or logits with shape of (num_tasks, n_sample) or (n_sample, ).
            target (Tensor): Tensor of ground truth labels with shape of (num_tasks, n_sample) or (n_sample, ).
        """
        input = input.to(self.device)
        target = target.to(self.device)

        _binary_binned_auroc_update_input_check(
            input, target, self.num_tasks, self.threshold
        )
        self.inputs.append(input)
        self.targets.append(target)
        return self

    @torch.inference_mode()
    def compute(
        self: TBinaryBinnedAUROC,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return Binned_AUROC.  If no ``update()`` calls are made before
        ``compute()`` is called, return an empty tensor.

        Returns:
            Tuple:
                - Binned_AUROC (Tensor): The return value of Binned_AUROC for each task (num_tasks,).
                - threshold (Tensor): Tensor of threshold. Its shape is (n_thresholds, ).
        """
        return _binary_binned_auroc_compute(
            torch.cat(self.inputs, -1), torch.cat(self.targets, -1), self.threshold
        )

    @torch.inference_mode()
    def merge_state(
        self: TBinaryBinnedAUROC, metrics: Iterable[TBinaryBinnedAUROC]
    ) -> TBinaryBinnedAUROC:
        for metric in metrics:
            if metric.inputs:
                metric_inputs = torch.cat(metric.inputs, -1).to(self.device)
                metric_targets = torch.cat(metric.targets, -1).to(self.device)
                self.inputs.append(metric_inputs)
                self.targets.append(metric_targets)
        return self

    @torch.inference_mode()
    def _prepare_for_merge_state(self: TBinaryBinnedAUROC) -> None:
        if self.inputs and self.targets:
            self.inputs = [torch.cat(self.inputs, -1)]
            self.targets = [torch.cat(self.targets, -1)]


class MulticlassBinnedAUROC(Metric[tuple[torch.Tensor, torch.Tensor]]):
    """
    Compute AUROC, which is the area under the ROC Curve, for multiclass classification.
    Its functional version is :func:`torcheval.metrics.functional.multiclass_binned_auroc`.
    See also :class:`BinaryBinnedAUROC <BinaryBinnedAUROC>`

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
        >>> from torcheval.metrics import MulticlassBinnedAUROC
        >>> metric = MulticlassBinnedAUROC(num_classes=4, threshold=5)
        >>> input = torch.tensor([[0.1, 0.1, 0.1, 0.1], [0.5, 0.5, 0.5, 0.5], [0.7, 0.7, 0.7, 0.7], [0.8, 0.8, 0.8, 0.8]])
        >>> target = torch.tensor([0, 1, 2, 3])
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor(0.5000)

        >>> metric = MulticlassBinnedAUROC(num_classes=4, threshold=5, average=None)
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor([0.5000, 0.5000, 0.5000, 0.5000])
    """

    def __init__(
        self: TMulticlassBinnedAUROC,
        *,
        num_classes: int,
        threshold: int | list[float] | torch.Tensor = 200,
        average: str | None = "macro",
        device: torch.device | None = None,
    ) -> None:
        super().__init__(device=device)
        # TODO: @ningli move `_create_threshold_tensor()` to utils
        threshold = _create_threshold_tensor(
            threshold,
            self.device,
        )
        _multiclass_binned_auroc_param_check(num_classes, threshold, average)
        self.num_classes = num_classes
        self.threshold = threshold
        self.average = average
        self._add_state("inputs", [])
        self._add_state("targets", [])

    @torch.inference_mode()
    # pyre-ignore[14]: inconsistent override on *_:Any, **__:Any
    def update(
        self: TMulticlassBinnedAUROC,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> TMulticlassBinnedAUROC:
        """
        Update states with the ground truth labels and predictions.

        Args:
            input (Tensor): Tensor of label predictions
                It should be probabilities or logits with shape of (n_sample, n_class).
            target (Tensor): Tensor of ground truth labels with shape of (n_samples, ).
        """
        input = input.to(self.device)
        target = target.to(self.device)

        _multiclass_binned_auroc_update_input_check(input, target, self.num_classes)
        self.inputs.append(input)
        self.targets.append(target)
        return self

    @torch.inference_mode()
    def compute(
        self: TMulticlassBinnedAUROC,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _multiclass_binned_auroc_compute(
            torch.cat(self.inputs),
            torch.cat(self.targets),
            self.num_classes,
            self.threshold,
            self.average,
        )

    @torch.inference_mode()
    def merge_state(
        self: TMulticlassBinnedAUROC, metrics: Iterable[TMulticlassBinnedAUROC]
    ) -> TMulticlassBinnedAUROC:
        for metric in metrics:
            if metric.inputs:
                metric_inputs = torch.cat(metric.inputs).to(self.device)
                metric_targets = torch.cat(metric.targets).to(self.device)
                self.inputs.append(metric_inputs)
                self.targets.append(metric_targets)
        return self

    @torch.inference_mode()
    def _prepare_for_merge_state(self: TMulticlassBinnedAUROC) -> None:
        if self.inputs and self.targets:
            self.inputs = [torch.cat(self.inputs)]
            self.targets = [torch.cat(self.targets)]
