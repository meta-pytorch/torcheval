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
from torcheval.metrics.functional.classification.precision_recall_curve import (
    _binary_precision_recall_curve_compute,
    _binary_precision_recall_curve_update,
    _multiclass_precision_recall_curve_compute,
    _multiclass_precision_recall_curve_update,
    _multilabel_precision_recall_curve_compute,
    _multilabel_precision_recall_curve_update,
)
from torcheval.metrics.metric import Metric

"""
This file contains BinaryPrecisionRecallCurve, MulticlassPrecisionRecallCurve and MultilabelPrecisionRecallCurve classes.
"""

TBinaryPrecisionRecallCurve = TypeVar("TBinaryPrecisionRecallCurve")
TMulticlassPrecisionRecallCurve = TypeVar("TMulticlassPrecisionRecallCurve")
TMultilabelPrecisionRecallCurve = TypeVar("TMultilabelPrecisionRecallCurve")


class BinaryPrecisionRecallCurve(
    Metric[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
):
    """
    Returns precision-recall pairs and their corresponding thresholds for
    binary classification tasks. If a class is missing from the target tensor,
    its recall values are set to 1.0.

    Its functional version is :func:`torcheval.metrics.functional.binary_precision_recall_curve`.
    See also :class:`MulticlassPrecisionRecallCurve <MulticlassPrecisionRecallCurve>`, :class:`MultilabelPrecisionRecallCurve <MultilabelPrecisionRecallCurve>`

        >>> import torch
        >>> from torcheval.metrics import BinaryPrecisionRecallCurve
        >>> metric = BinaryPrecisionRecallCurve()
        >>> input = torch.tensor([0.1, 0.5, 0.7, 0.8])
        >>> target = torch.tensor([0, 0, 1, 1])
        >>> metric.update(input, target)
        >>> metric.compute()
        (tensor([1., 1., 1.]),
        tensor([1.0000, 0.5000, 0.0000]),
        tensor([0.7000, 0.8000]))
    """

    def __init__(
        self: TBinaryPrecisionRecallCurve,
        *,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(device=device)
        self._add_state("inputs", [])
        self._add_state("targets", [])

    @torch.inference_mode()
    # pyre-ignore[14]: inconsistent override on *_:Any, **__:Any
    def update(
        self: TBinaryPrecisionRecallCurve,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> TBinaryPrecisionRecallCurve:
        """
        Update states with the ground truth labels and predictions.

        Args:
            input (Tensor): Tensor of label predictions
                It should be probabilities or logits with shape of (n_sample, ).
            target (Tensor): Tensor of ground truth labels with shape of (n_samples, ).
        """
        input = input.to(self.device)
        target = target.to(self.device)

        _binary_precision_recall_curve_update(
            input,
            target,
        )
        self.inputs.append(input)
        self.targets.append(target)
        return self

    @torch.inference_mode()
    def compute(
        self: TBinaryPrecisionRecallCurve,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            Tuple:
                - precision (Tensor): Tensor of precision result. Its shape is (n_thresholds + 1, )
                - recall (Tensor): Tensor of recall result. Its shape is (n_thresholds + 1, )
                - thresholds (Tensor): Tensor of threshold. Its shape is (n_thresholds, )
        """
        return _binary_precision_recall_curve_compute(
            torch.cat(self.inputs), torch.cat(self.targets)
        )

    @torch.inference_mode()
    def merge_state(
        self: TBinaryPrecisionRecallCurve,
        metrics: Iterable[TBinaryPrecisionRecallCurve],
    ) -> TBinaryPrecisionRecallCurve:
        for metric in metrics:
            if metric.inputs:
                metric_inputs = torch.cat(metric.inputs).to(self.device)
                metric_targets = torch.cat(metric.targets).to(self.device)
                self.inputs.append(metric_inputs)
                self.targets.append(metric_targets)
        return self

    @torch.inference_mode()
    def _prepare_for_merge_state(self: TBinaryPrecisionRecallCurve) -> None:
        if self.inputs and self.targets:
            self.inputs = [torch.cat(self.inputs)]
            self.targets = [torch.cat(self.targets)]


class MulticlassPrecisionRecallCurve(
    Metric[tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]]
):
    """
    Returns precision-recall pairs and their corresponding thresholds for
    multi-class classification tasks. If a class is missing from the target
    tensor, its recall values are set to 1.0.

    Its class version is :func:`torcheval.metrics.functional.multiclass_precision_recall_curve`.
    See also :class:`BinaryPrecisionRecallCurve <BinaryPrecisionRecallCurve>`, :class:`MultilabelPrecisionRecallCurve <MultilabelPrecisionRecallCurve>`

    Args:
        num_classes (int, Optional):
            Number of classes. Set to the second dimension of the input if num_classes is None.

    Examples::

        >>> import torch
        >>> from torcheval.metrics import MulticlassPrecisionRecallCurve
        >>> metric = MulticlassPrecisionRecallCurve(num_classes=4)
        >>> input = torch.tensor([[0.1, 0.1, 0.1, 0.1], [0.5, 0.5, 0.5, 0.5], [0.7, 0.7, 0.7, 0.7], [0.8, 0.8, 0.8, 0.8]])
        >>> target = torch.tensor([0, 1, 2, 3])
        >>> metric.update(input, target)
        >>> metric.compute()
        ([tensor([0.2500, 0.0000, 0.0000, 0.0000, 1.0000]),
        tensor([0.3333, 0.0000, 0.0000, 1.0000]),
        tensor([0.5000, 0.0000, 1.0000]),
        tensor([1., 1.])],
        [tensor([1., 0., 0., 0., 0.]),
        tensor([1., 0., 0., 0.]),
        tensor([1., 0., 0.]),
        tensor([1., 0.])],
        [tensor([0.1000, 0.5000, 0.7000, 0.8000]),
        tensor([0.5000, 0.7000, 0.8000]),
        tensor([0.7000, 0.8000]),
        tensor([0.8000])])
    """

    def __init__(
        self: TMulticlassPrecisionRecallCurve,
        *,
        num_classes: int | None = None,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(device=device)
        self.num_classes = num_classes
        self._add_state("inputs", [])
        self._add_state("targets", [])

    @torch.inference_mode()
    # pyre-ignore[14]: inconsistent override on *_:Any, **__:Any
    def update(
        self: TMulticlassPrecisionRecallCurve,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> TMulticlassPrecisionRecallCurve:
        """
        Update states with the ground truth labels and predictions.

        Args:
            input (Tensor): Tensor of label predictions
                It should be probabilities or logits with shape of (n_sample, n_class).
            target (Tensor): Tensor of ground truth labels with shape of (n_samples, ).
        """
        input = input.to(self.device)
        target = target.to(self.device)

        _multiclass_precision_recall_curve_update(
            input,
            target,
            self.num_classes,
        )
        self.inputs.append(input)
        self.targets.append(target)
        return self

    @torch.inference_mode()
    def compute(
        self: TMulticlassPrecisionRecallCurve,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """
        Returns:

            - precision: List of precision result. Each index indicates the result of a class.
            - recall: List of recall result. Each index indicates the result of a class.
            - thresholds: List of threshold. Each index indicates the result of a class.

        """
        return _multiclass_precision_recall_curve_compute(
            torch.cat(self.inputs), torch.cat(self.targets), self.num_classes
        )

    @torch.inference_mode()
    def merge_state(
        self: TMulticlassPrecisionRecallCurve,
        metrics: Iterable[TMulticlassPrecisionRecallCurve],
    ) -> TMulticlassPrecisionRecallCurve:
        for metric in metrics:
            if metric.inputs:
                metric_inputs = torch.cat(metric.inputs).to(self.device)
                metric_targets = torch.cat(metric.targets).to(self.device)
                self.inputs.append(metric_inputs)
                self.targets.append(metric_targets)
        return self

    @torch.inference_mode()
    def _prepare_for_merge_state(self: TMulticlassPrecisionRecallCurve) -> None:
        if self.inputs and self.targets:
            self.inputs = [torch.cat(self.inputs)]
            self.targets = [torch.cat(self.targets)]


class MultilabelPrecisionRecallCurve(
    Metric[tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]]
):
    """
    Returns precision-recall pairs and their corresponding thresholds for
    multi-label classification tasks. If there are no samples for a label
    in the target tensor, its recall values are set to 1.0.

    Its class version is :func:`torcheval.metrics.functional.multilabel_precision_recall_curve`.
    See also :class:`BinaryPrecisionRecallCurve <BinaryPrecisionRecallCurve>`, :class:`MulticlassPrecisionRecallCurve <MulticlassPrecisionRecallCurve>`

    Args:
        num_labels (int): Number of labels.

    Examples::

        >>> import torch
        >>> from torcheval.metrics import MultilabelPrecisionRecallCurve
        >>> metric = MultilabelPrecisionRecallCurve(num_labels=3)
        >>> input = torch.tensor([[0.75, 0.05, 0.35], [0.45, 0.75, 0.05], [0.05, 0.55, 0.75], [0.05, 0.65, 0.05]])
        >>> target = torch.tensor([[1, 0, 1], [0, 0, 0], [0, 1, 1], [1, 1, 1]])
        >>> metric.update(input, target)
        >>> metric.compute()
        ([tensor([0.5, 0.5, 1.0, 1.0]),
        tensor([0.5, 0.66666667, 0.5, 0.0, 1.0]),
        tensor([0.75, 1.0, 1.0, 1.0])],
        [tensor([1.0, 0.5, 0.5, 0.0]),
        tensor([1.0, 1.0, 0.5, 0.0, 0.0]),
        tensor([1.0, 0.66666667, 0.33333333, 0.0])],
        [tensor([0.05, 0.45, 0.75]),
        tensor([0.05, 0.55, 0.65, 0.75]),
        tensor([0.05, 0.35, 0.75])])
    """

    def __init__(
        self: TMultilabelPrecisionRecallCurve,
        *,
        num_labels: int,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(device=device)
        self.num_labels = num_labels
        self._add_state("inputs", [])
        self._add_state("targets", [])

    @torch.inference_mode()
    # pyre-ignore[14]: inconsistent override on *_:Any, **__:Any
    def update(
        self: TMultilabelPrecisionRecallCurve,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> TMultilabelPrecisionRecallCurve:
        """
        Update states with the ground truth labels and predictions.

        Args:
            input (Tensor): Tensor of label predictions
                It should be the probabilites with shape of (n_samples, n_labels).
            target (Tensor):Tensor of ground truth labels with shape of (n_samples, n_labels).
        """
        input = input.to(self.device)
        target = target.to(self.device)

        _multilabel_precision_recall_curve_update(
            input,
            target,
            self.num_labels,
        )
        self.inputs.append(input)
        self.targets.append(target)
        return self

    @torch.inference_mode()
    def compute(
        self: TMultilabelPrecisionRecallCurve,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """
        Returns:

            - precision: List of precision result. Each index indicates the result of a label.
            - recall: List of recall result. Each index indicates the result of a label.
            - thresholds: List of threshold. Each index indicates the result of a label.
        """
        return _multilabel_precision_recall_curve_compute(
            torch.cat(self.inputs), torch.cat(self.targets), self.num_labels
        )

    @torch.inference_mode()
    def merge_state(
        self: TMultilabelPrecisionRecallCurve,
        metrics: Iterable[TMultilabelPrecisionRecallCurve],
    ) -> TMultilabelPrecisionRecallCurve:
        for metric in metrics:
            if metric.inputs:
                metric_inputs = torch.cat(metric.inputs).to(self.device)
                metric_targets = torch.cat(metric.targets).to(self.device)
                self.inputs.append(metric_inputs)
                self.targets.append(metric_targets)
        return self

    @torch.inference_mode()
    def _prepare_for_merge_state(self: TMulticlassPrecisionRecallCurve) -> None:
        if self.inputs and self.targets:
            self.inputs = [torch.cat(self.inputs)]
            self.targets = [torch.cat(self.targets)]
