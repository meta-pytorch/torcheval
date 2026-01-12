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
from torcheval.metrics.functional.classification.auprc import (
    _binary_auprc_compute,
    _binary_auprc_update_input_check,
    _multiclass_auprc_compute,
    _multiclass_auprc_param_check,
    _multiclass_auprc_update_input_check,
    _multilabel_auprc_compute,
    _multilabel_auprc_param_check,
    _multilabel_auprc_update_input_check,
)
from torcheval.metrics.metric import Metric


TBinaryAUPRC = TypeVar("TBinaryAUPRC", bound=Metric[torch.Tensor])
TMulticlassAUPRC = TypeVar("TMulticlassAUPRC", bound=Metric[torch.Tensor])
TMultilabelAUPRC = TypeVar("TMultilabelAUPRC", bound=Metric[torch.Tensor])


class BinaryAUPRC(Metric[torch.Tensor]):
    """
    Compute AUPRC, also called Average Precision, which is the area under the Precision-Recall Curve, for binary classification.

    Precision is defined as :math:`\\frac{T_p}{T_p+F_p}`, it is the probability that a positive prediction from the model is a true positive.
    Recall is defined as :math:`\\frac{T_p}{T_p+F_n}`, it is the probability that a true positive is predicted to be positive by the model.

    The precision-recall curve plots the recall on the x axis and the precision on the y axis, both of which are bounded between 0 and 1.
    This function returns the area under that graph. If the area is near one, the model supports a threshold which correctly identifies
    a high percentage of true positives while also rejecting enough false examples so that most of the true predictions are true positives.

    Binary auprc supports multiple tasks, if the input and target tensors are 2 dimensional each row will be evaluated as if it were an independent
    instance of binary auprc.

    The functional version of this metric is :func:`torcheval.metrics.functional.binary_auprc`.
    See also :class:`MulticlassAUPRC <MulticlassAUPRC>`, :class:`MultilabelAUPRC <MultilabelAUPRC>`

    Args:
        num_tasks (int): Number of tasks that need BinaryAUPRC calculation. Default value
                    is 1. Binary AUPRC for each task will be calculated independently. Results are
                    equivalent to running Binary AUPRC calculation for each row.

    Examples::

        >>> import torch
        >>> from torcheval import BinaryAUPRC
        >>> metric = BinaryAUPRC()
        >>> input = torch.tensor([0.1, 0.5, 0.7, 0.8])
        >>> target = torch.tensor([1, 0, 1, 1])
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor(0.9167) # scalar returned with 1D input tensors

        # with logits
        >>> metric = BinaryAUPRC()
        >>> input = torch.tensor([[.5, 2]])
        >>> target = torch.tensor([[0, 0]])
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor([-0.])
        >>> input = torch.tensor([[2, 1.5]])
        >>> target = torch.tensor([[1, 0]])
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor([0.5000]) # 1D tensor returned with 2D input tensors

        # multiple tasks
        >>> metric = BinaryAUPRC(num_tasks=3)
        >>> input = torch.tensor([[0.1, 0, 0.1, 0], [0, 1, 0.2, 0], [0, 0, 0.7, 1]])
        >>> target = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1]])
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor([0.5000, 1.0000, 1.0000])
    """

    def __init__(
        self: TBinaryAUPRC,
        *,
        num_tasks: int = 1,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(device=device)
        if num_tasks < 1:
            raise ValueError(
                "`num_tasks` must be an integer greater than or equal to 1"
            )
        self.num_tasks = num_tasks

        self._add_state("inputs", [])
        self._add_state("targets", [])

    @torch.inference_mode()
    # pyre-ignore[14]: inconsistent override on *_:Any, **__:Any
    def update(
        self: TBinaryAUPRC,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> TBinaryAUPRC:
        """
        Update states with the ground truth labels and predictions.

        Args:
            input (Tensor): Tensor of label predictions
                It should be probabilities or logits with shape of (n_sample, n_class).
            target (Tensor): Tensor of ground truth labels with shape of (n_samples, ).
        """
        input = input.to(self.device)
        target = target.to(self.device)

        _binary_auprc_update_input_check(input, target, self.num_tasks)
        self.inputs.append(input)
        self.targets.append(target)
        return self

    @torch.inference_mode()
    def compute(
        self: TBinaryAUPRC,
    ) -> torch.Tensor:
        return _binary_auprc_compute(
            torch.cat(self.inputs, -1),
            torch.cat(self.targets, -1),
            self.num_tasks,
        )

    @torch.inference_mode()
    def merge_state(
        self: TBinaryAUPRC, metrics: Iterable[TBinaryAUPRC]
    ) -> TBinaryAUPRC:
        for metric in metrics:
            if metric.inputs:
                metric_inputs = torch.cat(metric.inputs, -1).to(self.device)
                metric_targets = torch.cat(metric.targets, -1).to(self.device)
                self.inputs.append(metric_inputs)
                self.targets.append(metric_targets)
        return self

    @torch.inference_mode()
    def _prepare_for_merge_state(self: TBinaryAUPRC) -> None:
        if self.inputs and self.targets:
            self.inputs = [torch.cat(self.inputs, -1)]
            self.targets = [torch.cat(self.targets, -1)]


class MulticlassAUPRC(Metric[torch.Tensor]):
    """
    Compute AUPRC, also called Average Precision, which is the area under the Precision-Recall Curve, for multiclass classification.

    Precision is defined as :math:`\\frac{T_p}{T_p+F_p}`, it is the probability that a positive prediction from the model is a true positive.
    Recall is defined as :math:`\\frac{T_p}{T_p+F_n}`, it is the probability that a true positive is predicted to be positive by the model.

    The precision-recall curve plots the recall on the x axis and the precision on the y axis, both of which are bounded between 0 and 1.
    This function returns the area under that graph. If the area is near one, the model supports a threshold which correctly identifies
    a high percentage of true positives while also rejecting enough false examples so that most of the true predictions are true positives.

    In the multiclass version of auprc, the target tensor is a 1 dimensional and contains an integer entry representing the class for each example
    in the input tensor. Each class is considered independently in a one-vs-all fashion, examples for that class are labeled condition true and all other
    classes are considered condition false.

    The results of N class multiclass auprc without an average is equivalent to binary auprc with N tasks if:

    1. the input is transposed, in binary classification examples are associated with columns, whereas they are associated with rows in multiclass classification.
    2. the `target` is translated from the form [1,0,1] to the form [[0,1,0], [1,0,1]]

    See examples below for more details on the connection between Multiclass and Binary AUPRC.

    The functional version of this metric is :func:`torcheval.metrics.functional.multiclass_auprc`.
    See also :class:`BinaryAUPRC <BinaryAUPRC>`, :class:`MultilabelAUPRC <MultilabelAUPRC>`

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
        >>> from torcheval.metrics import MulticlassAUPRC
        >>> metric = MulticlassAUPRC(num_classes=3)
        >>> input = torch.tensor([[0.1, 0.1, 0.1], [0.5, 0.5, 0.5], [0.7, 0.7, 0.7], [0.8, 0.8, 0.8]])
        >>> target = torch.tensor([0, 2, 1, 1])
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor(0.5278)

        >>> metric = MulticlassAUPRC(num_classes=3)
        >>> input = torch.tensor([[0.5, .2, 3], [2, 1, 6]])
        >>> target = torch.tensor([0, 2])
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor(0.5000)
        >>> input = torch.tensor([[5, 3, 2], [.2, 2, 3], [3, 3, 3]])
        >>> target = torch.tensor([2, 2, 1])
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor(0.4833)

        Connection to BinaryAUPRC
        >>> metric = MulticlassAUPRC(num_classes=3, average=None)
        >>> input = torch.tensor([[0.1, 0, 0], [0, 1, 0], [0.1, 0.2, 0.7], [0, 0, 1]])
        >>> target = torch.tensor([0, 1, 2, 2])
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor([0.5000, 1.0000, 1.0000])

        the above is equivalent to
        >>> from torcheval.metrics import BinaryAUPRC
        >>> metric = BinaryAUPRC(num_tasks=3)
        >>> input = torch.tensor([[0.1, 0, 0.1, 0], [0, 1, 0.2, 0], [0, 0, 0.7, 1]])
        >>> target = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1]])
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor([0.5000, 1.0000, 1.0000])
    """

    def __init__(
        self: TMulticlassAUPRC,
        *,
        num_classes: int,
        average: str | None = "macro",
        device: torch.device | None = None,
    ) -> None:
        super().__init__(device=device)
        _multiclass_auprc_param_check(num_classes, average)
        self.num_classes = num_classes
        self.average = average
        self._add_state("inputs", [])
        self._add_state("targets", [])

    @torch.inference_mode()
    # pyre-ignore[14]: inconsistent override on *_:Any, **__:Any
    def update(
        self: TMulticlassAUPRC,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> TMulticlassAUPRC:
        """
        Update states with the ground truth labels and predictions.

        Args:
            input (Tensor): Tensor of label predictions
                It should be probabilities or logits with shape of (n_sample, n_class).
            target (Tensor): Tensor of ground truth labels with shape of (n_samples, ).
        """
        input = input.to(self.device)
        target = target.to(self.device)

        _multiclass_auprc_update_input_check(input, target, self.num_classes)
        self.inputs.append(input)
        self.targets.append(target)
        return self

    @torch.inference_mode()
    def compute(
        self: TMulticlassAUPRC,
    ) -> torch.Tensor:
        return _multiclass_auprc_compute(
            torch.cat(self.inputs),
            torch.cat(self.targets),
            self.average,
        )

    @torch.inference_mode()
    def merge_state(
        self: TMulticlassAUPRC, metrics: Iterable[TMulticlassAUPRC]
    ) -> TMulticlassAUPRC:
        for metric in metrics:
            if metric.inputs:
                metric_inputs = torch.cat(metric.inputs).to(self.device)
                metric_targets = torch.cat(metric.targets).to(self.device)
                self.inputs.append(metric_inputs)
                self.targets.append(metric_targets)
        return self

    @torch.inference_mode()
    def _prepare_for_merge_state(self: TMulticlassAUPRC) -> None:
        if self.inputs and self.targets:
            self.inputs = [torch.cat(self.inputs)]
            self.targets = [torch.cat(self.targets)]


class MultilabelAUPRC(Metric[torch.Tensor]):
    """
    Compute AUPRC, also called Average Precision, which is the area under the Precision-Recall Curve, for multilabel classification.

    Precision is defined as :math:`\\frac{T_p}{T_p+F_p}`, it is the probability that a positive prediction from the model is a true positive.
    Recall is defined as :math:`\\frac{T_p}{T_p+F_n}`, it is the probability that a true positive is predicted to be positive by the model.

    The precision-recall curve plots the recall on the x axis and the precision on the y axis, both of which are bounded between 0 and 1.
    This function returns the area under that graph. If the area is near one, the model supports a threshold which correctly identifies
    a high percentage of true positives while also rejecting enough false examples so that most of the true predictions are true positives.

    In the multilabel version of AUPRC, the input and target tensors are 2-dimensional. The rows of each tensor are associated with a particular example and the columns are associated with a particular class.

    For the target tensor, the entry of the r'th row and c'th column (r and c are 0-indexed) is 1 if the r'th example belongs to the c'th class, and 0 if not. For the input tensor, the entry in the same position is the output of the classification model prediciting the inclusion of the r'th example in the c'th class.
    Note that in the multilabel setting, multiple labels are allowed to apply to a single sample. This stands in contrast to the multiclass
    sample, in which there may be more than 2 distinct classes but each sample must have exactly one class.

    The results of N label multilabel auprc without an average is equivalent to binary auprc with N tasks if:

    1. the `input` is transposed, in binary labelification examples are associated with columns, whereas they are associated with rows in multilabel classification.
    2. the `target` is transposed for the same reason

    See examples below for more details on the connection between Multilabel and Binary AUPRC.

    The functional version of this metric is :func:`torcheval.metrics.functional.multilabel_auprc`.
    See also :class:`BinaryAUPRC <BinaryAUPRC>`, :class:`MulticlassAUPRC <MulticlassAUPRC>`

    Args:
        num_labels (int): Number of labels.
        average (str, optional):

          - ``'macro'`` [default]:
                Calculate metrics for each label separately, and return their unweighted mean.
          - ``None``:
                Calculate the metric for each label separately, and return
                the metric for every label.

    Examples::

        >>> import torch
        >>> from torcheval.metrics import MultilabelAUPRC
        >>> metric = MultilabelAUPRC(num_labels=3, average=None)
        >>> input = torch.tensor([[0.75, 0.05, 0.35], [0.45, 0.75, 0.05], [0.05, 0.55, 0.75], [0.05, 0.65, 0.05]])
        >>> target = torch.tensor([[1, 0, 1], [0, 0, 0], [0, 1, 1], [1, 1, 1]])
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor([0.7500, 0.5833, 0.9167])

        >>> metric = MultilabelAUPRC(num_labels=3, average='macro')
        >>> input = torch.tensor([[0.75, 0.05, 0.35], [0.05, 0.55, 0.75]])
        >>> target = torch.tensor([[1, 0, 1], [0, 1, 1]])
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor(1.)
        >>> input = torch.tensor([[0.45, 0.75, 0.05], [0.05, 0.65, 0.05]])
        >>> target = torch.tensor([[0, 0, 0], [1, 1, 1]])
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor(0.7500)

        Connection to BinaryAUPRC
        >>> metric = MultilabelAUPRC(num_labels=3, average=None)
        >>> input = torch.tensor([[0.1, 0, 0], [0, 1, 0], [0.1, 0.2, 0.7], [0, 0, 1]])
        >>> target = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]])
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor([0.5000, 1.0000, 1.0000])

        the above is equivalent to
        >>> from torcheval.metrics import BinaryAUPRC
        >>> metric = BinaryAUPRC(num_tasks=3)
        >>> input = torch.tensor([[0.1, 0, 0.1, 0], [0, 1, 0.2, 0], [0, 0, 0.7, 1]])
        >>> target = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1]])
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor([0.5000, 1.0000, 1.0000])
    """

    def __init__(
        self: TMultilabelAUPRC,
        *,
        num_labels: int,
        average: str | None = "macro",
        device: torch.device | None = None,
    ) -> None:
        super().__init__(device=device)
        _multilabel_auprc_param_check(num_labels, average)
        self.num_labels = num_labels
        self.average = average
        self._add_state("inputs", [])
        self._add_state("targets", [])

    @torch.inference_mode()
    # pyre-ignore[14]: inconsistent override on *_:Any, **__:Any
    def update(
        self: TMultilabelAUPRC,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> TMultilabelAUPRC:
        """
        Update states with the ground truth labels and predictions.

        Args:
            input (Tensor): Tensor of label predictions
                It should be probabilities or logits with shape of (n_sample, n_label).
            target (Tensor): Tensor of ground truth labels with shape of (n_samples, n_label).
        """
        input = input.to(self.device)
        target = target.to(self.device)

        _multilabel_auprc_update_input_check(input, target, self.num_labels)
        self.inputs.append(input)
        self.targets.append(target)
        return self

    @torch.inference_mode()
    def compute(
        self: TMultilabelAUPRC,
    ) -> torch.Tensor:
        return _multilabel_auprc_compute(
            torch.cat(self.inputs),
            torch.cat(self.targets),
            self.num_labels,
            self.average,
        )

    @torch.inference_mode()
    def merge_state(
        self: TMultilabelAUPRC, metrics: Iterable[TMultilabelAUPRC]
    ) -> TMultilabelAUPRC:
        for metric in metrics:
            if metric.inputs:
                metric_inputs = torch.cat(metric.inputs).to(self.device)
                metric_targets = torch.cat(metric.targets).to(self.device)
                self.inputs.append(metric_inputs)
                self.targets.append(metric_targets)
        return self

    @torch.inference_mode()
    def _prepare_for_merge_state(self: TMultilabelAUPRC) -> None:
        if self.inputs and self.targets:
            self.inputs = [torch.cat(self.inputs)]
            self.targets = [torch.cat(self.targets)]
