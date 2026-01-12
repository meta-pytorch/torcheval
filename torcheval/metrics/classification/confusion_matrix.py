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
from torcheval.metrics.functional.classification.confusion_matrix import (
    _binary_confusion_matrix_update,
    _confusion_matrix_compute,
    _confusion_matrix_param_check,
    _confusion_matrix_update,
)
from torcheval.metrics.metric import Metric


TMulticlassConfusionMatrix = TypeVar("TMulticlassConfusionMatrix")
TBinaryConfusionMatrix = TypeVar("TBinaryConfusionMatrix")


class MulticlassConfusionMatrix(Metric[torch.Tensor]):
    """
    Compute multi-class confusion matrix, a matrix of dimension num_classes x num_classes where each element at position `(i,j)` is the number of examples with true class `i` that were predicted to be class `j`.
    See also :class:`BinaryConfusionMatrix <BinaryConfusionMatrix>`

    Args:
        input (Tensor): Tensor of label predictions.
            It could be the predicted labels, with shape of (n_sample, ).
            It could also be probabilities or logits with shape of (n_sample, n_class).
            ``torch.argmax`` will be used to convert input into predicted labels.
        target (Tensor): Tensor of ground truth labels with shape of (n_sample, ).
        num_classes (int):
            Number of classes.
        normalize (str):
            - ``None`` [default]:
                Give raw counts ('none' also defaults to this)
            - ``'pred'``:
                Normalize across the prediction class, i.e. such that the rows add to one.
            - ``'true'``:
                Normalize across the condition positive, i.e. such that the columns add to one.
            - ``'all'``"
                Normalize across all examples, i.e. such that all matrix entries add to one.
        device (torch.device): Device for internal tensors

    Examples::

        >>> import torch
        >>> from torcheval.metrics import MulticlassConfusionMatrix
        >>> input = torch.tensor([0, 2, 1, 3])
        >>> target = torch.tensor([0, 1, 2, 3])
        >>> metric = MulticlassConfusionMatrix(4)
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor([[1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1]])

        >>> input = torch.tensor([0, 0, 1, 1, 1])
        >>> target = torch.tensor([0, 0, 0, 0, 1])
        >>> metric = MulticlassConfusionMatrix(2)
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor([[2, 2],
                [0, 1]])

        >>> input = torch.tensor([0, 0, 1, 1, 1, 2, 1, 2])
        >>> target = torch.tensor([2, 0, 2, 0, 1, 2, 1, 0])
        >>> metric = MulticlassConfusionMatrix(3)
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor([[1, 1, 1],
                [0, 2, 0],
                [1, 1, 1]])

        >>> input = torch.tensor([0, 0, 1, 1, 1, 2, 1, 2])
        >>> target = torch.tensor([2, 0, 2, 0, 1, 2, 1, 0])
        >>> metric = MulticlassConfusionMatrix(3)
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor([[1., 1., 1.],
                [0., 2., 0.],
                [1., 1., 1.]])
        >>> metric.normalized("pred")
        tensor([[0.5000, 0.2500, 0.5000],
                [0.0000, 0.5000, 0.0000],
                [0.5000, 0.2500, 0.5000]])
        >>> metric.normalized("true")
        tensor([[0.3333, 0.3333, 0.3333],
                [0.0000, 1.0000, 0.0000],
                [0.3333, 0.3333, 0.3333]])
        >>> metric.normalized("all")
        tensor([[0.1250, 0.1250, 0.1250],
            [0.0000, 0.2500, 0.0000],
            [0.1250, 0.1250, 0.1250]])

        >>> input = torch.tensor([0, 0, 1, 1, 1, 2, 1, 2])
        >>> target = torch.tensor([2, 0, 2, 0, 1, 2, 1, 0])
        >>> metric = MulticlassConfusionMatrix(3, normalize="true")
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor([[0.3333, 0.3333, 0.3333],
                [0.0000, 1.0000, 0.0000],
                [0.3333, 0.3333, 0.3333]])
        >>> metric.normalized(None)
        tensor([[1., 1., 1.],
                [0., 2., 0.],
                [1., 1., 1.]])

        >>> input = torch.tensor([0, 0, 1, 1, 1])
        >>> target = torch.tensor([0, 0, 0, 0, 1])
        >>> metric = MulticlassConfusionMatrix(4)
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor([[2, 2, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0]])

        >>> input = torch.tensor([[0.9, 0.1, 0, 0], [0.1, 0.2, 0.4, 0.3], [0, 1.0, 0, 0], [0, 0, 0.2, 0.8]])
        >>> target = torch.tensor([0, 1, 2, 3])
        >>> metric = MulticlassConfusionMatrix(4)
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor([[1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1]])
    """

    def __init__(
        self: TMulticlassConfusionMatrix,
        num_classes: int,
        *,
        normalize: str | None = None,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(device=device)
        _confusion_matrix_param_check(num_classes, normalize)
        self.normalize = normalize
        self.num_classes = num_classes

        self._add_state(
            "confusion_matrix",
            torch.zeros([num_classes, num_classes], device=self.device),
        )

    @torch.inference_mode()
    # pyre-ignore[14]: inconsistent override on *_:Any, **__:Any
    def update(
        self: TMulticlassConfusionMatrix, input: torch.Tensor, target: torch.Tensor
    ) -> TMulticlassConfusionMatrix:
        """
        Update Confusion Matrix.

        Args:
            input (Tensor): Tensor of label predictions.
                It could be the predicted labels, with shape of (n_sample, ).
                It could also be probabilities or logits with shape of (n_sample, n_class).
                ``torch.argmax`` will be used to convert input into predicted labels.
            target (Tensor): Tensor of ground truth labels with shape of (n_sample, ).
        """
        input = input.to(self.device)
        target = target.to(self.device)

        self.confusion_matrix += _confusion_matrix_update(
            input, target, self.num_classes
        )
        return self

    @torch.inference_mode()
    def compute(self: TMulticlassConfusionMatrix) -> torch.Tensor:
        """
        Return the confusion matrix.
        """

        return _confusion_matrix_compute(
            self.confusion_matrix, normalize=self.normalize
        )

    @torch.inference_mode()
    def normalized(
        self: TMulticlassConfusionMatrix, normalize: str | None = None
    ) -> torch.Tensor:
        """
        Return the normalized confusion matrix

        Args:
            normalize (str): Can be overridden when calling compute()
            - ``None`` [default]:
                Give raw counts ('none' also defaults to this)
            - ``'pred'``:
                Normalize across the prediction class, i.e. such that the rows add to one.
            - ``'true'``:
                Normalize across the condition positive, i.e. such that the columns add to one.
            - ``'all'``"
                Normalize across all examples, i.e. such that all matrix entries add to one.
        """
        _confusion_matrix_param_check(self.num_classes, normalize)
        return _confusion_matrix_compute(self.confusion_matrix, normalize)

    @torch.inference_mode()
    def merge_state(
        self: TMulticlassConfusionMatrix, metrics: Iterable[TMulticlassConfusionMatrix]
    ) -> TMulticlassConfusionMatrix:
        for metric in metrics:
            self.confusion_matrix += metric.confusion_matrix.to(self.device)
        return self


class BinaryConfusionMatrix(MulticlassConfusionMatrix):
    """
    Compute binary confusion matrix, a 2 by 2 tensor with counts ( (true positive, false negative) , (false positive, true negative) )
    See also :class:`MulticlassConfusionMatrix <MulticlassConfusionMatrix>`

    Args:
        input (Tensor): Tensor of label predictions with shape of (n_sample,).
            ``torch.where(input < threshold, 0, 1)`` will be applied to the input.
        target (Tensor): Tensor of ground truth labels with shape of (n_sample,).
        threshold (float, default 0.5): Threshold for converting input into predicted labels for each sample.
            ``torch.where(input < threshold, 0, 1)`` will be applied to the ``input``.
        normalize (str):
            - ``None`` [default]:
                Give raw counts ('none' also defaults to this)
            - ``'pred'``:
                Normalize across the prediction class, i.e. such that the rows add to one.
            - ``'true'``:
                Normalize across the condition positive, i.e. such that the columns add to one.
            - ``'all'``"
                Normalize across all examples, i.e. such that all matrix entries add to one.
        device (torch.device): Device for internal tensors
    Examples::

        >>> import torch
        >>> from torcheval.metrics import BinaryConfusionMatrix
        >>> input = torch.tensor([0, 1, 0.7, 0.6])
        >>> target = torch.tensor([0, 1, 1, 0])
        >>> metric = BinaryConfusionMatrix()
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor([[1, 1],
                [0, 2]])

        >>> input = torch.tensor([0, 1, 0.7, 0.6])
        >>> target = torch.tensor([0, 1, 1, 0])
        >>> metric = BinaryConfusionMatrix(threshold=1)
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor([[0, 1],
                [2, 1]])

        >>> input = torch.tensor([1, 1, 0, 0])
        >>> target = torch.tensor([0, 1, 1, 1])
        >>> metric = BinaryConfusionMatrix()
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor([[0., 1.],
                [2., 1.]])
        >>> metric.normalized("pred")
        tensor([[0.0000, 0.5000],
                [1.0000, 0.5000]])
        >>> metric.normalized("true")
        tensor([[0.0000, 1.0000],
                [0.6667, 0.3333]])
        >>> metric.normalized("all")
        tensor([[0.0000, 0.5000],
                [1.0000, 0.5000]])

        >>> input = torch.tensor([1, 1, 0, 0])
        >>> target = torch.tensor([0, 1, 1, 1])
        >>> metric = BinaryConfusionMatrix(normalize="true")
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor([[0.0000, 1.0000],
                [0.6667, 0.3333]])
        >>> metric.normalized(None)
        tensor([[0., 1.],
                [2., 1.]])

    """

    def __init__(
        self: TBinaryConfusionMatrix,
        *,
        threshold: float = 0.5,
        normalize: str | None = None,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(num_classes=2, device=device, normalize=normalize)
        self.threshold = threshold

    @torch.inference_mode()
    def update(
        self: TBinaryConfusionMatrix, input: torch.Tensor, target: torch.Tensor
    ) -> TBinaryConfusionMatrix:
        """
        Update the confusion matrix
        Args:
            input (Tensor): Tensor of label predictions with shape of (n_sample,).
                ``torch.where(input < threshold, 0, 1)`` will be applied to the input.
            target (Tensor): Tensor of ground truth labels with shape of (n_sample,).
        """
        input = input.to(self.device)
        target = target.to(self.device)

        self.confusion_matrix += _binary_confusion_matrix_update(
            input, target, self.threshold
        )
        return self
