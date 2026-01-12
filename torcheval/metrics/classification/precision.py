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
from torcheval.metrics.functional.classification.precision import (
    _binary_precision_update,
    _precision_compute,
    _precision_param_check,
    _precision_update,
)
from torcheval.metrics.metric import Metric

TPrecision = TypeVar("TPrecision")
TBinaryPrecision = TypeVar("TBinaryPrecision")


class MulticlassPrecision(Metric[torch.Tensor]):
    """
    Compute the precision score, the ratio of the true positives and the sum of
    true positives and false positives.
    Its functional version is :func:`torcheval.metrics.functional.multiclass_precision`.
    We cast NaNs to 0 in case some classes have zero instances in the predictions.
    See also :class:`BinaryPrecision <BinaryPrecision>`

    Args:
        num_classes (int):
            Number of classes.
        average (str):
            - ``"micro"`` (default): Calculate the metrics globally.
            - ``"macro"``: Calculate metrics for each class separately, and return their unweighted mean.
              Classes with 0 true and predicted instances are ignored.
            - ``"weighted"``: Calculate metrics for each class separately, and return their weighted sum.
              Weights are defined as the proportion of occurrences of each class in "target".
              Classes with 0 true and predicted instances are ignored.
            - ``None``: Calculate the metric for each class separately, and return
              the metric for every class.

    Examples::

        >>> import torch
        >>> from torcheval.metrics import MulticlassPrecision
        >>> metric = MulticlassPrecision(num_classes=4)
        >>> input = torch.tensor([0, 2, 1, 3])
        >>> target = torch.tensor([0, 1, 2, 3])
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor(0.5000)

        >>> metric = MulticlassPrecision(average=None, num_classes=4)
        >>> input = torch.tensor([0, 2, 1, 3])
        >>> target = torch.tensor([0, 1, 2, 3])
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor([1., 0., 0., 1.])

        >>> metric = MulticlassPrecision(average="macro", num_classes=2)
        >>> input = torch.tensor([0, 0, 1, 1, 1])
        >>> target = torch.tensor([0, 0, 0, 0, 1])
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor(0.5833)

        >>> metric = MulticlassPrecision(num_classes=4)
        >>> input = torch.tensor([[0.9, 0.1, 0, 0], [0.1, 0.2, 0.4, 0.3], [0, 1.0, 0, 0], [0, 0, 0.2, 0.8]])
        >>> target = torch.tensor([0, 1, 2, 3])
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor(0.5)
    """

    def __init__(
        self: TPrecision,
        *,
        num_classes: int | None = None,
        average: str | None = "micro",
        device: torch.device | None = None,
    ) -> None:
        super().__init__(device=device)
        _precision_param_check(num_classes, average)
        self.num_classes = num_classes
        self.average = average
        if average == "micro":
            self._add_state("num_tp", torch.tensor(0.0, device=self.device))
            self._add_state("num_fp", torch.tensor(0.0, device=self.device))
            self._add_state("num_label", torch.tensor(0.0, device=self.device))
        else:
            # num_classes has been verified as a positive integer. Add this line to bypass pyre.
            assert isinstance(num_classes, int), (
                f"num_classes must be a integer, but got {num_classes}"
            )

            self._add_state(
                "num_tp",
                torch.zeros(num_classes, device=self.device),
            )
            self._add_state(
                "num_fp",
                torch.zeros(num_classes, device=self.device),
            )
            self._add_state(
                "num_label",
                torch.zeros(num_classes, device=self.device),
            )

    @torch.inference_mode()
    # pyre-ignore[14]: inconsistent override on *_:Any, **__:Any
    def update(
        self: TPrecision, input: torch.Tensor, target: torch.Tensor
    ) -> TPrecision:
        """
        Update states with the ground truth labels and predictions.

        Args:
            input (Tensor): Tensor of label predictions.
                It could be the predicted labels, with shape of (n_sample, ).
                It could also be probabilities or logits with shape of (n_sample, n_class).
                ``torch.argmax`` will be used to convert input into predicted labels.
            target (Tensor): Tensor of ground truth labels with shape of (n_sample, ).
        """
        input = input.to(self.device)
        target = target.to(self.device)

        num_tp, num_fp, num_label = _precision_update(
            input, target, self.num_classes, self.average
        )
        self.num_tp += num_tp
        self.num_fp += num_fp
        self.num_label += num_label
        return self

    @torch.inference_mode()
    def compute(self: TPrecision) -> torch.Tensor:
        """
        Return the precision score.

        0 is returned if no calls to ``update()`` are made before ``compute()`` is called.
        """
        return _precision_compute(
            self.num_tp, self.num_fp, self.num_label, self.average
        )

    @torch.inference_mode()
    def merge_state(self: TPrecision, metrics: Iterable[TPrecision]) -> TPrecision:
        for metric in metrics:
            self.num_tp += metric.num_tp.to(self.device)
            self.num_fp += metric.num_fp.to(self.device)
            self.num_label += metric.num_label.to(self.device)
        return self


class BinaryPrecision(MulticlassPrecision):
    """
    Compute the precision score for binary classification tasks, which is calculated
    as the ratio of the true positives and the sum of true positives and false positives.
    Its functional version is :func:`torcheval.metrics.functional.binary_precision`.
    We cast NaNs to 0 when classes have zero positive instances in prediction labels
    (when TP + FP = 0).
    See also :class:`MulticlassPrecision <MulticlassPrecision>`

    Args:
        threshold (float, default = 0.5): Threshold for converting input into predicted labels for each sample.
            ``torch.where(input < threshold, 0, 1)`` will be applied to the ``input``.

    Examples::

        >>> import torch
        >>> from torcheval.metrics import BinaryPrecision
        >>> metric = BinaryPrecision()
        >>> input = torch.tensor([0, 1, 0, 1])
        >>> target = torch.tensor([1, 0, 1, 1])
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor(0.5)  # 1 / 2

        >>> metric = BinaryPrecision(threshold=0.7)
        >>> input = torch.tensor([0, 0.9, 0.6, 0.7])
        >>> target = torch.tensor([1, 0, 1, 1])
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor(0.5)  # 1 / 2
    """

    def __init__(
        self: TBinaryPrecision,
        *,
        threshold: float = 0.5,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(num_classes=2, device=device)
        self.threshold = threshold

    @torch.inference_mode()
    def update(
        self: TBinaryPrecision,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> TBinaryPrecision:
        """
        Update states with the ground truth labels and predictions.

            input (Tensor): Tensor of the predicted labels/logits/probabilities, with shape of (n_sample, ).
                   ``torch.where(input < threshold, 0, 1)`` will be used to convert input into predicted labels.
            target (Tensor): Tensor of ground truth labels with shape of (n_sample,).
        """
        input = input.to(self.device)
        target = target.to(self.device)

        num_tp, num_fp, num_label = _binary_precision_update(
            input, target, self.threshold
        )
        self.num_tp += num_tp
        self.num_fp += num_fp
        self.num_label += num_label
        return self
