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
from torcheval.metrics.functional.classification.accuracy import (
    _accuracy_compute,
    _accuracy_param_check,
    _binary_accuracy_update,
    _multiclass_accuracy_update,
    _multilabel_accuracy_param_check,
    _multilabel_accuracy_update,
    _topk_multilabel_accuracy_param_check,
    _topk_multilabel_accuracy_update,
)
from torcheval.metrics.metric import Metric


TAccuracy = TypeVar("TAccuracy")
TBinaryAccuracy = TypeVar("TBinaryAccuracy")
TMultilabelAccuracy = TypeVar("TMultilabelAccuracy")
TTopKMultilabelAccuracy = TypeVar("TTopKMultilabelAccuracy")


class MulticlassAccuracy(Metric[torch.Tensor]):
    """
    Compute accuracy score, which is the frequency of input matching target.
    Its functional version is :func:`torcheval.metrics.functional.multiclass_accuracy`.
    See also :class:`BinaryAccuracy <BinaryAccuracy>`, :class:`MultilabelAccuracy <MultilabelAccuracy>`, :class:`TopKMultilabelAccuracy <TopKMultilabelAccuracy>`

    Args:
        average (str, Optional)
            - ``'micro'`` [default]: Calculate the metrics globally.
            - ``'macro'`` : Calculate metrics for each class separately, and return their unweighted
              mean. Classes with 0 true instances are ignored.
            - ``None``: Calculate the metric for each class separately, and return
              the metric for every class.
              NaN is returned if a class has no sample in ``target``.
        num_classes:
            Number of classes. Required for ``'macro'`` and ``None`` average methods.
        k: Number of top probabilities to be considered. K should be an integer greater than or equal to 1.
            If k >1, the input tensor must contain probabilities or logits for every class.

    Examples::

        >>> import torch
        >>> from torcheval.metrics import MulticlassAccuracy
        >>> metric = MulticlassAccuracy()
        >>> input = torch.tensor([0, 2, 1, 3])
        >>> target = torch.tensor([0, 1, 2, 3])
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor(0.5)

        >>> metric = MulticlassAccuracy(average=None, num_classes=4)
        >>> input = torch.tensor([0, 2, 1, 3])
        >>> target = torch.tensor([0, 1, 2, 3])
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor([1., 0., 0., 1.])

        >>> metric = MulticlassAccuracy(average="macro", num_classes=2)
        >>> input = torch.tensor([0, 0, 1, 1, 1])
        >>> target = torch.tensor([0, 0, 0, 0, 1])
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor(0.75)

        >>> metric = MulticlassAccuracy()
        >>> input = torch.tensor([[0.9, 0.1, 0, 0], [0.1, 0.2, 0.4, 0,3], [0, 1.0, 0, 0], [0, 0, 0.2, 0.8]])
        >>> target = torch.tensor([0, 1, 2, 3])
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor(0.5)
    """

    def __init__(
        self: TAccuracy,
        *,
        average: str | None = "micro",
        num_classes: int | None = None,
        k: int = 1,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(device=device)
        _accuracy_param_check(average, num_classes, k)
        self.average = average
        self.num_classes = num_classes
        self.k = k
        if average == "micro":
            self._add_state("num_correct", torch.tensor(0.0, device=self.device))
            self._add_state("num_total", torch.tensor(0.0, device=self.device))
        else:
            # num_classes is verified to be not None when average != "micro"
            self._add_state(
                "num_correct",
                torch.zeros(num_classes or 0, device=self.device),
            )
            self._add_state(
                "num_total",
                torch.zeros(num_classes or 0, device=self.device),
            )

    @torch.inference_mode()
    # pyre-ignore[14]: inconsistent override on *_:Any, **__:Any
    def update(self: TAccuracy, input: torch.Tensor, target: torch.Tensor) -> TAccuracy:
        """
        Update states with the ground truth labels and predictions.

        Args:
            input (Tensor): Tensor of label predictions
                It could be the predicted labels, with shape of (n_sample, ).
                It could also be probabilities or logits with shape of (n_sample, n_class).
                ``torch.argmax`` will be used to convert input into predicted labels.
            target (Tensor): Tensor of ground truth labels with shape of (n_sample, ).
        """
        input = input.to(self.device)
        target = target.to(self.device)

        num_correct, num_total = _multiclass_accuracy_update(
            input, target, self.average, self.num_classes, self.k
        )
        self.num_correct += num_correct
        self.num_total += num_total
        return self

    @torch.inference_mode()
    def compute(self: TAccuracy) -> torch.Tensor:
        """
        Return the accuracy score.

        NaN is returned if no calls to ``update()`` are made before ``compute()`` is called.
        """
        return _accuracy_compute(self.num_correct, self.num_total, self.average)

    @torch.inference_mode()
    def merge_state(self: TAccuracy, metrics: Iterable[TAccuracy]) -> TAccuracy:
        for metric in metrics:
            self.num_correct += metric.num_correct.to(self.device)
            self.num_total += metric.num_total.to(self.device)
        return self


class BinaryAccuracy(MulticlassAccuracy):
    """
    Compute binary accuracy score, which is the frequency of input matching target.
    Its functional version is :func:`torcheval.metrics.functional.binary_accuracy`.
    See also :class:`MulticlassAccuracy <MulticlassAccuracy>`, :class:`MultilabelAccuracy <MultilabelAccuracy>`, :class:`TopKMultilabelAccuracy <TopKMultilabelAccuracy>`

    Args:

        threshold (float, default 0.5): Threshold for converting input into predicted labels for each sample.
          ``torch.where(input < threshold, 0, 1)`` will be applied to the ``input``.

    Examples::

        >>> import torch
        >>> from torcheval.metrics import BinaryAccuracy
        >>> metric = BinaryAccuracy()
        >>> input = torch.tensor([0, 0, 1, 1])
        >>> target = torch.tensor([1, 0, 1, 1])
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor(0.75)  # 3 / 4


        >>> metric = BinaryAccuracy(threshold=0.7)
        >>> input = torch.tensor([0, 0.2, 0.6, 0.7])
        >>> target = torch.tensor([1, 0, 1, 1])
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor(0.5)  # 2 / 4

    """

    def __init__(
        self: TBinaryAccuracy,
        *,
        threshold: float = 0.5,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(device=device)
        self.threshold = threshold

    @torch.inference_mode()
    def update(
        self: TBinaryAccuracy,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> TBinaryAccuracy:
        """
        Update states with the ground truth labels and predictions.

        Args:
            input (Tensor): Tensor of label predictions with shape of (n_sample,).
                ``torch.where(input < threshold, 0, 1)`` will be applied to the input.
            target (Tensor): Tensor of ground truth labels with shape of (n_sample,).
        """
        input = input.to(self.device)
        target = target.to(self.device)

        num_correct, num_total = _binary_accuracy_update(input, target, self.threshold)
        self.num_correct += num_correct
        self.num_total += num_total
        return self


class MultilabelAccuracy(MulticlassAccuracy):
    """
    Compute multilabel accuracy score, which is the frequency of input matching target.
    Its functional version is :func:`torcheval.metrics.functional.multilabel_accuracy`.
    See also :class:`MulticlassAccuracy <MulticlassAccuracy>`, :class:`BinaryAccuracy <BinaryAccuracy>`, :class:`TopKMultilabelAccuracy <TopKMultilabelAccuracy>`

    Args:
        threshold (float, Optional): Threshold for converting input into predicted labels for each sample.
            ``torch.where(input < threshold, 0, 1)`` will be applied to the ``input``.
        criteria (str, Optional):
            - ``'exact_match'`` [default]:
              The set of labels predicted for a sample must exactly match the corresponding
              set of labels in target. Also known as subset accuracy.
            - ``'hamming'``:
              Fraction of correct labels over total number of labels.
            - ``'overlap'``:
              The set of labels predicted for a sample must overlap with the corresponding
              set of labels in target.
            - ``'contain'``:
              The set of labels predicted for a sample must contain the corresponding
              set of labels in target.
            - ``'belong'``:
              The set of labels predicted for a sample must (fully) belong to the corresponding
              set of labels in target.

    Examples::

        >>> import torch
        >>> from torcheval.metrics import MultilabelAccuracy
        >>> metric = MultilabelAccuracy()
        >>> input = torch.tensor([[0, 1], [1, 1], [0, 0], [0, 1]])
        >>> target = torch.tensor([[0, 1], [1, 0], [0, 0], [1, 1]])
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor(0.5)  # 2 / 4

        >>> metric = MultilabelAccuracy(criteria="hamming")
        >>> input = torch.tensor([[0, 1], [1, 1], [0, 0], [0, 1]])
        >>> target = torch.tensor([[0, 1], [1, 0], [0, 0], [1, 1]])
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor(0.75)  # 6 / 8

        >>> metric = MultilabelAccuracy(criteria="overlap")
        >>> input = torch.tensor([[0, 1], [1, 1], [0, 0], [0, 1]])
        >>> target = torch.tensor([[0, 1], [1, 0], [0, 0], [1, 1]])
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor(1)  # 4 / 4

        >>> metric = MultilabelAccuracy(criteria="contain")
        >>> input = torch.tensor([[0, 1], [1, 1], [0, 0], [0, 1]])
        >>> target = torch.tensor([[0, 1], [1, 0], [0, 0], [1, 1]])
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor(0.75)  # 3 / 4, input[0],input[1],input[2]

        >>> metric = MultilabelAccuracy(criteria="belong")
        >>> input = torch.tensor([[0, 1], [1, 1], [0, 0], [0, 1]])
        >>> target = torch.tensor([[0, 1], [1, 0], [0, 0], [1, 1]])
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor(0.75)  # 3 / 4, input[0],input[1],input[3]
    """

    def __init__(
        self: TMultilabelAccuracy,
        *,
        threshold: float = 0.5,
        criteria: str = "exact_match",
        device: torch.device | None = None,
    ) -> None:
        super().__init__(device=device)
        _multilabel_accuracy_param_check(criteria)
        self.threshold = threshold
        self.criteria = criteria

    @torch.inference_mode()
    def update(
        self: TMultilabelAccuracy,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> TMultilabelAccuracy:
        """
        Update states with the ground truth labels and predictions.

        Args:
            input (Tensor): Tensor of label predictions with shape of (n_sample, n_class).
                ``torch.where(input < threshold, 0, 1)`` will be applied to the input.
            target (Tensor): Tensor of ground truth labels with shape of (n_sample, n_class).
        """
        input = input.to(self.device)
        target = target.to(self.device)

        num_correct, num_total = _multilabel_accuracy_update(
            input, target, self.threshold, self.criteria
        )
        self.num_correct += num_correct
        self.num_total += num_total
        return self


class TopKMultilabelAccuracy(MulticlassAccuracy):
    """
    Compute multilabel accuracy score, which is the frequency of the top k label predicted matching target.
    Its functional version is :func:`torcheval.metrics.functional.topk_multilabel_accuracy`.
    See also :class:`MulticlassAccuracy <MulticlassAccuracy>`, :class:`BinaryAccuracy <BinaryAccuracy>`, :class:`MultilabelAccuracy <MultilabelAccuracy>`

    Args:
        criteria (string):
            - ``'exact_match'`` [default]: The set of top-k labels predicted for a sample must exactly match the corresponding
              set of labels in target. Also known as subset accuracy.
            - ``'hamming'``: Fraction of top-k correct labels over total number of labels.
            - ``'overlap'``: The set of top-k labels predicted for a sample must overlap with the corresponding
              set of labels in target.
            - ``'contain'``: The set of top-k labels predicted for a sample must contain the corresponding
              set of labels in target.
            - ``'belong'``: The set of top-k labels predicted for a sample must (fully) belong to the corresponding
              set of labels in target.
        k (int): Number of top probabilities to be considered. K should be an integer greater than or equal to 1.

    Examples::

        >>> import torch
        >>> from torcheval.metrics import TopKMultilabelAccuracy
        >>> metric = TopKMultilabelAccuracy(k = 2)
        >>> input = torch.tensor([[0.1, 0.5, 0.2], [0.3, 0.2, 0.1], [0.2, 0.4, 0.5], [0, 0.1, 0.9]])
        >>> target = torch.tensor([[1, 1, 0], [0, 1, 0], [1, 1, 1], [0, 1, 0]])
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor(0)  # 0 / 4

        >>> metric = TopKMultilabelAccuracy(criteria="hamming", k=2)
        >>> input = torch.tensor([[0.1, 0.5, 0.2], [0.3, 0.2, 0.1], [0.2, 0.4, 0.5], [0, 0.1, 0.9]])
        >>> target = torch.tensor([[1, 1, 0], [0, 1, 0], [1, 1, 1], [0, 1, 0]])
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor(0.583)  # 7 / 12

        >>> metric = TopKMultilabelAccuracy(criteria="overlap", k=2)
        >>> input = torch.tensor([[0.1, 0.5, 0.2], [0.3, 0.2, 0.1], [0.2, 0.4, 0.5], [0, 0.1, 0.9]])
        >>> target = torch.tensor([[1, 1, 0], [0, 1, 0], [1, 1, 1], [0, 1, 0]])
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor(1)  # 4 / 4

        >>> metric = TopKMultilabelAccuracy(criteria="contain", k=2)
        >>> input = torch.tensor([[0.1, 0.5, 0.2], [0.3, 0.2, 0.1], [0.2, 0.4, 0.5], [0, 0.1, 0.9]])
        >>> target = torch.tensor([[1, 1, 0], [0, 1, 0], [1, 1, 1], [0, 1, 0]])
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor(0.5)  # 2 / 4

        >>> metric = TopKMultilabelAccuracy(criteria="belong", k=2)
        >>> input = torch.tensor([[0.1, 0.5, 0.2], [0.3, 0.2, 0.1], [0.2, 0.4, 0.5], [0, 0.1, 0.9]])
        >>> target = torch.tensor([[1, 1, 0], [0, 1, 0], [1, 1, 1], [0, 1, 0]])
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor(0.25)  # 1 / 4
    """

    def __init__(
        self: TTopKMultilabelAccuracy,
        *,
        criteria: str = "exact_match",
        k: int = 1,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(device=device)
        _topk_multilabel_accuracy_param_check(criteria, k)
        self.criteria = criteria
        self.k = k

    @torch.inference_mode()
    def update(
        self: TTopKMultilabelAccuracy,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> TTopKMultilabelAccuracy:
        """
        Update states with the ground truth labels and predictions.

        Args:
            input (Tensor): Tensor of label predictions with shape of (n_sample, n_class).
                ``torch.where(input < threshold, 0, 1)`` will be applied to the input.
            target (Tensor): Tensor of ground truth labels with shape of (n_sample, n_class).
        """
        input = input.to(self.device)
        target = target.to(self.device)

        num_correct, num_total = _topk_multilabel_accuracy_update(
            input, target, self.criteria, self.k
        )
        self.num_correct += num_correct
        self.num_total += num_total
        return self
