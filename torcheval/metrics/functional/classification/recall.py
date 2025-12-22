# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging

import torch


@torch.inference_mode()
def binary_recall(
    input: torch.Tensor,
    target: torch.Tensor,
    *,
    threshold: float = 0.5,
) -> torch.Tensor:
    """
    Compute recall score for binary classification class, which is calculated as the ratio between the number of
    true positives (TP) and the total number of actual positives (TP + FN).
    Its class version is :obj:`torcheval.metrics.BinaryRecall`.
    See also :func:`multiclass_recall <torcheval.metrics.functional.multiclass_recall>`

    Args:
        input (Tensor): Tensor of the predicted labels/logits/probabilities, with shape of (n_sample, ).
        target (Tensor): Tensor of ground truth labels with shape of (n_sample, ).
        threshold (float, default 0.5): Threshold for converting input into predicted labels for each sample.
            ``torch.where(input < threshold, 0, 1)`` will be applied to the ``input``.
    Examples::

        >>> import torch
        >>> from torcheval.metrics.functional.classification import binary_recall
        >>> input = torch.tensor([0, 0, 1, 1])
        >>> target = torch.tensor([0, 1, 1, 1])
        >>> binary_recall(input, target)
        tensor(0.6667)  # 2 / 3
        >>> input = torch.tensor([0, 0.2, 0.4, 0.7])
        >>> target = torch.tensor([1, 0, 1, 1])
        >>> binary_recall(input, target)
        tensor(0.3333)  # 1 / 3
        >>> binary_recall(input, target, threshold=0.4)
        tensor(0.5000)  # 1 / 2
    """
    num_tp, num_true_labels = _binary_recall_update(input, target, threshold)
    return _binary_recall_compute(num_tp, num_true_labels)


def _binary_recall_update(
    input: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    _binary_recall_update_input_check(input, target)

    input = torch.where(input < threshold, 0, 1)

    num_tp = (input & target).sum()
    num_true_labels = target.sum()

    return num_tp, num_true_labels


def _binary_recall_compute(
    num_tp: torch.Tensor,
    num_true_labels: torch.Tensor,
) -> torch.Tensor:
    recall = num_tp / num_true_labels

    if torch.isnan(recall):
        logging.warning(
            "No positive instances have been seen in target. Recall is converted from NaN to 0s."
        )
        return torch.nan_to_num(recall)

    return recall


def _binary_recall_update_input_check(
    input: torch.Tensor,
    target: torch.Tensor,
) -> None:
    if input.shape != target.shape:
        raise ValueError(
            "The `input` and `target` should have the same dimensions, "
            f"got shapes {input.shape} and {target.shape}."
        )
    if target.ndim != 1:
        raise ValueError(
            f"target should be a one-dimensional tensor, got shape {target.shape}."
        )


@torch.inference_mode()
def multiclass_recall(
    input: torch.Tensor,
    target: torch.Tensor,
    *,
    num_classes: int | None = None,
    average: str | None = "micro",
) -> torch.Tensor:
    """
    Compute recall score, which is calculated as the ratio between the number of
    true positives (TP) and the total number of actual positives (TP + FN).
    Its class version is :obj:`torcheval.metrics.MulticlassRecall`.
    See also :func:`binary_recall <torcheval.metrics.functional.binary_recall>`

    Args:
        input (Tensor): Tensor of label predictions
            It could be the predicted labels, with shape of (n_sample, ).
            It could also be probabilities or logits with shape of (n_sample, n_class).
            ``torch.argmax`` will be used to convert input into predicted labels.
        target (Tensor): Tensor of ground truth labels with shape of (n_sample, ).
        num_classes:
            Number of classes.
        average:
            - ``'micro'`` [default]:
                Calculate the metrics globally, by using the total true positives and false
                negatives across all classes.
            - ``'macro'``:
                Calculate metrics for each class separately, and return their unweighted
                mean. Classes with 0 true and predicted instances are ignored.
            - ``'weighted'``:
                Calculate metrics for each class separately, and return their average weighted
                by the number of instances for each class in the ``target`` tensor. Classes with
                0 true and predicted instances are ignored.
            - ``None``:
                Calculate the metric for each class separately, and return
                the metric for every class.

    Examples::

        >>> import torch
        >>> from torcheval.metrics.functional.classification import multiclass_recall
        >>> input = torch.tensor([0, 2, 1, 3])
        >>> target = torch.tensor([0, 1, 2, 3])
        >>> multiclass_recall(input, target)
        tensor(0.5000)
        >>> multiclass_recall(input, target, average=None, num_classes=4)
        tensor([1., 0., 0., 1.])
        >>> multiclass_recall(input, target, average="macro", num_classes=4)
        tensor(0.5000)
        >>> input = torch.tensor([[0.9, 0.1, 0, 0], [0.1, 0.2, 0.4, 0.3], [0, 1.0, 0, 0], [0, 0, 0.2, 0.8]])
        >>> multiclass_recall(input, target)
        tensor(0.5000)
    """
    _recall_param_check(num_classes, average)
    num_tp, num_labels, num_predictions = _recall_update(
        input, target, num_classes, average
    )
    return _recall_compute(num_tp, num_labels, num_predictions, average)


def _recall_update(
    input: torch.Tensor,
    target: torch.Tensor,
    num_classes: int | None,
    average: str | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _recall_update_input_check(input, target, num_classes)

    if input.ndim == 2:
        input = torch.argmax(input, dim=1)

    if average == "micro":
        num_tp = (input == target).sum()
        num_labels = target.new_tensor(target.numel())
        num_predictions = num_labels
        return num_tp, num_labels, num_predictions

    assert isinstance(
        num_classes, int
    ), f"`num_classes` must be an integer, but received {num_classes}."
    num_labels = target.new_zeros(num_classes).scatter_add_(
        0, target, torch.ones_like(target)
    )
    num_predictions = target.new_zeros(num_classes).scatter_add_(
        0, input, torch.ones_like(input)
    )
    num_tp = target.new_zeros(num_classes).scatter_add_(
        0, target[input == target], torch.ones_like(target)
    )
    return num_tp, num_labels, num_predictions


def _recall_compute(
    num_tp: torch.Tensor,
    num_labels: torch.Tensor,
    num_predictions: torch.Tensor,
    average: str | None,
) -> torch.Tensor:
    if average in ("macro", "weighted"):
        # Ignore classes which have no samples in `target` and `input`
        mask = (num_labels != 0) | (num_predictions != 0)
        num_tp = num_tp[mask]
        num_labels = num_labels[mask]

    recall = num_tp / num_labels

    isnan_class = torch.isnan(recall)
    if isnan_class.any():
        nan_classes = isnan_class.nonzero(as_tuple=True)[0]
        logging.warning(
            f"One or more NaNs identified, as no ground-truth instances of "
            f"{nan_classes.tolist()} have been seen. These have been converted to zero."
        )
        recall = torch.nan_to_num(recall)

    if average == "micro":
        return recall
    elif average == "macro":
        return recall.mean()
    elif average == "weighted":
        # pyre-fixme[61]: `mask` is undefined, or not always defined.
        weights = num_labels / num_labels.sum()
        return (recall * weights).sum()
    else:  # average is None
        return recall


def _recall_param_check(num_classes: int | None, average: str | None) -> None:
    average_options = ("micro", "macro", "weighted", None)
    if average not in average_options:
        raise ValueError(
            f"`average` was not in the allowed values of {average_options}, "
            f"got {average}."
        )
    if average != "micro" and (num_classes is None or num_classes <= 0):
        raise ValueError(
            f"`num_classes` should be a positive number when average={average}, "
            f"got num_classes={num_classes}."
        )


def _recall_update_input_check(
    input: torch.Tensor, target: torch.Tensor, num_classes: int | None
) -> None:
    if input.size(0) != target.size(0):
        raise ValueError(
            f"The `input` and `target` should have the same first dimension, "
            f"got shapes {input.shape} and {target.shape}."
        )
    if target.ndim != 1:
        raise ValueError(
            f"`target` should be a one-dimensional tensor, got shape {target.shape}."
        )
    if input.ndim != 1 and not (
        input.ndim == 2 and (num_classes is None or input.shape[1] == num_classes)
    ):
        raise ValueError(
            f"`input` should have shape (num_samples,) or (num_samples, num_classes), "
            f"got {input.shape}."
        )
