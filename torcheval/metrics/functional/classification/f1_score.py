# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# pyre-ignore-all-errors[6]: expect int got Optional[int] for num_classes

import logging

import torch


@torch.inference_mode()
def binary_f1_score(
    input: torch.Tensor,
    target: torch.Tensor,
    *,
    threshold: float = 0.5,
) -> torch.Tensor:
    """
    Compute binary f1 score, the harmonic mean of precision and recall.
    See also :func:`multiclass_f1_score <torcheval.metrics.functional.multiclass_f1_score>`

    Args:
        input (Tensor): Tensor of label predictions with shape of (n_sample,).
            ``torch.where(input < threshold, 0, 1)`` will be applied to the input.
        target (Tensor): Tensor of ground truth labels with shape of (n_sample,).
        threshold (float, default 0.5): Threshold for converting input into predicted labels for each sample.
            ``torch.where(input < threshold, 0, 1)`` will be applied to the ``input``.
    Examples::

        >>> import torch
        >>> from torcheval.metrics.functional import binary_f1_score
        >>> input = torch.tensor([0, 1, 0.7, 0.6])
        >>> target = torch.tensor([0, 1, 1, 0])
        >>> binary_f1_score(input, target, threshold=0.5)
        tensor(0.8000)

        >>> input = torch.tensor([1, 1, 0, 0])
        >>> target = torch.tensor([0, 1, 1, 1])
        >>> binary_f1_score(input, target, threshold=1)
        tensor(0.4000)
    """
    num_tp, num_label, num_prediction = _binary_f1_score_update(
        input, target, threshold
    )
    return _f1_score_compute(num_tp, num_label, num_prediction, "micro")


@torch.inference_mode()
def multiclass_f1_score(
    input: torch.Tensor,
    target: torch.Tensor,
    *,
    num_classes: int | None = None,
    average: str | None = "micro",
) -> torch.Tensor:
    """
    Compute f1 score, which is defined as the harmonic mean of precision and recall.
    We convert NaN to zero when f1 score is NaN. This happens when either precision
    or recall is NaN or when both precision and recall are zero.
    Its class version is :obj:`torcheval.metrics.MulticlassF1Score`.
    See also :func:`binary_f1_score <torcheval.metrics.functional.binary_f1_score>`

    Args:
        input (Tensor): Tensor of label predictions.
            It could be the predicted labels, with shape of (n_sample, ).
            It could also be probabilities or logits with shape of (n_sample, n_class).
            ``torch.argmax`` will be used to convert input into predicted labels.
        target (Tensor): Tensor of ground truth labels with shape of (n_sample, ).
        num_classes:
            Number of classes.
        average:
            - ``'micro'`` [default]:
                Calculate the metrics globally.
            - ``'macro'``:
                Calculate metrics for each class separately, and return their unweighted mean.
                Classes with 0 true and predicted instances are ignored.
            - ``'weighted'``"
                Calculate metrics for each class separately, and return their weighted sum.
                Weights are defined as the proportion of occurrences of each class in "target".
                Classes with 0 true and predicted instances are ignored.
            - ``None``:
                Calculate the metric for each class separately, and return
                the metric for every class.

    Examples::

        >>> import torch
        >>> from torcheval.metrics.functional import multiclass_f1_score
        >>> input = torch.tensor([0, 2, 1, 3])
        >>> target = torch.tensor([0, 1, 2, 3])
        >>> multiclass_f1_score(input, target, num_classes=4)
        tensor(0.5000)

        >>> input = torch.tensor([0, 2, 1, 3])
        >>> target = torch.tensor([0, 1, 2, 3])
        >>> multiclass_f1_score(input, target, num_classes=4, average=None)
        tensor([1., 0., 0., 1.])

        >>> input = torch.tensor([0, 0, 1, 1, 1])
        >>> target = torch.tensor([0, 0, 0, 0, 1])
        >>> multiclass_f1_score(input, target, num_classes=2, average="macro")
        tensor(0.5833)

        >>> input = torch.tensor([[0.9, 0.1, 0, 0], [0.1, 0.2, 0.4, 0.3], [0, 1.0, 0, 0], [0, 0, 0.2, 0.8]])
        >>> target = torch.tensor([0, 1, 2, 3])
        >>> multiclass_f1_score(input, target, num_classes=4)
        tensor(0.5)
    """
    _f1_score_param_check(num_classes, average)
    num_tp, num_label, num_prediction = _f1_score_update(
        input, target, num_classes, average
    )
    return _f1_score_compute(num_tp, num_label, num_prediction, average)


def _binary_f1_score_update(
    input: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _binary_f1_score_update_input_check(input, target)

    input = torch.where(input < threshold, 0, 1)

    num_tp = (input * target).sum()  # num true positive
    num_label = target.sum()  # tp + fn (num condition positive)
    num_prediction = input.sum()  # tp + fp (num predicted positive)

    return num_tp, num_label, num_prediction


def _binary_f1_score_update_input_check(
    input: torch.Tensor,
    target: torch.Tensor,
) -> None:
    if input.ndim != 1:
        raise ValueError(
            f"input should be a one-dimensional tensor for binary f1 score, got shape {input.shape}."
        )
    if target.ndim != 1:
        raise ValueError(
            f"target should be a one-dimensional tensor for binary f1 score, got shape {target.shape}."
        )
    if input.shape != target.shape:
        raise ValueError(
            "The `input` and `target` should have the same dimensions, "
            f"got shapes {input.shape} and {target.shape}."
        )


def _f1_score_update(
    input: torch.Tensor,
    target: torch.Tensor,
    num_classes: int | None,
    average: str | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _f1_score_update_input_check(input, target, num_classes)
    return _update(input, target, num_classes, average)


@torch.jit.script
def _update(
    input: torch.Tensor,
    target: torch.Tensor,
    num_classes: int | None,
    average: str | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if input.ndim == 2:
        input = torch.argmax(input, dim=1)

    if isinstance(average, str) and average == "micro":
        num_tp = (input == target).sum()
        num_label = torch.tensor(target.shape[0], device=target.device)
        num_prediction = num_label
        return num_tp, num_label, num_prediction

    # Add this line to bypass torch jit datatype checking
    assert isinstance(num_classes, int)
    num_label = target.new_zeros(num_classes).scatter_add_(
        0, target, torch.ones_like(target)
    )
    num_prediction = target.new_zeros(num_classes).scatter_add_(
        0, input, torch.ones_like(input)
    )
    num_tp = target.new_zeros(num_classes).scatter_add_(
        0, target[input == target], torch.ones_like(target)
    )
    return num_tp, num_label, num_prediction


def _f1_score_compute(
    num_tp: torch.Tensor,
    num_label: torch.Tensor,
    num_prediction: torch.Tensor,
    average: str | None,
) -> torch.Tensor:
    # Check if all classes exist in either ``target``.
    num_label_is_zero = num_label == 0
    if num_label_is_zero.any():
        logging.warning(
            "Warning: Some classes do not exist in the target. F1 scores for these classes will be cast to zeros."
        )

    if average in ("macro", "weighted"):
        # Ignore the class that has no samples in both ``input`` and `target`.
        mask = (~num_label_is_zero) | (num_prediction != 0)
        num_tp, num_label, num_prediction = (
            num_tp[mask],
            num_label[mask],
            num_prediction[mask],
        )

    precision = num_tp / num_prediction
    recall = num_tp / num_label
    f1 = 2 * precision * recall / (precision + recall)

    # Convert NaN to zero when f1 score is NaN. This happens when either precision or recall is NaN or when both precision and recall are zero.
    f1 = torch.nan_to_num(f1)

    if average == "micro":
        return f1
    elif average == "macro":
        return f1.mean()
    elif average == "weighted":
        return (f1 * (num_label / num_label.sum())).sum()
    else:  # average is None
        return f1


def _f1_score_param_check(
    num_classes: int | None,
    average: str | None,
) -> None:
    average_options = ("micro", "macro", "weighted", None)
    if average not in average_options:
        raise ValueError(
            f"`average` was not in the allowed value of {average_options}, got {average}."
        )
    if average != "micro" and (num_classes is None or num_classes <= 0):
        raise ValueError(
            f"num_classes should be a positive number when average={average}, "
            f"got num_classes={num_classes}."
        )


def _f1_score_update_input_check(
    input: torch.Tensor,
    target: torch.Tensor,
    num_classes: int | None,
) -> None:
    if input.size(0) != target.size(0):
        raise ValueError(
            "The `input` and `target` should have the same first dimension, "
            f"got shapes {input.shape} and {target.shape}."
        )

    if target.ndim != 1:
        raise ValueError(
            "target should be a one-dimensional tensor, " f"got shape {target.shape}."
        )

    if not input.ndim == 1 and not (
        input.ndim == 2 and (num_classes is None or input.shape[1] == num_classes)
    ):
        raise ValueError(
            f"input should have shape of (num_sample,) or (num_sample, num_classes), "
            f"got {input.shape}."
        )
