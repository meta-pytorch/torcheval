# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import torch
import torch.nn.functional as F

"""
This file contains binary_precision_recall_curve, multiclass_precision_recall_curve and multilabel_precision_recall_curve functions.
"""


@torch.inference_mode()
def binary_precision_recall_curve(
    input: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns precision-recall pairs and their corresponding thresholds for
    binary classification tasks. If a class is missing from the target tensor,
    its recall values are set to 1.0.

    Its class version is :obj:`torcheval.metrics.BinaryPrecisionRecallCurve`.
    See also :func:`multiclass_precision_recall_curve <torcheval.metrics.functional.multiclass_precision_recall_curve>`, :func:`multilabel_precision_recall_curve <torcheval.metrics.functional.multilabel_precision_recall_curve>`

    Args:
        input (Tensor): Tensor of label predictions
            It should be probabilities or logits with shape of (n_sample, ).
        target (Tensor): Tensor of ground truth labels with shape of (n_samples, ).

    Returns:
        Tuple:
            - precision (Tensor): Tensor of precision result. Its shape is (n_thresholds + 1, )
            - recall (Tensor): Tensor of recall result. Its shape is (n_thresholds + 1, )
            - thresholds (Tensor): Tensor of threshold. Its shape is (n_thresholds, )

    Examples::

        >>> import torch
        >>> from torcheval.metrics.functional import binary_precision_recall_curve
        >>> input = torch.tensor([0.1, 0.5, 0.7, 0.8])
        >>> target = torch.tensor([0, 0, 1, 1])
        >>> binary_precision_recall_curve(input, target)
        (tensor([0.5000, 0.6667, 1.0000, 1.0000, 1.0000]),
        tensor([1.0000, 1.0000, 1.0000, 0.5000, 0.0000]),
        tensor([0.1000, 0.5000, 0.7000, 0.8000]))
    """
    _binary_precision_recall_curve_update(input, target)
    return _binary_precision_recall_curve_compute(input, target)


def _binary_precision_recall_curve_update(
    input: torch.Tensor,
    target: torch.Tensor,
) -> None:
    _binary_precision_recall_curve_update_input_check(input, target)


def _binary_precision_recall_curve_compute(
    input: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    precision, recall, threshold = _compute_for_each_class(input, target, 1, weights)

    return precision, recall, threshold


def _binary_precision_recall_curve_update_input_check(
    input: torch.Tensor,
    target: torch.Tensor,
) -> None:
    if input.ndim != 1:
        raise ValueError(
            "input should be a one-dimensional tensor, " f"got shape {input.shape}."
        )

    if target.ndim != 1:
        raise ValueError(
            "target should be a one-dimensional tensor, " f"got shape {target.shape}."
        )

    if input.shape != target.shape:
        raise ValueError(
            "The `input` and `target` should have the same shape, "
            f"got shapes {input.shape} and {target.shape}."
        )


@torch.inference_mode()
def multiclass_precision_recall_curve(
    input: torch.Tensor,
    target: torch.Tensor,
    *,
    num_classes: int | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """
    Returns precision-recall pairs and their corresponding thresholds for
    multi-class classification tasks. If a class is missing from the target
    tensor, its recall values are set to 1.0.

    Its class version is :obj:`torcheval.metrics.MulticlassPrecisionRecallCurve`.
    See also :func:`binary_precision_recall_curve <torcheval.metrics.functional.binary_precision_recall_curve>`, :func:`multilabel_precision_recall_curve <torcheval.metrics.functional.multilabel_precision_recall_curve>`

    Args:
        input (Tensor): Tensor of label predictions
            It should be probabilities or logits with shape of (n_sample, n_class).
        target (Tensor): Tensor of ground truth labels with shape of (n_samples, ).
        num_classes (Optional):
            Number of classes. Set to the second dimension of the input if num_classes is None.

    Return:
        a tuple of (precision: List[torch.Tensor], recall: List[torch.Tensor], thresholds: List[torch.Tensor])
            precision: List of precision result. Each index indicates the result of a class.
            recall: List of recall result. Each index indicates the result of a class.
            thresholds: List of threshold. Each index indicates the result of a class.

    Examples::

        >>> import torch
        >>> from torcheval.metrics.functional import multiclass_precision_recall_curve
        >>> input = torch.tensor([[0.1, 0.1, 0.1, 0.1], [0.5, 0.5, 0.5, 0.5], [0.7, 0.7, 0.7, 0.7], [0.8, 0.8, 0.8, 0.8]])
        >>> target = torch.tensor([0, 1, 2, 3])
        >>> multiclass_precision_recall_curve(input, target, num_classes=4)
        ([tensor([0.2500, 0.0000, 0.0000, 0.0000, 1.0000]),
        tensor([0.2500, 0.3333, 0.0000, 0.0000, 1.0000]),
        tensor([0.2500, 0.3333, 0.5000, 0.0000, 1.0000]),
        tensor([0.2500, 0.3333, 0.5000, 1.0000, 1.0000])],
        [tensor([1., 0., 0., 0., 0.]),
        tensor([1., 1., 0., 0., 0.]),
        tensor([1., 1., 1., 0., 0.]),
        tensor([1., 1., 1., 1., 0.])],
        [tensor([0.1000, 0.5000, 0.7000, 0.8000]),
        tensor([0.1000, 0.5000, 0.7000, 0.8000]),
        tensor([0.1000, 0.5000, 0.7000, 0.8000]),
        tensor([0.1000, 0.5000, 0.7000, 0.8000])])
    """
    if num_classes is None and input.ndim == 2:
        num_classes = input.shape[1]
    _multiclass_precision_recall_curve_update(input, target, num_classes)
    return _multiclass_precision_recall_curve_compute(input, target, num_classes)


def _multiclass_precision_recall_curve_update(
    input: torch.Tensor,
    target: torch.Tensor,
    num_classes: int | None,
) -> None:
    _multiclass_precision_recall_curve_update_input_check(input, target, num_classes)


@torch.jit.script
def _multiclass_precision_recall_curve_compute(
    input: torch.Tensor,
    target: torch.Tensor,
    num_classes: int | None,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    if num_classes is None:
        num_classes = input.shape[1]
    thresholds, indices = input.T.sort(dim=1, descending=True)
    mask = F.pad(thresholds.diff(dim=1) != 0, [0, 1], value=1.0).flip(1)
    sizes: list[int] = mask.sum(1).tolist()
    thresholds = thresholds.flip(1)[mask].split(sizes)

    arange = torch.arange(num_classes, device=target.device)
    cmp = target[indices] == arange[:, None]
    num_tp = cmp.cumsum(1).flip(1)
    num_fp = (~cmp).cumsum(1).flip(1)

    # The last precision and recall values are 1.0 and 0.0, respectively
    precision = F.pad(num_tp / (num_tp + num_fp), [0, 1], value=1.0)
    recall = F.pad((num_tp / num_tp[:, :1]).nan_to_num_(1.0), [0, 1], value=0.0)
    mask = F.pad(mask, [0, 1], value=1.0)
    sizes: list[int] = mask.sum(1).tolist()

    precision = precision[mask].split(sizes)
    recall = recall[mask].split(sizes)
    return list(precision), list(recall), list(thresholds)


def _multiclass_precision_recall_curve_update_input_check(
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

    if not (input.ndim == 2 and (num_classes is None or input.shape[1] == num_classes)):
        raise ValueError(
            f"input should have shape of (num_sample, num_classes), "
            f"got {input.shape} and num_classes={num_classes}."
        )


@torch.jit.script
def _compute_for_each_class(
    input: torch.Tensor,
    target: torch.Tensor,
    pos_label: int,
    weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    threshold, indices = input.sort(descending=True)
    mask = F.pad(threshold.diff(dim=0) != 0, [0, 1], value=1.0)

    if weights is not None:
        # Weighted version: compute weighted cumulative sums
        sorted_weights = weights[indices]
        is_positive = (target[indices] == pos_label).long()
        is_negative = 1 - is_positive

        # Compute weighted cumulative sums
        num_tp = (is_positive * sorted_weights).cumsum(0)[mask]
        num_fp = (is_negative * sorted_weights).cumsum(0)[mask]
    else:
        # Unweighted version: count occurrences
        num_tp = (target[indices] == pos_label).cumsum(0)[mask]
        num_fp = (1 - (target[indices] == pos_label).long()).cumsum(0)[mask]

    precision = (num_tp / (num_tp + num_fp)).flip(0)
    recall = (num_tp / num_tp[-1]).flip(0)
    threshold = threshold[mask].flip(0)

    # The last precision and recall values are 1.0 and 0.0 without a corresponding threshold.
    # This ensures that the graph starts on the y-axis.
    precision = torch.cat([precision, precision.new_ones(1)])
    recall = torch.cat([recall, recall.new_zeros(1)])

    # If recalls are NaNs, set NaNs to 1.0s.
    if torch.isnan(recall[0]):
        recall = torch.nan_to_num(recall, 1.0)

    return precision, recall, threshold


@torch.inference_mode()
def multilabel_precision_recall_curve(
    input: torch.Tensor,
    target: torch.Tensor,
    *,
    num_labels: int | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """
    Returns precision-recall pairs and their corresponding thresholds for
    multi-label classification tasks. If there are no samples for a label
    in the target tensor, its recall values are set to 1.0.

    Its class version is :obj:`torcheval.metrics.MultilabelPrecisionRecallCurve`.
    See also :func:`binary_precision_recall_curve <torcheval.metrics.functional.binary_precision_recall_curve>`, :func:`multiclass_precision_recall_curve <torcheval.metrics.functional.multiclass_precision_recall_curve>`

    Args:
        input (Tensor): Tensor of label predictions
            It should be probabilities or logits with shape of (n_sample, n_label).
        target (Tensor): Tensor of ground truth labels with shape of (n_samples, n_label).
        num_labels (Optional): Number of labels.

    Return:
        a tuple of (precision: List[torch.Tensor], recall: List[torch.Tensor], thresholds: List[torch.Tensor])
            precision: List of precision result. Each index indicates the result of a label.
            recall: List of recall result. Each index indicates the result of a label.
            thresholds: List of threshold. Each index indicates the result of a label.

    Examples::

        >>> import torch
        >>> from torcheval.metrics.functional import multilabel_precision_recall_curve
        >>> input = torch.tensor([[0.75, 0.05, 0.35], [0.45, 0.75, 0.05], [0.05, 0.55, 0.75], [0.05, 0.65, 0.05]])
        >>> target = torch.tensor([[1, 0, 1], [0, 0, 0], [0, 1, 1], [1, 1, 1]])
        >>> multilabel_precision_recall_curve(input, target, num_labels=3)
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
    if input.ndim != 2:
        raise ValueError(
            f"input should be a two-dimensional tensor, got shape {input.shape}."
        )
    if num_labels is None:
        num_labels = input.shape[1]
    _multilabel_precision_recall_curve_update(input, target, num_labels)
    return _multilabel_precision_recall_curve_compute(input, target, num_labels)


def _multilabel_precision_recall_curve_update(
    input: torch.Tensor,
    target: torch.Tensor,
    num_labels: int,
) -> None:
    _multilabel_precision_recall_curve_update_input_check(input, target, num_labels)


@torch.jit.script
def _multilabel_precision_recall_curve_compute(
    input: torch.Tensor,
    target: torch.Tensor,
    num_labels: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    precisions, recalls, thresholds = [], [], []
    for i in range(num_labels):
        precision, recall, threshold = _compute_for_each_class(
            input[:, i], target[:, i], 1
        )
        precisions.append(precision)
        recalls.append(recall)
        thresholds.append(threshold)
    return precisions, recalls, thresholds


def _multilabel_precision_recall_curve_update_input_check(
    input: torch.Tensor,
    target: torch.Tensor,
    num_labels: int,
) -> None:
    if input.shape != target.shape:
        raise ValueError(
            "Expected both input.shape and target.shape to have the same shape"
            f" but got {input.shape} and {target.shape}."
        )

    if input.ndim != 2:
        raise ValueError(
            f"input should be a two-dimensional tensor, got shape {input.shape}."
        )

    if input.shape[1] != num_labels:
        raise ValueError(
            f"input should have shape of (num_sample, num_labels), "
            f"got {input.shape} and num_labels={num_labels}."
        )
