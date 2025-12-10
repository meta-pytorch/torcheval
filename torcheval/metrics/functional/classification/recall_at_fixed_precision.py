# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import torch
from torcheval.metrics.functional.classification.precision_recall_curve import (
    _binary_precision_recall_curve_compute,
    _binary_precision_recall_curve_update_input_check,
    _multilabel_precision_recall_curve_compute,
    _multilabel_precision_recall_curve_update_input_check,
)

"""
This file contains binary_recall_at_fixed_precision and multilabel_recall_at_fixed_precision functions.
"""


@torch.inference_mode()
def binary_recall_at_fixed_precision(
    input: torch.Tensor, target: torch.Tensor, *, min_precision: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns the highest possible recall value given the minimum precision
    for binary classification tasks.

    Its class version is ``torcheval.metrics.BinaryRecallAtFixedPrecision``.
    See also :func:`multilabel_recall_at_fixed_precision <torcheval.metrics.functional.multilabel_recall_at_fixed_precision>`

    Args:
        input (Tensor): Tensor of label predictions
            It should be probabilities with shape of (n_samples, )
        target (Tensor): Tensor of ground truth labels with shape of (n_samples, )
        min_precision (float): Minimum precision threshold

    Return:
        Tuple:
            - recall (Tensor): Max recall value given the minimum precision
            - thresholds (Tensor): Corresponding threshold to max recall

    Examples::

        >>> import torch
        >>> from torcheval.metrics.functional import binary_recall_at_fixed_precision
        >>> input = torch.tensor([0.1, 0.4, 0.6, 0.6, 0.6, 0.35, 0.8])
        >>> target = torch.tensor([0, 0, 1, 1, 1, 1, 1])
        >>> binary_recall_at_fixed_precision(input, target, min_precision=0.5)
        (torch.tensor(1.0), torch.tensor(0.35))
    """
    _binary_recall_at_fixed_precision_update_input_check(input, target, min_precision)
    return _binary_recall_at_fixed_precision_compute(input, target, min_precision)


def _binary_recall_at_fixed_precision_update_input_check(
    input: torch.Tensor, target: torch.Tensor, min_precision: float
) -> None:
    _binary_precision_recall_curve_update_input_check(input, target)
    if not isinstance(min_precision, float) or not (0 <= min_precision <= 1):
        raise ValueError(
            "Expected min_precision to be a float in the [0, 1] range"
            f" but got {min_precision}."
        )


def _binary_recall_at_fixed_precision_compute(
    input: torch.Tensor,
    target: torch.Tensor,
    min_precision: float,
    weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    precision, recall, threshold = _binary_precision_recall_curve_compute(
        input, target, weights
    )
    return _recall_at_precision(precision, recall, threshold, min_precision)


@torch.inference_mode()
def multilabel_recall_at_fixed_precision(
    input: torch.Tensor, target: torch.Tensor, *, num_labels: int, min_precision: float
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Returns the highest possible recall value give the minimum precision
    for each label and their corresponding thresholds for multi-label
    classification tasks. The maximum recall computation for each label is
    equivalent to _binary_recall_at_fixed_precision_compute in binary_recall_at_fixed_precision.

    Its class version is ``torcheval.metrics.MultilabelRecallAtFixedPrecision``.
    See also :func:`binary_recall_at_fixed_precision <torcheval.metrics.functional.binary_recall_at_fixed_precision>`

    Args:
        input (Tensor): Tensor of label predictions
            It should be probabilities with shape of (n_samples, n_label)
        target (Tensor): Tensor of ground truth labels with shape of (n_samples, n_label)
        num_labels (int): Number of labels
        min_precision (float): Minimum precision threshold

    Return:
        a tuple of (recall: List[torch.Tensor], thresholds: List[torch.Tensor])
            recall: List of max recall values for each label
            thresholds: List of best threshold values for each label

    Examples:

        >>> import torch
        >>> from torcheval.metrics.functional import multilabel_recall_at_fixed_precision
        >>> input = torch.tensor([[0.75, 0.05, 0.35], [0.45, 0.75, 0.05], [0.05, 0.55, 0.75], [0.05, 0.65, 0.05]])
        >>> target = torch.tensor([[1, 0, 1], [0, 0, 0], [0, 1, 1], [1, 1, 1]])
        >>> multilabel_recall_at_fixed_precision(input, target, num_labels=3, min_precision=0.5)
        ([tensor([1.0, 1.0, 1.0], tensor([0.05, 0.55, 0.05])])
    """
    if num_labels is None and input.ndim == 2:
        num_labels = input.shape[1]
    _multilabel_recall_at_fixed_precision_update_input_check(
        input, target, num_labels, min_precision
    )
    return _multilabel_recall_at_fixed_precision_compute(
        input, target, num_labels, min_precision
    )


def _multilabel_recall_at_fixed_precision_update_input_check(
    input: torch.Tensor, target: torch.Tensor, num_labels: int, min_precision: float
) -> None:
    _multilabel_precision_recall_curve_update_input_check(input, target, num_labels)
    if not isinstance(min_precision, float) or not (0 <= min_precision <= 1):
        raise ValueError(
            "Expected min_precision to be a float in the [0, 1] range"
            f" but got {min_precision}."
        )


@torch.jit.script
def _recall_at_precision(
    precision: torch.Tensor,
    recall: torch.Tensor,
    thresholds: torch.Tensor,
    min_precision: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_recall = torch.max(recall[precision >= min_precision])
    thresholds = torch.cat((thresholds, torch.tensor([-1.0], device=thresholds.device)))
    best_threshold = torch.max(thresholds[recall == max_recall])
    return max_recall, torch.abs(best_threshold)


@torch.jit.script
def _multilabel_recall_at_fixed_precision_compute(
    input: torch.Tensor, target: torch.Tensor, num_labels: int, min_precision: float
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    precision, recall, thresholds = _multilabel_precision_recall_curve_compute(
        input, target, num_labels
    )
    max_recall, best_threshold = [], []
    for p, r, t in zip(precision, recall, thresholds):
        max_r, best_t = _recall_at_precision(p, r, t, min_precision)
        max_recall.append(max_r)
        best_threshold.append(best_t)
    return max_recall, best_threshold
