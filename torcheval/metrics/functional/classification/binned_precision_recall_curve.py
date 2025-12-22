# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import torch
import torch.nn.functional as F
from torcheval.metrics.functional.classification.precision_recall_curve import (
    _binary_precision_recall_curve_update_input_check,
    _multiclass_precision_recall_curve_update_input_check,
    _multilabel_precision_recall_curve_update_input_check,
)
from torcheval.metrics.functional.tensor_utils import _create_threshold_tensor
from torcheval.utils.device import largest_float


@torch.inference_mode()
def binary_binned_precision_recall_curve(
    input: torch.Tensor,
    target: torch.Tensor,
    *,
    threshold: int | list[float] | torch.Tensor = 100,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute precision recall curve with given thresholds.
    Its class version is :obj:`torcheval.metrics.BinaryBinnedPrecisionRecallCurve`.
    See also :func:`multiclass_binned_precision_recall_curve <torcheval.metrics.functional.multiclass_binned_precision_recall_curve>`

    Args:
        input (Tensor): Tensor of label predictions
            It should be probabilities or logits with shape of (n_sample, ).
        target (Tensor): Tensor of ground truth labels with shape of (n_samples, ).
        threshold:
            a integer representing number of bins, a list of thresholds,
            or a tensor of thresholds.

    Returns:
        Tuple:
            - precision (Tensor): Tensor of precision result. Its shape is (n_thresholds + 1, )
            - recall (Tensor): Tensor of recall result. Its shape is (n_thresholds + 1, )
            - thresholds (Tensor): Tensor of threshold. Its shape is (n_thresholds, )

    Examples::

        >>> import torch
        >>> from torcheval.metrics.functional import binary_binned_precision_recall_curve
        >>> input = torch.tensor([0.2, 0.8, 0.5, 0.9])
        >>> target = torch.tensor([0, 1, 0, 1])
        >>> threshold = 5
        >>> binary_binned_precision_recall_curve(input, target, threshold)
        (tensor([0.5000, 0.6667, 0.6667, 1.0000, 1.0000, 1.0000]),
        tensor([1., 1., 1., 1., 0., 0.]),
        tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000]))

        >>> input = torch.tensor([0.2, 0.3, 0.4, 0.5])
        >>> target = torch.tensor([0, 0, 1, 1])
        >>> threshold = torch.tensor([0.0000, 0.2500, 0.7500, 1.0000])
        >>> binary_binned_precision_recall_curve(input, target, threshold)
        (tensor([0.5000, 0.6667, 1.0000, 1.0000, 1.0000]),
        tensor([1., 1., 0., 0., 0.]),
        tensor([0.0000, 0.2500, 0.7500, 1.0000]))
    """
    threshold = _create_threshold_tensor(threshold, target.device)
    _binned_precision_recall_curve_param_check(threshold)
    num_tp, num_fp, num_fn = _binary_binned_precision_recall_curve_update(
        input, target, threshold
    )
    return _binary_binned_precision_recall_curve_compute(
        num_tp, num_fp, num_fn, threshold
    )


def _binary_binned_precision_recall_curve_update(
    input: torch.Tensor,
    target: torch.Tensor,
    threshold: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _binary_precision_recall_curve_update_input_check(input, target)
    return _update(input, target, threshold)


@torch.jit.script
def _update(
    input: torch.Tensor,
    target: torch.Tensor,
    threshold: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_thresholds = len(threshold)

    # (index, target) values, stored as 2 * index + target
    index_target = (
        2 * (torch.searchsorted(threshold, input, right=True) - 1) + target
    ).type(largest_float(target.device))
    hist = torch.histc(
        index_target, bins=2 * num_thresholds, min=0, max=2 * num_thresholds
    )
    # For each index j, Find the last index i such that input[j] >= threshold[i]

    # false positives are positives_idx[0], true positives are positives_idx[1]
    target_sum = target.sum()
    positives_idx = hist.reshape((num_thresholds, 2)).T.type(target_sum.dtype)

    # suffix sum: For each threshold index/("true/false" positives) combination,
    # find how many indices j such that input[j] >= threshold[i]
    suffix_total = positives_idx.flip(dims=(1,)).cumsum(dim=1).flip(dims=(1,))
    num_fp, num_tp = suffix_total[0], suffix_total[1]
    num_fn = target_sum - num_tp
    return num_tp, num_fp, num_fn


@torch.jit.script
def _binary_binned_precision_recall_curve_compute(
    num_tp: torch.Tensor,
    num_fp: torch.Tensor,
    num_fn: torch.Tensor,
    threshold: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Set precision to 1.0 if all predictions are zeros.
    precision = torch.nan_to_num(num_tp / (num_tp + num_fp), 1.0)
    recall = num_tp / (num_tp + num_fn)

    # The last precision and recall values are 1.0 and 0.0 without a corresponding threshold.
    # This ensures that the graph starts on the y-axis.
    precision = torch.cat([precision, precision.new_ones(1)], dim=0)
    recall = torch.cat([recall, recall.new_zeros(1)], dim=0)

    return precision, recall, threshold


@torch.inference_mode()
def multiclass_binned_precision_recall_curve(
    input: torch.Tensor,
    target: torch.Tensor,
    num_classes: int | None = None,
    threshold: int | list[float] | torch.Tensor = 100,
    optimization: str = "vectorized",
) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
    """
    Compute precision recall curve with given thresholds.
    Its class version is :obj:`torcheval.metrics.MulticlassBinnedPrecisionRecallCurve`.
    See also :func:`binary_binned_precision_recall_curve <torcheval.metrics.functional.binary_binned_precision_recall_curve>`

    Args:
        input (Tensor): Tensor of label predictions
            It should be probabilities or logits with shape of (n_sample, n_class).
        target (Tensor): Tensor of ground truth labels with shape of (n_samples, ).
        num_classes (Optional):
            Number of classes. Set to the second dimension of the input if num_classes is None.
        threshold (Tensor):
            a integer representing number of bins, a list of thresholds,
            or a tensor of thresholds.
        optimization (str):
            Choose the optimization to use. Accepted values: "vectorized" and "memory".
            The "vectorized" optimization makes more use of vectorization but uses more memory; the "memory" optimization uses less memory but takes more steps.
            Here are the tradeoffs between these two options:
            - "vectorized": consumes more memory but is faster on some hardware, e.g. modern GPUs.
            - "memory": consumes less memory but can be significantly slower on some hardware, e.g. modern GPUs
            Generally, on GPUs, the "vectorized" optimization requires more memory but is faster; the "memory" optimization requires less memory but is slower.
            On CPUs, the "memory" optimization is recommended in all cases; it uses less memory and is faster.

    Return:
        a tuple of (precision: List[torch.Tensor], recall: List[torch.Tensor], thresholds: torch.Tensor)
            precision: List of precision result. Each index indicates the result of a class.
            recall: List of recall result. Each index indicates the result of a class.
            thresholds: Tensor of threshold. The threshold is used for all classes.

    Examples::

        >>> import torch
        >>> from torcheval.metrics.functional import multiclass_binned_precision_recall_curve
        >>> input = torch.tensor([[0.1, 0.1, 0.1, 0.1], [0.5, 0.5, 0.5, 0.5], [0.7, 0.7, 0.7, 0.7], [0.8, 0.8, 0.8, 0.8]])
        >>> target = torch.tensor([0, 1, 2, 3])
        >>> multiclass_binned_precision_recall_curve(input, target, num_classes=4, threshold=5)
        ([tensor([0.2500, 0.0000, 0.0000, 0.0000, 1.0000, 1.0000]),
        tensor([0.2500, 0.3333, 0.3333, 0.0000, 1.0000, 1.0000]),
        tensor([0.2500, 0.3333, 0.3333, 0.0000, 1.0000, 1.0000]),
        tensor([0.2500, 0.3333, 0.3333, 1.0000, 1.0000, 1.0000])],
        [tensor([1., 0., 0., 0., 0., 0.]),
        tensor([1., 1., 1., 0., 0., 0.]),
        tensor([1., 1., 1., 0., 0., 0.]),
        tensor([1., 1., 1., 1., 0., 0.])],
        tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000]))

        >>> input = torch.tensor([[0.1, 0.1, 0.1, 0.1], [0.5, 0.5, 0.5, 0.5], [0.7, 0.7, 0.7, 0.7], [0.8, 0.8, 0.8, 0.8]])
        >>> target = torch.tensor([0, 1, 2, 3])
        >>> threshold = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        >>> multiclass_binned_precision_recall_curve(input, target, num_classes=4, threshold=threshold)
        ([tensor([0.2500, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 1.0000]),
        tensor([0.2500, 0.3333, 0.3333, 0.3333, 0.3333, 0.0000, 0.0000, 0.0000, 1.0000, 1.0000]),
        tensor([0.2500, 0.3333, 0.3333, 0.3333, 0.3333, 0.5000, 0.5000, 0.0000, 1.0000, 1.0000]),
        tensor([0.2500, 0.3333, 0.3333, 0.3333, 0.3333, 0.5000, 0.5000, 1.0000, 1.0000, 1.0000])],
        [tensor([1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
        tensor([1., 1., 1., 1., 1., 0., 0., 0., 0., 0.]),
        tensor([1., 1., 1., 1., 1., 1., 1., 0., 0., 0.]),
        tensor([1., 1., 1., 1., 1., 1., 1., 1., 0., 0.])],
        tensor([0.1000, 0.2000, 0.3000, 0.4000, 0.5000, 0.6000, 0.7000, 0.8000, 0.9000]))
    """
    _optimization_param_check(optimization)
    threshold = _create_threshold_tensor(threshold, target.device)
    _binned_precision_recall_curve_param_check(threshold)

    if num_classes is None and input.ndim == 2:
        num_classes = input.shape[1]
    num_tp, num_fp, num_fn = _multiclass_binned_precision_recall_curve_update(
        input, target, num_classes, threshold, optimization
    )
    return _multiclass_binned_precision_recall_curve_compute(
        num_tp, num_fp, num_fn, num_classes, threshold
    )


def _multiclass_binned_precision_recall_curve_update_vectorized(
    input: torch.Tensor,
    target: torch.Tensor,
    num_classes: int | None,
    threshold: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Vectorized version for the update function on multiclass binned precision curve.
    This function uses O(num_thresholds * num_samples * num_classes) memory but compared with `_multiclass_binned_precision_recall_curve_update_memory`
    is faster *on GPU* due to more effective vectorization during computation.

    On GPU, this is recommended if time is more critical than memory.
    Note that this is still slower on CPU, so it is not recommended to use this function on CPU.
    """
    _multiclass_precision_recall_curve_update_input_check(input, target, num_classes)
    labels = input >= threshold[:, None, None]
    # pyre-fixme[6]: For 2nd argument expected `int` but got `Optional[int]`.
    target = F.one_hot(target, num_classes)
    num_tp = (labels & target).sum(dim=1)
    num_fp = labels.sum(dim=1) - num_tp
    num_fn = target.sum(dim=0) - num_tp

    return num_tp, num_fp, num_fn


def _multiclass_binned_precision_recall_curve_update_memory(
    input: torch.Tensor,
    target: torch.Tensor,
    num_classes: int | None,
    threshold: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Memory-optimized version for the update function on multiclass binned precision curve.
    This function uses O(num_samples * num_classes) memory but is slower *on GPU*, as
    `_multiclass_binned_precision_recall_curve_update_vectorized` uses more effective vectorization.
    However, this function is faster on CPU.

    This version is recommended for CPU in all cases, and for GPU if memory usage is more critical than time.
    """
    _multiclass_precision_recall_curve_update_input_check(input, target, num_classes)

    num_samples, num_classes = tuple(input.shape)
    num_thresholds = len(threshold)

    big_float = largest_float(target.device)
    # false positives are positives_idx[0], true positives are positives_idx[1]
    # For each j, k we find largest i such that input[j,k] >= threshold[i]. We also need to store whether target[j] == k.
    # largest_index: (index, class, target) values, stored as 2 * ((num_classes * index) + class) + target
    largest_index = (
        2
        * (
            num_classes * (torch.searchsorted(threshold, input, right=True) - 1)
            + torch.arange(num_classes, device=input.device)
        )
    ).type(big_float)
    largest_index[range(num_samples), target] += 1

    hist = torch.histc(
        largest_index,
        bins=2 * num_thresholds * num_classes,
        min=0,
        max=2 * num_thresholds * num_classes,
    )
    positives_idx = (
        hist.reshape((num_thresholds, num_classes, 2))
        .transpose(0, 2)
        .type(target.dtype)
    )

    class_counts = torch.histc(
        target.type(big_float), bins=num_classes, min=0, max=num_classes
    ).type(target.dtype)

    # suffix sum: For each threshold index/("true/false" positives) combination,
    # find how many indices j such that input[j] >= threshold[i].
    suffix_total = positives_idx.flip(dims=(-1,)).cumsum(dim=-1).flip(dims=(-1,))
    num_fp, num_tp = suffix_total[0].T, suffix_total[1].T
    num_fn = class_counts[None, :] - num_tp
    return num_tp, num_fp, num_fn


def _multiclass_binned_precision_recall_curve_update(
    input: torch.Tensor,
    target: torch.Tensor,
    num_classes: int | None,
    threshold: torch.Tensor,
    optimization: str = "vectorized",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _optimization_param_check(optimization)
    if optimization == "vectorized":
        return _multiclass_binned_precision_recall_curve_update_vectorized(
            input, target, num_classes, threshold
        )
    else:
        return _multiclass_binned_precision_recall_curve_update_memory(
            input, target, num_classes, threshold
        )


def _multiclass_binned_precision_recall_curve_compute(
    num_tp: torch.Tensor,
    num_fp: torch.Tensor,
    num_fn: torch.Tensor,
    num_classes: int | None,
    threshold: torch.Tensor,
) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
    # Set precision to 1.0 if all predictions are zeros.
    precision = torch.nan_to_num(num_tp / (num_tp + num_fp), 1.0)
    recall = num_tp / (num_tp + num_fn)

    # The last precision and recall values are 1.0 and 0.0 without a corresponding threshold.
    # This ensures that the graph starts on the y-axis.
    assert isinstance(num_classes, int)
    precision = torch.cat([precision, precision.new_ones(1, num_classes)], dim=0)
    recall = torch.cat([recall, recall.new_zeros(1, num_classes)], dim=0)

    return (
        list(precision.T),
        list(recall.T),
        threshold,
    )


@torch.inference_mode()
def multilabel_binned_precision_recall_curve(
    input: torch.Tensor,
    target: torch.Tensor,
    num_labels: int | None = None,
    threshold: int | list[float] | torch.Tensor = 100,
    optimization: str = "vectorized",
) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
    """
    Compute precision recall curve with given thresholds.
    Its class version is :obj:`torcheval.metrics.MultilabelBinnedPrecisionRecallCurve`.
    See also :func:`binary_binned_precision_recall_curve <torcheval.metrics.functional.binary_binned_precision_recall_curve>`,
    :func:`multiclass_precision_recall_curve <torcheval.metrics.functional.multiclass_precision_recall_curve>`

    Args:
        input (Tensor): Tensor of label predictions
            It should be probabilities or logits with shape of (n_sample, n_label).
        target (Tensor): Tensor of ground truth labels with shape of (n_samples, n_label).
        num_labels (Optional): Number of labels.
        threshold:
            a integer representing number of bins, a list of thresholds,
            or a tensor of thresholds.
        optimization (str):
            Choose the optimization to use. Accepted values: "vectorized" and "memory".
            The "vectorized" optimization makes more use of vectorization but uses more memory; the "memory" optimization uses less memory but takes more steps.
            Here are the tradeoffs between these two options:
            - "vectorized": consumes more memory but is faster on some hardware, e.g. modern GPUs.
            - "memory": consumes less memory but can be significantly slower on some hardware, e.g. modern GPUs
            Generally, on GPUs, the "vectorized" optimization requires more memory but is faster; the "memory" optimization requires less memory but is slower.
            On CPUs, the "memory" optimization is recommended in all cases; it uses less memory and is faster.

    Return:
        a tuple of (precision: List[torch.Tensor], recall: List[torch.Tensor], thresholds: List[torch.Tensor])
            precision: List of precision result. Each index indicates the result of a label.
            recall: List of recall result. Each index indicates the result of a label.
            thresholds: List of threshold. Each index indicates the result of a label.

    Examples::

        >>> import torch
        >>> from torcheval.metrics.functional import multilabel_binned_precision_recall_curve
        >>> input = torch.tensor([[0.75, 0.05, 0.35], [0.45, 0.75, 0.05], [0.05, 0.55, 0.75], [0.05, 0.65, 0.05]])
        >>> target = torch.tensor([[1, 0, 1], [0, 0, 0], [0, 1, 1], [1, 1, 1]])
        >>> multilabel_binned_precision_recall_curve(input, target, num_labels=3, thresholds=5)
        (tensor([[0.5000, 0.5000, 1.0000, 1.0000, 0.0000, 1.0000],
                 [0.5000, 0.6667, 0.6667, 0.0000, 0.0000, 1.0000],
                 [0.7500, 1.0000, 1.0000, 1.0000, 0.0000, 1.0000]]),
         tensor([[1.0000, 0.5000, 0.5000, 0.5000, 0.0000, 0.0000],
                 [1.0000, 1.0000, 1.0000, 0.0000, 0.0000, 0.0000],
                 [1.0000, 0.6667, 0.3333, 0.3333, 0.0000, 0.0000]]),
         tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000]))
    """
    _optimization_param_check(optimization)
    threshold = _create_threshold_tensor(threshold, target.device)
    _binned_precision_recall_curve_param_check(threshold)

    if input.ndim != 2:
        raise ValueError(
            f"input should be a two-dimensional tensor, got shape {input.shape}."
        )
    if num_labels is None:
        num_labels = input.shape[1]
    num_tp, num_fp, num_fn = _multilabel_binned_precision_recall_curve_update(
        input, target, num_labels, threshold, optimization
    )
    return _multilabel_binned_precision_recall_curve_compute(
        num_tp, num_fp, num_fn, num_labels, threshold
    )


def _multilabel_binned_precision_recall_curve_update_vectorized(
    input: torch.Tensor,
    target: torch.Tensor,
    num_labels: int,
    threshold: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Vectorized version for the update function on multilabel binned precision curve.
    This function uses O(num_thresholds * num_samples * num_labels) memory but compared with `_multilabel_binned_precision_recall_curve_update_memory`
    is faster *on GPU* due to more effective vectorization during computation.

    On GPU, this is recommended if time is more critical than memory.
    Note that this is still slower on CPU, so it is not recommended to use this function on CPU.
    """
    _multilabel_precision_recall_curve_update_input_check(input, target, num_labels)
    labels = input >= threshold[:, None, None]
    try:
        num_tp = (labels & target).sum(dim=1)
    except RuntimeError:
        # target could be a floating-point tensor
        num_tp = (labels & target.bool()).sum(dim=1)

    num_fp = labels.sum(dim=1) - num_tp
    num_fn = target.sum(dim=0) - num_tp

    return num_tp, num_fp, num_fn


def _multilabel_binned_precision_recall_curve_update_memory(
    input: torch.Tensor,
    target: torch.Tensor,
    num_labels: int,
    threshold: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Memory-optimized version for the update function on multilabel binned precision-recall curve.
    This function uses O(num_samples * num_labels) memory but is slower *on GPU*, as
    `_multilabel_binned_precision_recall_curve_update_optimize_speed` uses more effective vectorization.
    However, this function is faster on CPU.

    This version is recommended for CPU in all cases, and for GPU if memory usage is more critical than time.
    """
    _multilabel_precision_recall_curve_update_input_check(input, target, num_labels)

    _, num_labels = tuple(input.shape)
    num_thresholds = len(threshold)

    # For each sample index j and label k, we need:
    # (largest_threshold_index, k, target[j, k])
    # where largest_threshold_index is largest i such that input[j,k] >= threshold[i].
    # We flatten this tuple into a single value.
    largest_index = (
        2
        * (
            num_labels * (torch.searchsorted(threshold, input, right=True) - 1)
            + torch.arange(num_labels, device=input.device, dtype=torch.int64)
        )
        + target
    )

    hist = torch.histc(
        largest_index.type(largest_float(target.device)),
        bins=2 * num_thresholds * num_labels,
        min=0,
        max=2 * num_thresholds * num_labels,
    )
    class_counts = target.sum(dim=0)

    # false positives are positives_idx[0], true positives are positives_idx[1]
    positives_idx = (
        hist.reshape((num_thresholds, num_labels, 2))
        .transpose(0, 2)
        .type(class_counts.dtype)
    )

    # suffix sum: For each threshold index/("true/false" positives) combination,
    # find how many indices j such that input[j] >= threshold[i].
    suffix_total = positives_idx.flip(dims=(-1,)).cumsum(dim=-1).flip(dims=(-1,))
    num_fp, num_tp = suffix_total[0].T, suffix_total[1].T
    num_fn = class_counts[None, :] - num_tp
    return num_tp, num_fp, num_fn


def _multilabel_binned_precision_recall_curve_update(
    input: torch.Tensor,
    target: torch.Tensor,
    num_labels: int,
    threshold: torch.Tensor,
    optimization: str = "vectorized",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _optimization_param_check(optimization)
    if optimization == "vectorized":
        return _multilabel_binned_precision_recall_curve_update_vectorized(
            input, target, num_labels, threshold
        )
    else:
        return _multilabel_binned_precision_recall_curve_update_memory(
            input, target, num_labels, threshold
        )


@torch.jit.script
def _multilabel_binned_precision_recall_curve_compute(
    num_tp: torch.Tensor,
    num_fp: torch.Tensor,
    num_fn: torch.Tensor,
    num_labels: int,
    threshold: torch.Tensor,
) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
    # Set precision to 1.0 if all predictions are zeros.
    precision = torch.nan_to_num(num_tp / (num_tp + num_fp), 1.0)
    recall = num_tp / (num_tp + num_fn)

    # The last precision and recall values are 1.0 and 0.0 without a corresponding threshold.
    # This ensures that the graph starts on the y-axis.
    assert isinstance(num_labels, int)
    precision = torch.cat([precision, precision.new_ones(1, num_labels)], dim=0)
    recall = torch.cat([recall, recall.new_zeros(1, num_labels)], dim=0)

    return (
        list(precision.T),
        list(recall.T),
        threshold,
    )


def _binned_precision_recall_curve_param_check(
    threshold: torch.Tensor,
) -> None:
    if (torch.diff(threshold) < 0.0).any():
        raise ValueError("The `threshold` should be a sorted tensor.")

    if (threshold < 0.0).any() or (threshold > 1.0).any():
        raise ValueError("The values in `threshold` should be in the range of [0, 1].")


def _optimization_param_check(
    optimization: str,
) -> None:
    if optimization not in ("vectorized", "memory"):
        raise ValueError(
            f"Unknown memory approach: expected 'vectorized' or 'memory', but got {optimization}."
        )
