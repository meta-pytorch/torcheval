# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import torch
from torcheval.metrics.functional.classification.binned_precision_recall_curve import (
    _binary_binned_precision_recall_curve_compute,
    _binary_binned_precision_recall_curve_update,
    _multiclass_binned_precision_recall_curve_compute,
    _multiclass_binned_precision_recall_curve_update,
    _multilabel_binned_precision_recall_curve_compute,
    _multilabel_binned_precision_recall_curve_update,
    _optimization_param_check,
)
from torcheval.metrics.functional.tensor_utils import (
    _create_threshold_tensor,
    _riemann_integral,
)

DEFAULT_NUM_THRESHOLD = 100


@torch.inference_mode()
def binary_binned_auprc(
    input: torch.Tensor,
    target: torch.Tensor,
    *,
    num_tasks: int = 1,
    threshold: int | list[float] | torch.Tensor = DEFAULT_NUM_THRESHOLD,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Binned Version of AUPRC, which is the area under the AUPRC Curve, for binary classification.
    Its class version is :obj:`torcheval.metrics.BinaryBinnedAUPRC`.

    Computation is done by computing the area under the precision/recall curve; precision and recall
    are computed for the buckets defined by `threshold`.

    Args:
        input (Tensor): Tensor of label predictions
            It should be predicted label, probabilities or logits with shape of (num_tasks, n_sample) or (n_sample, ).

        target (Tensor): Tensor of ground truth labels with shape of (num_tasks, n_sample) or (n_sample, ).

        num_tasks (int):  Number of tasks that need binary_binned_auprc calculation. Default value
                    is 1. binary_binned_auprc for each task will be calculated independently.

        threshold (Tensor, int, List[float]): Either an integer representing the number of bins, a list of thresholds, or a tensor of thresholds.
                    The same thresholds will be used for all tasks.
                    If `threshold` is a tensor, it must be 1D.
                    If list or tensor is given, the first element must be 0 and the last must be 1.

    Examples:

        >>> import torch
        >>> from torcheval.metrics.functional import binary_binned_auprc
        >>> input = torch.tensor([0.2, 0.3, 0.4, 0.5])
        >>> target = torch.tensor([0, 0, 1, 1])
        >>> binary_binned_auprc(input, target, threshold=5)
        (tensor(1.0),
        tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000]))

        >>> input = torch.tensor([0.2, 0.3, 0.4, 0.5])
        >>> target = torch.tensor([0, 0, 1, 1])
        >>> threshold = torch.tensor([0.0000, 0.2500, 0.7500, 1.0000])
        >>> binary_binned_auprc(input, target, threshold=threshold)
        (tensor(0.6667),
        tensor([0.0000, 0.2500, 0.7500, 1.0000]))

        >>> input = torch.tensor([[0.2, 0.3, 0.4, 0.5], [0, 1, 2, 3]])
        >>> target = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]])
        >>> threshold = torch.tensor([0.0000, 0.2500, 0.7500, 1.0000])
        >>> binary_binned_auprc(input, target, num_tasks=2, threshold=threshold)
        (tensor([0.6667, 1.0000],
        tensor([0.0000, 0.2500, 0.7500, 1.0000]))
    """
    threshold = _create_threshold_tensor(threshold, target.device)
    _binary_binned_auprc_param_check(num_tasks, threshold)
    _binary_binned_auprc_update_input_check(input, target, num_tasks, threshold)
    return _binary_binned_auprc_compute(input, target, num_tasks, threshold), threshold


def _binary_binned_auprc_compute(
    input: torch.Tensor,
    target: torch.Tensor,
    num_tasks: int,
    threshold: torch.Tensor,
) -> torch.Tensor:
    if num_tasks == 1 and input.ndim == 1:
        num_tp, num_fp, num_fn = _binary_binned_precision_recall_curve_update(
            input, target, threshold
        )
        precision, recall, _ = _binary_binned_precision_recall_curve_compute(
            num_tp, num_fp, num_fn, threshold
        )
        auprc_result = _riemann_integral(recall, precision)
    else:
        auprcs = []
        for i in range(num_tasks):
            num_tp, num_fp, num_fn = _binary_binned_precision_recall_curve_update(
                input[i, :], target[i, :], threshold
            )
            precision, recall, _ = _binary_binned_precision_recall_curve_compute(
                num_tp, num_fp, num_fn, threshold
            )
            auprcs.append(_riemann_integral(recall, precision))
        auprc_result = torch.tensor(auprcs, device=input.device)
    auprc_result = torch.nan_to_num(auprc_result, nan=0.0)
    return auprc_result


def _binary_binned_auprc_param_check(
    num_tasks: int,
    threshold: torch.Tensor,
) -> None:
    if num_tasks < 1:
        raise ValueError("`num_tasks` has to be at least 1.")

    if threshold.ndim != 1:
        raise ValueError(
            f"`threshold` should be 1-dimensional, but got {threshold.ndim}D tensor."
        )

    if (torch.diff(threshold) < 0.0).any():
        raise ValueError("The `threshold` should be a sorted tensor.")

    if (threshold < 0.0).any() or (threshold > 1.0).any():
        raise ValueError("The values in `threshold` should be in the range of [0, 1].")

    if threshold[0] != 0:
        raise ValueError("First value in `threshold` should be 0.")

    if threshold[-1] != 1:
        raise ValueError("Last value in `threshold` should be 1.")


def _binary_binned_auprc_update_input_check(
    input: torch.Tensor,
    target: torch.Tensor,
    num_tasks: int,
    threshold: torch.Tensor,
) -> None:
    if input.shape != target.shape:
        raise ValueError(
            "The `input` and `target` should have the same shape, "
            f"got shapes {input.shape} and {target.shape}."
        )
    elif num_tasks == 1:
        # for num_tasks = 1, accept 1D or 2D tensor
        if input.ndim not in (1, 2):
            raise ValueError(
                f"`num_tasks = 1`, `input` is expected to be 1D or 2D tensor, but got shape {input.shape}."
            )
    else:
        # for num_tasks > 1, accept 2D tensor only, and the shape should be (num_tasks, num_samples)
        if input.ndim != 2:
            raise ValueError(
                f"`num_tasks = {num_tasks}`, `input` is expected to be 2D tensor, but got shape {input.shape}."
            )
        elif input.shape[0] != num_tasks:
            raise ValueError(
                f"`num_tasks = {num_tasks}`, `input`'s shape is expected to be ({num_tasks}, num_samples), but got shape {input.shape}."
            )


@torch.inference_mode()
def multiclass_binned_auprc(
    input: torch.Tensor,
    target: torch.Tensor,
    num_classes: int | None = None,
    *,
    threshold: int | list[float] | torch.Tensor = DEFAULT_NUM_THRESHOLD,
    average: str | None = "macro",
    optimization: str = "vectorized",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Binned Version of AUPRC, which is the area under the AUPRC Curve, for multiclass classification.
    Its class version is :obj:`torcheval.metrics.MulticlassBinnedAUPRC`.

    Computation is done by computing the area under the precision/recall curve; precision and recall
    are computed for the buckets defined by `threshold`.

    Args:
        input (Tensor): Tensor of label predictions
            It should be probabilities or logits with shape of (n_samples, n_classes).
        target (Tensor): Tensor of ground truth labels with shape of (n_samples, ).
        num_classes (int): Number of classes.
        threshold (Tensor, int, List[float]): Either an integer representing the number of bins, a list of thresholds, or a tensor of thresholds.
                    The same thresholds will be used for all tasks.
                    If `threshold` is a tensor, it must be 1D.
                    If list or tensor is given, the first element must be 0 and the last must be 1.
        average (str, optional):
            - ``'macro'`` [default]:
                Calculate metrics for each class separately, and return their unweighted mean.
            - ``None``:
                Calculate the metric for each class separately, and return
                the metric for every class.
        optimization (str):
            Choose the optimization to use. Accepted values: "vectorized" and "memory".
            The "vectorized" optimization makes more use of vectorization but uses more memory; the "memory" optimization uses less memory but takes more steps.
            Here are the tradeoffs between these two options:
            - "vectorized": consumes more memory but is faster on some hardware, e.g. modern GPUs.
            - "memory": consumes less memory but can be significantly slower on some hardware, e.g. modern GPUs
            Generally, on GPUs, the "vectorized" optimization requires more memory but is faster; the "memory" optimization requires less memory but is slower.
            On CPUs, the "memory" optimization is recommended in all cases; it uses less memory and is faster.

    Examples::

        >>> import torch
        >>> from torcheval.metrics.functional import multiclass_binned_auroc
        >>> input = torch.tensor([[0.1, 0.2, 0.1], [0.4, 0.2, 0.1], [0.6, 0.1, 0.2], [0.4, 0.2, 0.3], [0.6, 0.2, 0.4]])
        >>> target = torch.tensor([0, 1, 2, 1, 0])
        >>> multiclass_binned_auprc(input, target, num_classes=3, threshold=5, average='macro')
        (tensor(0.35), tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000]))
        >>> multiclass_binned_auprc(input, target, num_classes=3, threshold=5, average=None)
        (tensor([0.4500, 0.4000, 0.2000]),
        tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000]))
        >>> input = torch.tensor([[0.1, 0.2, 0.1, 0.4], [0.4, 0.2, 0.1, 0.7], [0.6, 0.1, 0.2, 0.4], [0.4, 0.2, 0.3, 0.2], [0.6, 0.2, 0.4, 0.5]])
        >>> target = torch.tensor([0, 1, 2, 1, 0])
        >>> threshold = torch.tensor([0.0, 0.1, 0.4, 0.7, 0.8, 1.0])
        >>> multiclass_binned_auprc(input, target, num_classes=4, threshold=threshold, average='macro')
        (tensor(0.24375),
        tensor([0.0, 0.1, 0.4, 0.7, 0.8, 1.0]))
        >>> multiclass_binned_auprc(input, target, num_classes=4, threshold=threshold, average=None)
        (tensor([0.3250, 0.2000, 0.2000, 0.2500]),
        tensor([0.0, 0.1, 0.4, 0.7, 0.8, 1.0]))
    """
    _optimization_param_check(optimization)
    if num_classes is None:
        num_classes = input.shape[1]
    threshold = _create_threshold_tensor(threshold, target.device)
    _multiclass_binned_auprc_param_check(num_classes, threshold, average)
    _multiclass_binned_auprc_update_input_check(input, target, num_classes)
    return (
        _multiclass_binned_auprc_compute(
            input, target, num_classes, threshold, average, optimization
        ),
        threshold,
    )


def _multiclass_binned_auprc_compute(
    input: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    threshold: torch.Tensor,
    average: str | None = "macro",
    optimization: str = "vectorized",
) -> torch.Tensor:
    num_tp, num_fp, num_fn = _multiclass_binned_precision_recall_curve_update(
        input, target, num_classes, threshold, optimization
    )
    prec, recall, thresh = _multiclass_binned_precision_recall_curve_compute(
        num_tp, num_fp, num_fn, num_classes, threshold
    )
    return _compute_riemann_integrals(prec, recall, average, input.device)


def _multiclass_binned_auprc_param_check(
    num_classes: int,
    threshold: torch.Tensor,
    average: str | None,
) -> None:
    average_options = ("macro", "none", None)
    if average not in average_options:
        raise ValueError(
            f"`average` was not in the allowed value of {average_options}, got {average}."
        )
    if num_classes < 2:
        raise ValueError("`num_classes` has to be at least 2.")

    if threshold.ndim != 1:
        raise ValueError(
            f"`threshold` should be 1-dimensional, but got {threshold.ndim}D tensor."
        )

    if (torch.diff(threshold) < 0.0).any():
        raise ValueError("The `threshold` should be a sorted tensor.")

    if (threshold < 0.0).any() or (threshold > 1.0).any():
        raise ValueError("The values in `threshold` should be in the range of [0, 1].")

    if threshold[0] != 0:
        raise ValueError("First value in `threshold` should be 0.")

    if threshold[-1] != 1:
        raise ValueError("Last value in `threshold` should be 1.")


def _multiclass_binned_auprc_update_input_check(
    input: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
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

    if not (input.ndim == 2 and input.shape[1] == num_classes):
        raise ValueError(
            f"input should have shape of (num_sample, num_classes), "
            f"got {input.shape} and num_classes={num_classes}."
        )


@torch.inference_mode()
def multilabel_binned_auprc(
    input: torch.Tensor,
    target: torch.Tensor,
    num_labels: int | None = None,
    *,
    threshold: int | list[float] | torch.Tensor = DEFAULT_NUM_THRESHOLD,
    average: str | None = "macro",
    optimization: str = "vectorized",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Binned Version of AUPRC, which is the area under the AUPRC Curve, for multilabel classification.
    Its class version is :obj:`torcheval.metrics.MultilabelBinnedAUPRC`.

    Computation is done by computing the area under the precision/recall curve; precision and recall
    are computed for the buckets defined by `threshold`.

    Args:
        input (Tensor): Tensor of label predictions
            It should be probabilities or logits with shape of (n_samples, n_labels).
        target (Tensor): Tensor of ground truth labels with shape of (n_samples, n_labels).
        num_labels (int, optional): Number of labels.
        threshold (Tensor, int, List[float]): Either an integer representing the number of bins, a list of thresholds, or a tensor of thresholds.
                    The same thresholds will be used for all tasks.
                    If `threshold` is a tensor, it must be 1D.
                    If list or tensor is given, the first element must be 0 and the last must be 1.
        average (str, optional):
            - ``'macro'`` [default]:
                Calculate metrics for each label separately, and return their unweighted mean.
            - ``None``:
                Calculate the metric for each label separately, and return
                the metric for every label.
        optimization (str):
            Choose the optimization to use. Accepted values: "vectorized" and "memory".
            The "vectorized" optimization makes more use of vectorization but uses more memory; the "memory" optimization uses less memory but takes more steps.
            Here are the tradeoffs between these two options:
            - "vectorized": consumes more memory but is faster on some hardware, e.g. modern GPUs.
            - "memory": consumes less memory but can be significantly slower on some hardware, e.g. modern GPUs
            Generally, on GPUs, the "vectorized" optimization requires more memory but is faster; the "memory" optimization requires less memory but is slower.
            On CPUs, the "memory" optimization is recommended in all cases; it uses less memory and is faster.

    Examples::

        >>> import torch
        >>> from torcheval.metrics.functional import multilabel_binned_auprc
        >>> input = torch.tensor([[0.75, 0.05, 0.35], [0.45, 0.75, 0.05], [0.05, 0.55, 0.75], [0.05, 0.65, 0.05]])
        >>> target = torch.tensor([[1, 0, 1], [0, 0, 0], [0, 1, 1], [1, 1, 1]])
        >>> multilabel_binned_auprc(input, target, num_labels=3, threshold=5, average='macro')
        (tensor([0.7500, 0.6667, 0.9167]),
        tensor([0.0000, 0.1000, 0.4000, 0.7000, 0.8000, 1.0000]))
        >>> threshold = torch.tensor([0.0, 0.1, 0.4, 0.7, 0.8, 1.0])
        >>> multilabel_binned_auprc(input, target, num_labels=3, threshold=threshold, average=None)
        tensor(0.7500)
        >>> multilabel_binned_auprc(input, target, num_labels=3, threshold=threshold, average='macro')
        tensor([0.7500, 0.5833, 0.9167])
    """
    _optimization_param_check(optimization)
    if num_labels is None:
        num_labels = input.shape[1]
    threshold = _create_threshold_tensor(threshold, target.device)
    _multilabel_binned_auprc_param_check(num_labels, threshold, average)
    _multilabel_binned_auprc_update_input_check(input, target, num_labels)
    return (
        _multilabel_binned_auprc_compute(
            input, target, num_labels, threshold, average, optimization
        ),
        threshold,
    )


def _multilabel_binned_auprc_compute(
    input: torch.Tensor,
    target: torch.Tensor,
    num_labels: int,
    threshold: torch.Tensor,
    average: str | None = "macro",
    optimization: str = "vectorized",
) -> torch.Tensor:
    num_tp, num_fp, num_fn = _multilabel_binned_precision_recall_curve_update(
        input, target, num_labels, threshold, optimization
    )
    prec, recall, thresh = _multilabel_binned_precision_recall_curve_compute(
        num_tp, num_fp, num_fn, num_labels, threshold
    )
    return _compute_riemann_integrals(prec, recall, average, input.device)


def _multilabel_binned_auprc_param_check(
    num_labels: int,
    threshold: torch.Tensor,
    average: str | None,
) -> None:
    average_options = ("macro", "none", None)
    if average not in average_options:
        raise ValueError(
            f"`average` was not in the allowed value of {average_options}, got {average}."
        )
    if num_labels < 2:
        raise ValueError("`num_labels` has to be at least 2.")
    if threshold.ndim != 1:
        raise ValueError(
            f"`threshold` should be 1-dimensional, but got {threshold.ndim}D tensor."
        )

    if (torch.diff(threshold) < 0.0).any():
        raise ValueError("The `threshold` should be a sorted tensor.")

    if (threshold < 0.0).any() or (threshold > 1.0).any():
        raise ValueError("The values in `threshold` should be in the range of [0, 1].")

    if threshold[0] != 0:
        raise ValueError("First value in `threshold` should be 0.")

    if threshold[-1] != 1:
        raise ValueError("Last value in `threshold` should be 1.")


def _multilabel_binned_auprc_update_input_check(
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


def _compute_riemann_integrals(
    prec: list[torch.Tensor],
    recall: list[torch.Tensor],
    average: str | None = "macro",
    device: torch.device | None = None,
) -> torch.Tensor:
    auprcs = []
    for p, r in zip(prec, recall):
        auprcs.append(_riemann_integral(r, p))
    auprcs = torch.tensor(auprcs, device=device).nan_to_num(nan=0.0)

    if average == "macro":
        return torch.mean(auprcs)
    else:
        return auprcs
