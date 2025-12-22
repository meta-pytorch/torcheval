# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import torch
from torch.nn import functional as F
from torcheval.metrics.functional.tensor_utils import _create_threshold_tensor
from torcheval.utils.device import largest_float

DEFAULT_NUM_THRESHOLD = 200


@torch.inference_mode()
def binary_binned_auroc(
    input: torch.Tensor,
    target: torch.Tensor,
    *,
    num_tasks: int = 1,
    threshold: int | list[float] | torch.Tensor = DEFAULT_NUM_THRESHOLD,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute AUROC, which is the area under the ROC Curve, for binary classification.
    Its class version is :obj:`torcheval.metrics.BinaryBinnedAUROC`.
    See also :func:`multiclass_binned_auroc <torcheval.metrics.functional.multiclass_binned_auroc>`

    Args:
        input (Tensor): Tensor of label predictions
            It should be predicted label, probabilities or logits with shape of (num_tasks, n_sample) or (n_sample, ).
        target (Tensor): Tensor of ground truth labels with shape of (num_tasks, n_sample) or (n_sample, ).
        num_tasks (int):  Number of tasks that need binary_binned_auroc calculation. Default value
                    is 1. binary_binned_auroc for each task will be calculated independently.
        threshold: A integer representing number of bins, a list of thresholds, or a tensor of thresholds.
                    The same thresholds will be used for all tasks.
                    If `threshold` is a tensor, it must be 1D.
                    If list or tensor is given, the first element must be 0 and the last must be 1.

    Examples::

        >>> import torch
        >>> from torcheval.metrics.functional import binary_binned_auroc
        >>> input = torch.tensor([0.1, 0.5, 0.7, 0.8])
        >>> target = torch.tensor([1, 0, 1, 1])
        >>> binary_binned_auroc(input, target, threshold=5)
        (tensor(0.5)
        tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000]))

        >>> input = torch.tensor([0.1, 0.5, 0.7, 0.8])
        >>> target = torch.tensor([1, 0, 1, 1])
        >>> threshold = tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000])
        >>> binary_binned_auroc(input, target, threshold=threshold)
        (tensor(0.5)
        tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000]))

        >>> input = torch.tensor([[1, 1, 1, 0], [0.1, 0.5, 0.7, 0.8]])
        >>> target = torch.tensor([[1, 0, 1, 0], [1, 0, 1, 1]])
        >>> binary_binned_auroc(input, target, num_tasks=2, threshold=5)
        (tensor([0.7500, 0.5000],
        tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000]))
    """
    threshold = _create_threshold_tensor(
        threshold,
        target.device,
    )
    _binary_binned_auroc_param_check(num_tasks, threshold)
    _binary_binned_auroc_update_input_check(input, target, num_tasks, threshold)
    return _binary_binned_auroc_compute(input, target, threshold)


def _binary_binned_auroc_param_check(
    num_tasks: int,
    threshold: torch.Tensor,
) -> None:
    if num_tasks < 1:
        raise ValueError("`num_tasks` has to be at least 1.")
    if (torch.diff(threshold) < 0.0).any():
        raise ValueError("The `threshold` should be a sorted tensor.")

    if (threshold < 0.0).any() or (threshold > 1.0).any():
        raise ValueError("The values in `threshold` should be in the range of [0, 1].")


def _binary_binned_auroc_update_input_check(
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
    if len(input.shape) > 2:
        raise ValueError(
            f"`input` is expected to be two dimensions or less, but got {len(input.shape)}D tensor."
        )
    if num_tasks == 1:
        if len(input.shape) > 1:
            raise ValueError(
                f"`num_tasks = 1`, `input` is expected to be one-dimensional tensor, but got shape {input.shape}."
            )
    elif len(input.shape) == 1 or input.shape[0] != num_tasks:
        raise ValueError(
            f"`num_tasks = {num_tasks}`, `input`'s shape is expected to be ({num_tasks}, num_samples), but got shape ({input.shape})."
        )


@torch.jit.script
def _binary_binned_auroc_compute(
    input: torch.Tensor,
    target: torch.Tensor,
    threshold: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    pred_label = input >= threshold[:, None, None]
    input_target = pred_label * target

    cum_tp = F.pad(input_target.sum(dim=-1).rot90(1, [1, 0]), (1, 0), value=0.0)
    cum_fp = F.pad(
        (pred_label.sum(dim=-1) - input_target.sum(dim=-1)).rot90(1, [1, 0]),
        (1, 0),
        value=0.0,
    )

    if len(cum_tp.shape) > 1:
        factor = cum_tp[:, -1] * cum_fp[:, -1]
    else:
        factor = cum_tp[-1] * cum_fp[-1]
    # Set AUROC to 0.5 when the target contains all ones or all zeros.
    auroc = torch.where(
        factor == 0,
        0.5,
        torch.trapz(cum_tp, cum_fp).type(largest_float(target.device)) / factor,
    )
    return auroc, threshold


@torch.inference_mode()
def multiclass_binned_auroc(
    input: torch.Tensor,
    target: torch.Tensor,
    *,
    num_classes: int,
    threshold: int | list[float] | torch.Tensor = DEFAULT_NUM_THRESHOLD,
    average: str | None = "macro",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute AUROC, which is the area under the ROC Curve, for multiclass classification.
    Its class version is :obj:`torcheval.metrics.MulticlassAUROC`.
    See also :func:`binary_binned_auroc <torcheval.metrics.functional.binary_binned_auroc>`

    Args:
        input (Tensor): Tensor of label predictions
            It should be probabilities or logits with shape of (n_sample, n_class).
        target (Tensor): Tensor of ground truth labels with shape of (n_samples, ).
        num_classes (int): Number of classes.
        threshold: A integer representing number of bins, a list of thresholds, or a tensor of thresholds.
        average (str, optional):
            - ``'macro'`` [default]:
                Calculate metrics for each class separately, and return their unweighted mean.
            - ``None``:
                Calculate the metric for each class separately, and return
                the metric for every class.

    Examples::
        >>> import torch
        >>> from torcheval.metrics.functional import multiclass_binned_auroc
        >>> input = torch.tensor([[0.1, 0.2, 0.1], [0.4, 0.2, 0.1], [0.6, 0.1, 0.2], [0.4, 0.2, 0.3], [0.6, 0.2, 0.4]])
        >>> target = torch.tensor([0, 1, 2, 1, 0])
        >>> multiclass_binned_auroc(input, target, num_classes=3, threshold=5)
        tensor(0.4000)
        >>> multiclass_binned_auroc(input, target, num_classes=3, threshold=5, average=None)
        tensor([0.5000, 0.2500, 0.2500, 0.0000, 1.0000])
    """
    threshold = _create_threshold_tensor(
        threshold,
        target.device,
    )
    _multiclass_binned_auroc_param_check(num_classes, threshold, average)
    _multiclass_binned_auroc_update_input_check(input, target, num_classes)
    return _multiclass_binned_auroc_compute(
        input, target, num_classes, threshold, average
    )


@torch.jit.script
def _multiclass_binned_auroc_compute(
    input: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    threshold: torch.Tensor,
    average: str | None = "macro",
) -> tuple[torch.Tensor, torch.Tensor]:
    pred_label = input >= threshold[:, None, None]
    target = F.one_hot(target, num_classes)
    input_target = pred_label * target

    cum_tp = F.pad(input_target.sum(dim=-1).rot90(1, [1, 0]), (1, 0), value=0.0)
    cum_fp = F.pad(
        (pred_label.sum(dim=-1) - input_target.sum(dim=-1)).rot90(1, [1, 0]),
        (1, 0),
        value=0.0,
    )

    factor = cum_tp[:, -1] * cum_fp[:, -1]
    auroc = torch.where(
        factor == 0, 0.5, torch.trapezoid(cum_tp, cum_fp, dim=1) / factor
    )
    if isinstance(average, str) and average == "macro":
        return auroc.mean(), threshold
    return auroc, threshold


def _multiclass_binned_auroc_param_check(
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

    if (torch.diff(threshold) < 0.0).any():
        raise ValueError("The `threshold` should be a sorted tensor.")

    if (threshold < 0.0).any() or (threshold > 1.0).any():
        raise ValueError("The values in `threshold` should be in the range of [0, 1].")


def _multiclass_binned_auroc_update_input_check(
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
