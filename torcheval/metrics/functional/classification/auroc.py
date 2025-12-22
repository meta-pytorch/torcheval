# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import torch
from torch.nn import functional as F

# Optionally import fbgemm_gpu to enable use of hand fused kernel
try:
    import fbgemm_gpu.metrics
except ImportError:
    pass

try:
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:metric_ops")
except OSError:
    pass

from torcheval.utils.device import largest_float


@torch.inference_mode()
def binary_auroc(
    input: torch.Tensor,
    target: torch.Tensor,
    *,
    num_tasks: int = 1,
    weight: torch.Tensor | None = None,
    use_fbgemm: bool | None = False,
) -> torch.Tensor:
    """
    Compute AUROC, which is the area under the ROC Curve, for binary classification.
    Its class version is :obj:`torcheval.metrics.BinaryAUROC`.
    See also :func:`multiclass_auroc <torcheval.metrics.functional.multiclass_auroc>`

    Args:
        input (Tensor): Tensor of label predictions
            It should be predicted label, probabilities or logits with shape of (num_tasks, n_sample) or (n_sample, ).
        target (Tensor): Tensor of ground truth labels with shape of (num_tasks, n_sample) or (n_sample, ).
        num_tasks (int):  Number of tasks that need BinaryAUROC calculation. Default value
                    is 1. BinaryAUROC for each task will be calculated independently.
        weight (Tensor): Optional. A manual rescaling weight to match input tensor shape (num_tasks, num_samples) or (n_sample, ).
        use_fbgemm (bool): Optional. If set to True, use ``fbgemm_gpu.metrics.auc`` (a
            hand fused kernel).  FBGEMM AUC is an approximation of AUC. It does
            not mask data in case that input values are redundant. For the
            highly redundant input case, FBGEMM AUC can give a significantly
            different result.

    Examples::

        >>> import torch
        >>> from torcheval.metrics.functional import binary_auroc
        >>> input = torch.tensor([0.1, 0.5, 0.7, 0.8])
        >>> target = torch.tensor([1, 0, 1, 1])
        >>> binary_auroc(input, target)
        tensor(0.6667)

        >>> input = torch.tensor([1, 1, 1, 0])
        >>> target = torch.tensor([1, 0, 1, 0])
        >>> binary_auroc(input, target)
        tensor(0.7500)

        >>> input = torch.tensor([[1, 1, 1, 0], [0.1, 0.5, 0.7, 0.8]])
        >>> target = torch.tensor([[1, 0, 1, 0], [1, 0, 1, 1]])
        >>> binary_auroc(input, target, num_tasks=2)
        tensor([0.7500, 0.6667])
    """
    _binary_auroc_update_input_check(input, target, num_tasks, weight)
    return _binary_auroc_compute(input, target, weight, use_fbgemm)


@torch.inference_mode()
def multiclass_auroc(
    input: torch.Tensor,
    target: torch.Tensor,
    *,
    num_classes: int,
    average: str | None = "macro",
) -> torch.Tensor:
    """
    Compute AUROC, which is the area under the ROC Curve, for multiclass classification.
    Its class version is :obj:`torcheval.metrics.MulticlassAUROC`.
    See also :func:`binary_auroc <torcheval.metrics.functional.binary_auroc>`

    Args:
        input (Tensor): Tensor of label predictions
            It should be probabilities or logits with shape of (n_sample, n_class).
        target (Tensor): Tensor of ground truth labels with shape of (n_samples, ).
        num_classes (int): Number of classes.
        average (str, optional):
            - ``'macro'`` [default]:
                Calculate metrics for each class separately, and return their unweighted mean.
            - ``None``:
                Calculate the metric for each class separately, and return
                the metric for every class.

    Examples::

        >>> import torch
        >>> from torcheval.metrics.functional import multiclass_auroc
        >>> input = torch.tensor([[0.1, 0.1, 0.1, 0.1], [0.5, 0.5, 0.5, 0.5], [0.7, 0.7, 0.7, 0.7], [0.8, 0.8, 0.8, 0.8]])
        >>> target = torch.tensor([0, 1, 2, 3])
        >>> multiclass_auroc(input, target, num_classes=4)
        0.5
        >>> multiclass_auroc(input, target, num_classes=4, average=None)
        tensor([0.0000, 0.3333, 0.6667, 1.0000])
    """
    _multiclass_auroc_param_check(num_classes, average)
    _multiclass_auroc_update_input_check(input, target, num_classes)
    return _multiclass_auroc_compute(input, target, num_classes, average)


@torch.jit.script
def _binary_auroc_compute_jit(
    input: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    threshold, indices = input.sort(descending=True)
    mask = F.pad(threshold.diff(dim=-1) != 0, [0, 1], value=1.0)
    sorted_target = torch.gather(target, -1, indices)
    sorted_weight = (
        torch.tensor(1.0, device=target.device)
        if weight is None
        else torch.gather(weight, -1, indices.to(target.device))
    )
    cum_tp_before_pad = (sorted_weight * sorted_target).cumsum(-1)
    cum_fp_before_pad = (sorted_weight * (1 - sorted_target)).cumsum(-1)

    shifted_mask = mask.sum(-1, keepdim=True) >= torch.arange(
        mask.size(-1), 0, -1, device=target.device
    )

    cum_tp = torch.zeros_like(cum_tp_before_pad)
    cum_fp = torch.zeros_like(cum_fp_before_pad)

    cum_tp.masked_scatter_(shifted_mask, cum_tp_before_pad[mask])
    cum_fp.masked_scatter_(shifted_mask, cum_fp_before_pad[mask])

    if len(mask.shape) > 1:
        factor = cum_tp[:, -1] * cum_fp[:, -1]
    else:
        factor = cum_tp[-1] * cum_fp[-1]
    # Set AUROC to 0.5 when the target contains all ones or all zeros.
    auroc = torch.where(
        factor == 0,
        0.5,
        torch.trapz(cum_tp, cum_fp).type(largest_float(target.device)) / factor,
    )
    return auroc


def _binary_auroc_compute(
    input: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None = None,
    use_fbgemm: bool | None = False,
) -> torch.Tensor:
    if use_fbgemm:
        assert input.is_cuda and target.is_cuda, "Tensors have to be on GPU"
        # auroc does not have weight
        weight = torch.ones_like(input, dtype=largest_float(input.device))
        num_tasks = 1 if len(input.shape) == 1 else input.shape[0]
        # FBGEMM AUC is an approximation of AUC. It does not mask data in case
        # that input values are redundant. For the highly redundant input case,
        # FBGEMM AUC can give a significantly different result
        auroc = fbgemm_gpu.metrics.auc(num_tasks, input, target, weight)
        if num_tasks == 1:
            return auroc[0]
        else:
            return auroc
    else:
        return _binary_auroc_compute_jit(input, target, weight)


def _binary_auroc_update_input_check(
    input: torch.Tensor,
    target: torch.Tensor,
    num_tasks: int,
    weight: torch.Tensor | None = None,
) -> None:
    if input.shape != target.shape:
        raise ValueError(
            "The `input` and `target` should have the same shape, "
            f"got shapes {input.shape} and {target.shape}."
        )
    if weight is not None and weight.shape != target.shape:
        raise ValueError(
            "The `weight` and `target` should have the same shape, "
            f"got shapes {weight.shape} and {target.shape}."
        )

    if num_tasks == 1:
        if len(input.shape) > 1:
            raise ValueError(
                f"`num_tasks = 1`, `input` is expected to be one-dimensional tensor, but got shape ({input.shape})."
            )
    elif len(input.shape) == 1 or input.shape[0] != num_tasks:
        raise ValueError(
            f"`num_tasks = {num_tasks}`, `input`'s shape is expected to be ({num_tasks}, num_samples), but got shape ({input.shape})."
        )


@torch.jit.script
def _multiclass_auroc_compute(
    input: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    average: str | None = "macro",
) -> torch.Tensor:
    thresholds, indices = input.T.sort(dim=1, descending=True)
    mask = F.pad(thresholds.diff(dim=1) != 0, [0, 1], value=1.0)
    shifted_mask = mask.sum(-1, keepdim=True) >= torch.arange(
        mask.size(-1), 0, -1, device=target.device
    )

    arange = torch.arange(num_classes, device=target.device)
    cmp = target[indices] == arange[:, None]
    cum_tp_before_pad = cmp.cumsum(1)
    cum_fp_before_pad = (~cmp).cumsum(1)

    cum_tp = torch.zeros_like(cum_tp_before_pad)
    cum_fp = torch.zeros_like(cum_fp_before_pad)
    cum_tp.masked_scatter_(shifted_mask, cum_tp_before_pad[mask])
    cum_fp.masked_scatter_(shifted_mask, cum_fp_before_pad[mask])

    factor = cum_tp[:, -1] * cum_fp[:, -1]
    auroc = torch.where(
        factor == 0, 0.5, torch.trapezoid(cum_tp, cum_fp, dim=1) / factor
    )
    if isinstance(average, str) and average == "macro":
        return auroc.mean()
    return auroc


def _multiclass_auroc_param_check(
    num_classes: int,
    average: str | None,
) -> None:
    average_options = ("macro", "none", None)
    if average not in average_options:
        raise ValueError(
            f"`average` was not in the allowed value of {average_options}, got {average}."
        )
    if num_classes < 2:
        raise ValueError("`num_classes` has to be at least 2.")


def _multiclass_auroc_update_input_check(
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
