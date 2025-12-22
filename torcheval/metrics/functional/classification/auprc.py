# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import torch
from torcheval.metrics.functional.classification.precision_recall_curve import (
    _compute_for_each_class,
    multiclass_precision_recall_curve,
    multilabel_precision_recall_curve,
)
from torcheval.metrics.functional.tensor_utils import _riemann_integral


@torch.inference_mode()
def binary_auprc(
    input: torch.Tensor,
    target: torch.Tensor,
    *,
    num_tasks: int = 1,
) -> torch.Tensor:
    r"""
    Compute AUPRC, also called Average Precision, which is the area under the Precision-Recall Curve, for binary classification.
    Its class version is :obj:`torcheval.metrics.BinaryAUPRC`.

    Precision is defined as :math:`\frac{T_p}{T_p+F_p}`; it is the probability that a positive prediction from the model is a true positive.
    Recall is defined as :math:`\frac{T_p}{T_p+F_n}`; it is the probability that a true positive is predicted to be positive by the model.

    The precision-recall curve plots the recall on the x axis and the precision on the y axis, both of which are bounded between 0 and 1.
    This function returns the area under that graph. If the area is near one, the model supports a threshold which correctly identifies
    a high percentage of true positives while also rejecting enough false examples so that most of the true predictions are true positives.

    See also :func:`multiclass_auprc <torcheval.metrics.functional.multiclass_auprc>`, :func:`multilabel_auprc <torcheval.metrics.functional.multilabel_auprc>`

    Args:
        input (Tensor): Tensor of label predictions
            It should be predicted label, probabilities or logits with shape of (num_tasks, n_sample) or (n_sample, ).
        target (Tensor): Tensor of ground truth labels with shape of (num_tasks, n_sample) or (n_sample, ).
        num_tasks (int):  Number of tasks that need BinaryAUPRC calculation. Default value
                    is 1. Binary AUPRC for each task will be calculated independently. Results are
                    equivalent to calling binary_auprc for each row.

    Examples::

        >>> import torch
        >>> from torcheval.metrics.functional import binary_auprc
        >>> input = torch.tensor([0.1, 0.5, 0.7, 0.8])
        >>> target = torch.tensor([1, 0, 1, 1])
        >>> binary_auprc(input, target)
        tensor(0.9167) # scalar returned with 1D input tensors

        >>> input = torch.tensor([[1, 1, 1, 0]])
        >>> target = torch.tensor([[1, 0, 1, 0]])
        >>> binary_auprc(input, target)
        tensor([0.6667]) # 1D tensor returned with 2D input tensors

        >>> input = torch.tensor([[0.1, 0.5, 0.7, 0.8],
        >>>                       [1, 1, 1, 0]])
        >>> target = torch.tensor([[1, 0, 1, 1],
        >>>                        [1, 0, 1, 0]])
        >>> binary_auprc(input, target, num_tasks=2)
        tensor([0.9167, 0.6667])
    """
    _binary_auprc_update_input_check(input, target, num_tasks)
    return _binary_auprc_compute(input, target, num_tasks)


@torch.inference_mode()
def multiclass_auprc(
    input: torch.Tensor,
    target: torch.Tensor,
    num_classes: int | None = None,
    *,
    average: str | None = "macro",
) -> torch.Tensor:
    r"""
    Compute AUPRC, also called Average Precision, which is the area under the Precision-Recall Curve, for multiclass classification.
    Its class version is :obj:`torcheval.metrics.MulticlassAUPRC`.

    Precision is defined as :math:`\frac{T_p}{T_p+F_p}`; it is the probability that a positive prediction from the model is a true positive.
    Recall is defined as :math:`\frac{T_p}{T_p+F_n}`; it is the probability that a true positive is predicted to be positive by the model.

    The precision-recall curve plots the recall on the x axis and the precision on the y axis, both of which are bounded between 0 and 1.
    This function returns the area under that graph. If the area is near one, the model supports a threshold which correctly identifies
    a high percentage of true positives while also rejecting enough false examples so that most of the true predictions are true positives.

    In the multiclass version of auprc, the target tensor is 1-dimensional and contains an integer entry representing the class for each example
    in the input tensor. Each class is considered independently in a one-vs-all fashion, examples for that class are labeled condition true and all other
    classes are considered condition false.

    The results of N class multiclass auprc without an average is equivalent to binary auprc with N tasks if:

    1. the input is transposed, in binary classification examples are associated with columns, whereas they are associated with rows in multiclass classification.
    2. the `target` is translated from the form [1,0,1] to the form [[0,1,0], [1,0,1]]

    See also :func:`binary_auprc <torcheval.metrics.functional.binary_auprc>`, :func:`multilabel_auprc <torcheval.metrics.functional.multilabel_auprc>`

    Args:
        input (Tensor): 2 dimensional tensor of label predictions
            It should be probabilities or logits with shape of (n_sample, n_class).
        target (Tensor): 1 dimensional tensor of ground truth labels with shape of (n_samples, ).
        num_classes (int): Number of classes.
        average (str, optional):
            - ``'macro'`` [default]:
                Calculate metrics for each class separately, and return their unweighted mean.
            - ``None`` or ``'none'``:
                Calculate the metric for each class separately, and return
                the metric for every class.

    Examples::
        >>> import torch
        >>> from torcheval.metrics.functional import multiclass_auprc
        >>> input = tensor([[0.5647, 0.2726],
                            [0.9143, 0.1895],
                            [0.7782, 0.3082]])
        >>> target = tensor([0, 1, 0])
        >>> multiclass_auprc(input, target, average=None)
        tensor([0.5833, 0.3333])
        >>> multiclass_auprc(input, target)
        tensor(0.4583)

        >>> input = torch.tensor([[0.1, 1],
                                  [0.5, 1],
                                  [0.7, 1],
                                  [0.8, 0]])
        >>> target = torch.tensor([1, 0, 0, 1])
        >>> multiclass_auprc(input, target, 2, average=None)
        tensor([0.5833, 0.4167])

        Connection with binary
        >>> from torcheval.metrics.functional import binary_auprc
        >>> input = torch.tensor([[0.1, 0.5, 0.7, 0.8],
        >>>                       [1, 1, 1, 0]])
        >>> target = torch.tensor([[0, 1, 1, 0],
        >>>                        [1, 0, 0, 1]])
        >>> binary_auprc(input, target, num_tasks=2)
        tensor([0.5833, 0.4167])

    """
    if num_classes is None:
        num_classes = input.shape[1]

    _multiclass_auprc_param_check(num_classes, average)
    _multiclass_auprc_update_input_check(input, target, num_classes)
    return _multiclass_auprc_compute(input, target, average)


@torch.inference_mode()
def multilabel_auprc(
    input: torch.Tensor,
    target: torch.Tensor,
    num_labels: int | None = None,
    *,
    average: str | None = "macro",
) -> torch.Tensor:
    r"""
    Compute AUPRC, also called Average Precision, which is the area under the Precision-Recall Curve, for multilabel classification.
    Its class version is :obj:`torcheval.metrics.MultilabelAUPRC`.

    Precision is defined as :math:`\frac{T_p}{T_p+F_p}`, it is the probability that a positive prediction from the model is a true positive.
    Recall is defined as :math:`\frac{T_p}{T_p+F_n}`, it is the probability that a true positive is predicted to be positive by the model.

    The precision-recall curve plots the recall on the x axis and the precision on the y axis, both of which are bounded between 0 and 1.
    This function returns the area under that graph. If the area is near one, the model supports a threshold which correctly identifies
    a high percentage of true positives while also rejecting enough false examples so that most of the true predictions are true positives.

    In the multilabel version of AUPRC, the input and target tensors are 2-dimensional. The rows of each tensor are associated with a particular example and the columns are associated with a particular class.

    For the target tensor, the entry of the r'th row and c'th column (r and c are 0-indexed) is 1 if the r'th example belongs to the c'th class, and 0 if not. For the input tensor, the entry in the same position is the output of the classification model prediciting the inclusion of the r'th example in the c'th class.
    Note that in the multilabel setting, multiple labels are allowed to apply to a single sample. This stands in contrast to the multiclass
    sample, in which there may be more than 2 distinct classes but each sample must have exactly one class.

    The results of N label multilabel auprc without an average is equivalent to binary auprc with N tasks if:

    1. the `input` is transposed, in binary labelification examples are associated with columns, whereas they are associated with rows in multilabel classification.
    2. the `target` is transposed for the same reason

    See examples below for more details on the connection between Multilabel and Binary AUPRC.
    See also :func:`binary_auprc <torcheval.metrics.functional.binary_auprc>`, :func:`multiclass_auprc <torcheval.metrics.functional.multiclass_auprc>`

    Args:
        input (Tensor): Tensor of label predictions
            It should be probabilities or logits with shape of (n_sample, n_label).
        target (Tensor): Tensor of ground truth labels with shape of (n_samples, n_label).
        num_labels (int): Number of labels.
        average (str, optional):
            - ``'macro'`` [default]:
                Calculate metrics for each class separately, and return their unweighted mean.
            - ``None`` or ``'none'``:
                Calculate the metric for each class separately, and return
                the metric for every class.

    Examples::
        >>> import torch
        >>> from torcheval.metrics.functional import multilabel_auprc
        >>> input = torch.tensor([[0.75, 0.05, 0.35], [0.45, 0.75, 0.05], [0.05, 0.55, 0.75], [0.05, 0.65, 0.05]])
        >>> target = torch.tensor([[1, 0, 1], [0, 0, 0], [0, 1, 1], [1, 1, 1]])
        >>> multilabel_auprc(input, target, num_labels=3, average=None)
        tensor([0.7500, 0.5833, 0.9167])
        >>> multilabel_auprc(input, target, average=None)
        tensor([0.7500, 0.5833, 0.9167])
        >>> multilabel_auprc(input, target, num_labels=3, average='macro')
        tensor(0.7500)
        >>> multilabel_auprc(input, target, num_labels=3)
        tensor(0.7500)
        >>> multilabel_auprc(input, target, average='macro')
        tensor(0.7500)
        >>> multilabel_auprc(input, target)
        tensor(0.7500)

        Connection to BinaryAUPRC
        >>> input = torch.tensor([[0.1, 0, 0], [0, 1, 0], [0.1, 0.2, 0.7], [0, 0, 1]])
        >>> target = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]])
        >>> multilabel_auprc(input, target)
        tensor([0.5000, 1.0000, 1.0000])

        the above is equivalent to
        >>> from torcheval.metrics import BinaryAUPRC
        >>> input = torch.tensor([[0.1, 0, 0.1, 0], [0, 1, 0.2, 0], [0, 0, 0.7, 1]])
        >>> target = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1]])
        >>> binary_auprc(input, target, num_tasks=3)
        tensor([0.5000, 1.0000, 1.0000])
    """
    if input.ndim != 2:
        raise ValueError(
            f"input should be a two-dimensional tensor, got shape {input.shape}."
        )
    if num_labels is None:
        num_labels = input.shape[1]

    _multilabel_auprc_param_check(num_labels, average)
    _multilabel_auprc_update_input_check(input, target, num_labels)
    return _multilabel_auprc_compute(input, target, num_labels, average)


def _binary_auprc_compute(
    input: torch.Tensor, target: torch.Tensor, num_tasks: int = 1
) -> torch.Tensor:
    # for one task preserve the ndim of the input and target tensor
    if num_tasks == 1 and input.ndim == 1:
        p, r, t = _compute_for_each_class(input, target, 1)
        return _riemann_integral(r, p)
    else:
        auprcs = []
        for i in range(num_tasks):
            p, r, t = _compute_for_each_class(input[i, :], target[i, :], 1)
            auprcs.append(_riemann_integral(r, p))
        return torch.tensor(auprcs, device=input.device)


def _binary_auprc_update_input_check(
    input: torch.Tensor,
    target: torch.Tensor,
    num_tasks: int,
) -> None:
    if input.shape != target.shape:
        raise ValueError(
            "The `input` and `target` should have the same shape, "
            f"got shapes {input.shape} and {target.shape}."
        )
    if num_tasks == 1:
        if input.ndim == 2 and input.shape[0] > 1:
            raise ValueError(
                f"`num_tasks = 1`, `input` and `target` are expected to be one-dimensional tensors or 1xN tensors, but got shape input: {input.shape}, target: {target.shape}."
            )
        elif input.ndim > 2:
            raise ValueError(
                f"`num_tasks = 1`, `input` and `target` are expected to be one-dimensional tensors or 1xN tensors, but got shape input: {input.shape}, target: {target.shape}."
            )
    elif input.shape[0] != num_tasks:
        raise ValueError(
            f"`num_tasks = {num_tasks}`, `input` and `target` shape is expected to be ({num_tasks}, num_samples), but got shape input: {input.shape}, target: {target.shape}."
        )


def _multiclass_auprc_compute(
    input: torch.Tensor, target: torch.Tensor, average: str | None = "macro"
) -> torch.Tensor:
    prec, recall, thresh = multiclass_precision_recall_curve(input, target)
    auprcs = []
    for p, r in zip(prec, recall):
        auprcs.append(_riemann_integral(r, p))
    auprcs = torch.tensor(auprcs).to(input.device)

    if average == "macro":
        return torch.mean(auprcs)
    else:
        return auprcs


def _multiclass_auprc_param_check(
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


def _multiclass_auprc_update_input_check(
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


def _multilabel_auprc_compute(
    input: torch.Tensor,
    target: torch.Tensor,
    num_labels: int,
    average: str | None = "macro",
) -> torch.Tensor:
    prec, recall, thresh = multilabel_precision_recall_curve(
        input, target, num_labels=num_labels
    )
    auprcs = []
    for p, r in zip(prec, recall):
        auprcs.append(_riemann_integral(r, p))
    auprcs = torch.tensor(auprcs).to(input.device)

    if average == "macro":
        return torch.mean(auprcs)
    else:
        return auprcs


def _multilabel_auprc_param_check(
    num_labels: int,
    average: str | None,
) -> None:
    average_options = ("macro", "none", None)
    if average not in average_options:
        raise ValueError(
            f"`average` was not in the allowed value of {average_options}, got {average}."
        )
    if num_labels < 2:
        raise ValueError("`num_labels` has to be at least 2.")


def _multilabel_auprc_update_input_check(
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
