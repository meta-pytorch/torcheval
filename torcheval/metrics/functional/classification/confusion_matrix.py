# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import torch

norm = torch.nn.functional.normalize


@torch.inference_mode()
def binary_confusion_matrix(
    input: torch.Tensor,
    target: torch.Tensor,
    *,
    threshold: float = 0.5,
    normalize: str | None = None,
) -> torch.Tensor:
    """
    Compute binary confusion matrix, a 2 by 2 tensor with counts ( (true positive, false negative) , (false positive, true negative) )
    See also :func:`multiclass_confusion_matrix <torcheval.metrics.functional.multiclass_confusion_matrix>`

    Args:
        input (Tensor): Tensor of label predictions with shape of (n_sample,).
            ``torch.where(input < threshold, 0, 1)`` will be applied to the input.
        target (Tensor): Tensor of ground truth labels with shape of (n_sample,).
        threshold (float, default 0.5): Threshold for converting input into predicted labels for each sample.
            ``torch.where(input < threshold, 0, 1)`` will be applied to the ``input``.
        normalize:
            - ``None`` [default]:
                Give raw counts ('none' also defaults to this)
            - ``'pred'``:
                Normalize across the prediction, i.e. such that the rows add to one.
            - ``'true'``:
                Normalize across the condition positive, i.e. such that the columns add to one.
            - ``'all'``"
                Normalize across all examples, i.e. such that all matrix entries add to one.
    Examples::

        >>> import torch
        >>> from torcheval.metrics.functional import binary_confusion_matrix
        >>> input = torch.tensor([0, 1, 0.7, 0.6])
        >>> target = torch.tensor([0, 1, 1, 0])
        >>> binary_confusion_matrix(input, target)
        tensor([[1, 1],
                [0, 2]])

        >>> input = torch.tensor([1, 1, 0, 0])
        >>> target = torch.tensor([0, 1, 1, 1])
        >>> binary_confusion_matrix(input, target, threshold=1)
        tensor([[0, 1],
                [2, 1]])

        >>> input = torch.tensor([1, 1, 0, 0])
        >>> target = torch.tensor([0, 1, 1, 1])
        >>> binary_confusion_matrix(input, target, normalize="true")
        tensor([[0.0000, 1.0000],
                [0.6667, 0.3333]])
    """
    _confusion_matrix_param_check(2, normalize)
    matrix = _binary_confusion_matrix_update(input, target, threshold)
    return _confusion_matrix_compute(matrix, normalize)


@torch.inference_mode()
def multiclass_confusion_matrix(
    input: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    *,
    normalize: str | None = None,
) -> torch.Tensor:
    """
    Compute multi-class confusion matrix, a matrix of dimension num_classes x num_classes where each element at position `(i,j)` is the number of examples with true class `i` that were predicted to be class `j`.
    See also :func:`binary_confusion_matrix <torcheval.metrics.functional.binary_confusion_matrix>`

    Args:
        input (Tensor): Tensor of label predictions.
            It could be the predicted labels, with shape of (n_sample, ).
            It could also be probabilities or logits with shape of (n_sample, n_class).
            ``torch.argmax`` will be used to convert input into predicted labels.
        target (Tensor): Tensor of ground truth labels with shape of (n_sample, ).
        num_classes (int):
            Number of classes.
        normalize:
            - ``None`` [default]:
                Give raw counts ('none' also defaults to this)
            - ``'pred'``:
                Normalize across the prediction class, i.e. such that the rows add to one.
            - ``'true'``:
                Normalize across the condition positive, i.e. such that the columns add to one.
            - ``'all'``"
                Normalize across all examples, i.e. such that all matrix entries add to one.

    Examples::

        >>> import torch
        >>> from torcheval.metrics.functional import multiclass_confusion_matrix
        >>> input = torch.tensor([0, 2, 1, 3])
        >>> target = torch.tensor([0, 1, 2, 3])
        >>> multiclass_confusion_matrix(input, target, 4)
        tensor([[1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1]])

        >>> input = torch.tensor([0, 0, 1, 1, 1])
        >>> target = torch.tensor([0, 0, 0, 0, 1])
        >>> multiclass_confusion_matrix(input, target, 2)
        tensor([[2, 2],
                [0, 1]])

        >>> input = torch.tensor([0, 0, 1, 1, 1, 2, 1, 2])
        >>> target = torch.tensor([2, 0, 2, 0, 1, 2, 1, 0])
        >>> multiclass_confusion_matrix(input, target, 3)
        tensor([[1, 1, 1],
                [0, 2, 0],
                [1, 1, 1]])

        >>> input = torch.tensor([0, 0, 1, 1, 1, 2, 1, 2])
        >>> target = torch.tensor([2, 0, 2, 0, 1, 2, 1, 0])
        >>> multiclass_confusion_matrix(input, target, 3, normalize="pred")
        tensor([[0.5000, 0.2500, 0.5000],
                [0.0000, 0.5000, 0.0000],
                [0.5000, 0.2500, 0.5000]])


        >>> input = torch.tensor([0, 0, 1, 1, 1])
        >>> target = torch.tensor([0, 0, 0, 0, 1])
        >>> multiclass_confusion_matrix(input, target, 4)
        tensor([[2, 2, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0]])

        >>> input = torch.tensor([[0.9, 0.1, 0, 0], [0.1, 0.2, 0.4, 0.3], [0, 1.0, 0, 0], [0, 0, 0.2, 0.8]])
        >>> target = torch.tensor([0, 1, 2, 3])
        >>> multiclass_confusion_matrix(input, target, 4)
        tensor([[1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1]])
    """
    _confusion_matrix_param_check(num_classes, normalize)
    sparse_cm = _confusion_matrix_update(input, target, num_classes)
    return _confusion_matrix_compute(sparse_cm, normalize=normalize)


def _binary_confusion_matrix_compute(
    cm: torch.Tensor, normalize: str | None
) -> torch.Tensor:
    if normalize == "pred":
        return norm(cm.to(torch.float), p=1, dim=1)
    elif normalize == "true":
        return norm(cm.to(torch.float), p=1, dim=0)
    elif normalize == "all":
        return cm.to(torch.float) / torch.sum(cm)
    else:
        return cm


def _binary_confusion_matrix_update(
    input: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    _binary_confusion_matrix_update_input_check(input, target)

    input = torch.where(input < threshold, 0, 1)

    return _update(input, target, 2)


def _binary_confusion_matrix_update_input_check(
    input: torch.Tensor,
    target: torch.Tensor,
) -> None:
    if input.ndim != 1:
        raise ValueError(
            f"input should be a one-dimensional tensor for binary confusion matrix, got shape {input.shape}."
        )
    if target.ndim != 1:
        raise ValueError(
            f"target should be a one-dimensional tensor for binary confusion matrix, got shape {target.shape}."
        )
    if input.shape != target.shape:
        raise ValueError(
            "The `input` and `target` should have the same dimensions, "
            f"got shapes {input.shape} and {target.shape}."
        )


def _confusion_matrix_compute(
    confusion_matrix: torch.Tensor,
    normalize: str | None,
) -> torch.Tensor:
    if normalize == "pred":
        return norm(confusion_matrix.to(torch.float), p=1, dim=0)
    elif normalize == "true":
        return norm(confusion_matrix.to(torch.float), p=1, dim=1)
    elif normalize == "all":
        return confusion_matrix.to(torch.float) / torch.sum(confusion_matrix)
    else:
        return confusion_matrix


def _confusion_matrix_update(
    input: torch.Tensor, target: torch.Tensor, num_classes: int
) -> torch.Tensor:
    _confusion_matrix_update_input_check(input, target, num_classes)
    return _update(input, target, num_classes)


@torch.jit.script
def _update(
    input: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    if input.ndim == 2:
        input = torch.argmax(input, dim=1)

    coordinates = torch.vstack((target, input))
    cm_shape = torch.Size([num_classes, num_classes])

    # Each prediction creates an entry at the position (true, pred)
    sparse_cm = torch.sparse_coo_tensor(coordinates, torch.ones_like(target), cm_shape)

    return sparse_cm.to_dense()


def _confusion_matrix_param_check(
    num_classes: int,
    normalize: str | None,
) -> None:
    if num_classes < 2:
        raise ValueError("Must be at least two classes for confusion matrix")
    if (normalize is not None) and (normalize not in ["all", "pred", "true", "none"]):
        raise ValueError("normalize must be one of 'all', 'pred', 'true', or 'none'.")


def _confusion_matrix_update_input_check(
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
            f"target should be a one-dimensional tensor, got shape {target.shape}."
        )

    # check if num classes is high enough to cover inputs.
    if not input.ndim == 1:
        if not (input.ndim == 2 and (input.shape[1] == num_classes)):
            raise ValueError(
                f"input should have shape of (num_sample,) or (num_sample, num_classes), "
                f"got {input.shape}."
            )
    else:
        # pyre-fixme[58]: `>=` is not supported for operand types `Tensor` and
        #  `Optional[int]`.
        if torch.max(input) >= num_classes:
            raise ValueError(
                "Got `input` prediction class which is too large for the number of classes, "
                f"num_classes: {num_classes} must be strictly greater than max class predicted: {torch.max(input)}."
            )

    # check if num classes is high enough to cover targets.
    # pyre-fixme[58]: `>=` is not supported for operand types `Tensor` and
    #  `Optional[int]`.
    if torch.max(target) >= num_classes:
        raise ValueError(
            "Got `target` class which is larger than the number of classes, "
            f"num_classes: {num_classes} must be strictly greater than max target: {torch.max(target)}."
        )
