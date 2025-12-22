# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import torch


@torch.inference_mode()
def binary_accuracy(
    input: torch.Tensor,
    target: torch.Tensor,
    *,
    threshold: float = 0.5,
) -> torch.Tensor:
    """
    Compute binary accuracy score, which is the frequency of input matching target.
    Its class version is :obj:`torcheval.metrics.BinaryAccuracy`.
    See also :func:`multiclass_accuracy <torcheval.metrics.functional.multiclass_accuracy>`, :func:`multilabel_accuracy <torcheval.metrics.functional.multilabel_accuracy>`, :func:`topk_multilabel_accuracy <torcheval.metrics.functional.topk_multilabel_accuracy>`

    Args:
        input (Tensor): Tensor of label predictions with shape of (n_sample,).
            ``torch.where(input < threshold, 0, 1)`` will be applied to the input.
        target (Tensor): Tensor of ground truth labels with shape of (n_sample,).
        threshold (float, default 0.5): Threshold for converting input into predicted labels for each sample.
            ``torch.where(input < threshold, 0, 1)`` will be applied to the ``input``.
    Examples::

        >>> import torch
        >>> from torcheval.metrics.functional import binary_accuracy
        >>> input = torch.tensor([0, 0, 1, 1])
        >>> target = torch.tensor([1, 0, 1, 1])
        >>> binary_accuracy(input, target)
        tensor(0.75)  # 3 / 4

        >>> input = torch.tensor([0, 0.2, 0.6, 0.7])
        >>> target = torch.tensor([1, 0, 1, 1])
        >>> binary_accuracy(input, target, threshold=0.7)
        tensor(0.5)  # 2 / 4
    """

    num_correct, num_total = _binary_accuracy_update(input, target, threshold)
    return _accuracy_compute(num_correct, num_total, "micro")


@torch.inference_mode()
def multiclass_accuracy(
    input: torch.Tensor,
    target: torch.Tensor,
    *,
    average: str | None = "micro",
    num_classes: int | None = None,
    k: int = 1,
) -> torch.Tensor:
    """
    Compute accuracy score, which is the frequency of input matching target.
    Its class version is :obj:`torcheval.metrics.MulticlassAccuracy`.
    See also :func:`binary_accuracy <torcheval.metrics.functional.binary_accuracy>`, :func:`multilabel_accuracy <torcheval.metrics.functional.multilabel_accuracy>`, :func:`topk_multilabel_accuracy <torcheval.metrics.functional.topk_multilabel_accuracy>`

    Args:
        input (Tensor): Tensor of label predictions
            It could be the predicted labels, with shape of (n_sample, ).
            It could also be probabilities or logits with shape of (n_sample, n_class).
            ``torch.argmax`` will be used to convert input into predicted labels.
        target (Tensor): Tensor of ground truth labels with shape of (n_sample, ).
        average:
            - ``'micro'`` [default]:
                Calculate the metrics globally.
            - ``'macro'``:
                Calculate metrics for each class separately, and return their unweighted
                mean. Classes with 0 true instances are ignored.
            - ``None``:
                Calculate the metric for each class separately, and return
                the metric for every class.
                NaN is returned if a class has no sample in ``target``.
        num_classes:
            Number of classes. Required for ``'macro'`` and ``None`` average methods.
        k: Number of top probabilities to be considered. K should be an integer greater than or equal to 1.
            If k > 1, the input tensor must contain probabilities or logits for every class.

    Examples::

        >>> import torch
        >>> from torcheval.metrics.functional import multiclass_accuracy
        >>> input = torch.tensor([0, 2, 1, 3])
        >>> target = torch.tensor([0, 1, 2, 3])
        >>> multiclass_accuracy(input, target)
        tensor(0.5)
        >>> multiclass_accuracy(input, target, average=None, num_classes=4)
        tensor([1., 0., 0., 1.])
        >>> multiclass_accuracy(input, target, average="macro", num_classes=4)
        tensor(0.5)
        >>> input = torch.tensor([[0.9, 0.1, 0, 0], [0.1, 0.2, 0.4, 0,3], [0, 1.0, 0, 0], [0, 0, 0.2, 0.8]])
        >>> multiclass_accuracy(input, target)
        tensor(0.5)
    """

    _accuracy_param_check(average, num_classes, k)
    num_correct, num_total = _multiclass_accuracy_update(
        input, target, average, num_classes, k
    )
    return _accuracy_compute(num_correct, num_total, average)


@torch.inference_mode()
def multilabel_accuracy(
    input: torch.Tensor,
    target: torch.Tensor,
    *,
    threshold: float = 0.5,
    criteria: str = "exact_match",
) -> torch.Tensor:
    """
    Compute multilabel accuracy score, which is the frequency of input matching target.
    Its class version is :obj:`torcheval.metrics.MultilabelAccuracy`.
    See also :func:`binary_accuracy <torcheval.metrics.functional.binary_accuracy>`, :func:`multiclass_accuracy <torcheval.metrics.functional.multiclass_accuracy>`, :func:`topk_multilabel_accuracy <torcheval.metrics.functional.topk_multilabel_accuracy>`

    Args:
        input (Tensor): Tensor of label predictions with shape of (n_sample, n_class).
            `torch.where(input < threshold, 0, 1)`` will be applied to the ``input``.
        target (Tensor): Tensor of ground truth labels with shape of (n_sample, n_class).
        threshold: Threshold for converting input into predicted labels for each sample.
        criteria:
        - ``'exact_match'`` [default]:
            The set of labels predicted for a sample must exactly match the corresponding
            set of labels in target. Also known as subset accuracy.
        - ``'hamming'``:
            Fraction of correct labels over total number of labels.
        - ``'overlap'``:
            The set of labels predicted for a sample must overlap with the corresponding
            set of labels in target.
        - ``'contain'``:
            The set of labels predicted for a sample must contain the corresponding
            set of labels in target.
        - ``'belong'``:
            The set of labels predicted for a sample must (fully) belong to the corresponding
            set of labels in target.

    Examples::

        >>> import torch
        >>> from torcheval.metrics.functional import multilabel_accuracy
        >>> input = torch.tensor([[0, 1], [1, 1], [0, 0], [0, 1]])
        >>> target = torch.tensor([[0, 1], [1, 0], [0, 0], [1, 1]])
        >>> multilabel_accuracy(input, target)
        tensor(0.5)  # 2 / 4

        >>> input = torch.tensor([[0, 1], [1, 1], [0, 0], [0, 1]])
        >>> target = torch.tensor([[0, 1], [1, 0], [0, 0], [1, 1]])
        >>> multilabel_accuracy(input, target, criteria="hamming")
        tensor(0.75)  # 6 / 8

        >>> input = torch.tensor([[0, 1], [1, 1], [0, 0], [0, 1]])
        >>> target = torch.tensor([[0, 1], [1, 0], [0, 0], [1, 1]])
        >>> multilabel_accuracy(input, target, criteria="overlap")
        tensor(1)  # 4 / 4

        >>> input = torch.tensor([[0, 1], [1, 1], [0, 0], [0, 1]])
        >>> target = torch.tensor([[0, 1], [1, 0], [0, 0], [1, 1]])
        >>> multilabel_accuracy(input, target, criteria="contain")
        tensor(0.75)  # 3 / 4, input[0],input[1],input[2]

        >>> input = torch.tensor([[0, 1], [1, 1], [0, 0], [0, 1]])
        >>> target = torch.tensor([[0, 1], [1, 0], [0, 0], [1, 1]])
        >>> multilabel_accuracy(input, target, criteria="belong")
        tensor(0.75)  # 3 / 4, input[0],input[1],input[3]
    """

    _multilabel_accuracy_param_check(criteria)
    num_correct, num_total = _multilabel_accuracy_update(
        input, target, threshold, criteria
    )
    return _accuracy_compute(num_correct, num_total, "micro")


@torch.inference_mode()
def topk_multilabel_accuracy(
    input: torch.Tensor,
    target: torch.Tensor,
    *,
    criteria: str = "exact_match",
    k: int = 2,
) -> torch.Tensor:
    """
    Compute multilabel accuracy score, which is the frequency of the top k label predicted matching target.
    Its class version is :obj:`torcheval.metrics.TopKMultilabelAccuracy`.
    See also :func:`binary_accuracy <torcheval.metrics.functional.binary_accuracy>`, :func:`multiclass_accuracy <torcheval.metrics.functional.multiclass_accuracy>`, :func:`multilabel_accuracy <torcheval.metrics.functional.multilabel_accuracy>`

    Args:
        input (Tensor): Tensor of logits/probabilities with shape of (n_sample, n_class).
        target (Tensor): Tensor of ground truth labels with shape of (n_sample, n_class).
        criteria:
        - ``'exact_match'`` [default]:
            The set of top-k labels predicted for a sample must exactly match the corresponding
            set of labels in target. Also known as subset accuracy.
        - ``'hamming'``:
            Fraction of top-k correct labels over total number of labels.
        - ``'overlap'``:
            The set of top-k labels predicted for a sample must overlap with the corresponding
            set of labels in target.
        - ``'contain'``:
            The set of top-k labels predicted for a sample must contain the corresponding
            set of labels in target.
        - ``'belong'``:
            The set of top-k labels predicted for a sample must (fully) belong to the corresponding
            set of labels in target.
        k: Number of top probabilities to be considered. K should be an integer greater than or equal to 1.

    Examples::

        >>> import torch
        >>> from torcheval.metrics.functional import topk_multilabel_accuracy
        >>> input = torch.tensor([[0.1, 0.5, 0.2], [0.3, 0.2, 0.1], [0.2, 0.4, 0.5], [0, 0.1, 0.9]])
        >>> target = torch.tensor([[1, 1, 0], [0, 1, 0], [1, 1, 1], [0, 1, 0]])
        >>> topk_multilabel_accuracy(input, target, k = 2)
        tensor(0)  # 0 / 4

        >>> input = torch.tensor([[0.1, 0.5, 0.2], [0.3, 0.2, 0.1], [0.2, 0.4, 0.5], [0, 0.1, 0.9]])
        >>> target = torch.tensor([[1, 1, 0], [0, 1, 0], [1, 1, 1], [0, 1, 0]])
        >>> topk_multilabel_accuracy(input, target, criteria="hamming", k = 2)
        tensor(0.583)  # 7 / 12

        >>> input = torch.tensor([[0.1, 0.5, 0.2], [0.3, 0.2, 0.1], [0.2, 0.4, 0.5], [0, 0.1, 0.9]])
        >>> target = torch.tensor([[1, 1, 0], [0, 1, 0], [1, 1, 1], [0, 1, 0]])
        >>> topk_multilabel_accuracy(input, target, criteria="overlap", k = 2)
        tensor(1)  # 4 / 4

        >>> input = torch.tensor([[0.1, 0.5, 0.2], [0.3, 0.2, 0.1], [0.2, 0.4, 0.5], [0, 0.1, 0.9]])
        >>> target = torch.tensor([[1, 1, 0], [0, 1, 0], [1, 1, 1], [0, 1, 0]])
        >>> topk_multilabel_accuracy(input, target, criteria="contain", k = 2)
        tensor(0.5)  # 2 / 4

        >>> input = torch.tensor([[0.1, 0.5, 0.2], [0.3, 0.2, 0.1], [0.2, 0.4, 0.5], [0, 0.1, 0.9]])
        >>> target = torch.tensor([[1, 1, 0], [0, 1, 0], [1, 1, 1], [0, 1, 0]])
        >>> topk_multilabel_accuracy(input, target, criteria="belong", k = 2)
        tensor(0.25)  # 1 / 4

    """

    _topk_multilabel_accuracy_param_check(criteria, k)
    num_correct, num_total = _topk_multilabel_accuracy_update(
        input, target, criteria, k
    )
    return _accuracy_compute(num_correct, num_total, "micro")


def _multiclass_accuracy_update(
    input: torch.Tensor,
    target: torch.Tensor,
    average: str | None,
    num_classes: int | None,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    _accuracy_update_input_check(input, target, num_classes, k)

    if k == 1:
        if input.ndim == 2:
            input = torch.argmax(input, dim=1)
        mask = (input == target).long()
    else:
        y_score = torch.gather(input, dim=-1, index=target.unsqueeze(dim=-1))
        rank = torch.gt(input, y_score).sum(dim=-1)
        mask = (rank < k).float()

    if average == "micro":
        num_correct = mask.sum()
        num_total = torch.tensor(target.shape[0])
        return num_correct, num_total

    # pyre-ignore[6]: expect int got Optional[int] for num_classes
    num_correct = mask.new_zeros(num_classes).scatter_add_(0, target, mask)
    # pyre-ignore[6]: expect int got Optional[int] for num_classes
    num_total = target.new_zeros(num_classes).scatter_add_(
        0, target, torch.ones_like(target)
    )
    return num_correct, num_total


@torch.jit.script
def _accuracy_compute(
    num_correct: torch.Tensor,
    num_total: torch.Tensor,
    average: str | None,
) -> torch.Tensor:
    if isinstance(average, str) and average == "macro":
        mask = num_total != 0
        return (num_correct[mask] / num_total[mask]).mean()
    else:
        return num_correct / num_total


def _accuracy_param_check(
    average: str | None,
    num_classes: int | None,
    k: int,
) -> None:
    average_options = ("micro", "macro", "none", None)
    if average not in average_options:
        raise ValueError(
            f"`average` was not in the allowed value of {average_options}, got {average}."
        )
    if average != "micro" and (num_classes is None or num_classes <= 0):
        raise ValueError(
            f"num_classes should be a positive number when average={average}."
            f" Got num_classes={num_classes}."
        )
    if type(k) != int:
        raise TypeError(f"Expected `k` to be an integer, but {type(k)} was provided.")
    if k < 1:
        raise ValueError(
            f"Expected `k` to be an integer greater than 0, but {k} was provided."
        )


def _accuracy_update_input_check(
    input: torch.Tensor,
    target: torch.Tensor,
    num_classes: int | None,
    k: int,
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

    if k > 1 and input.ndim != 2:
        raise ValueError(
            "input should have shape (num_sample, num_classes) for k > 1, "
            f"got shape {input.shape}."
        )

    if not input.ndim == 1 and not (
        input.ndim == 2 and (num_classes is None or input.shape[1] == num_classes)
    ):
        raise ValueError(
            "input should have shape of (num_sample,) or (num_sample, num_classes), "
            f"got {input.shape}."
        )


def _binary_accuracy_update(
    input: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    _binary_accuracy_update_input_check(input, target)

    input = torch.where(input < threshold, 0, 1)

    num_correct = (input == target).sum()
    if target.dtype == torch.bool:
        num_total = torch.tensor(
            target.shape[0],
            dtype=torch.int64,
            device=target.device,
            requires_grad=False,
        )
    else:
        # this is faster than using torch.tensor, but breaks for bool tensors because the shape will be cast to 1 in a bool tensor
        num_total = target.new_tensor(target.shape[0])
    return num_correct, num_total


def _binary_accuracy_update_input_check(
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


def _multilabel_accuracy_update(
    input: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    criteria: str = "exact_match",
) -> tuple[torch.Tensor, torch.Tensor]:
    _multilabel_accuracy_update_input_check(input, target)
    input_label = torch.where(input < threshold, 0, 1)
    return _multilabel_update(input_label, target, criteria)


def _topk_multilabel_accuracy_update(
    input: torch.Tensor,
    target: torch.Tensor,
    criteria: str = "exact_match",
    k: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    _topk_multilabel_accuracy_update_input_check(input, target, k)
    input_label = torch.zeros(input.size(), device=input.device).scatter_(
        -1, input.topk(k=k, dim=-1).indices, 1.0
    )
    return _multilabel_update(input_label, target, criteria)


@torch.jit.script
def _multilabel_update(
    input: torch.Tensor,
    target: torch.Tensor,
    criteria: str = "exact_match",
) -> tuple[torch.Tensor, torch.Tensor]:
    if criteria == "exact_match":
        num_correct = torch.all(input == target, dim=1).sum()
        num_total = torch.tensor(target.shape[0], device=target.device)
        return num_correct, num_total

    elif criteria == "hamming":
        num_correct = (input == target).sum()
        num_total = torch.tensor(target.numel(), device=target.device)
        return num_correct, num_total

    elif criteria == "overlap":
        num_correct = (
            torch.logical_and(input == target, input == 1).max(dim=1)[0].sum()
            + torch.all(torch.logical_and(input == 0, target == 0), dim=1).sum()
        )
        num_total = torch.tensor(target.shape[0], device=target.device)
        return num_correct, num_total

    elif criteria == "contain":
        num_correct = torch.all((input - target) >= 0, dim=1).sum()
        num_total = torch.tensor(target.shape[0], device=target.device)
        return num_correct, num_total

    # belong
    num_correct = torch.all((input - target) <= 0, dim=1).sum()
    num_total = torch.tensor(target.shape[0], device=target.device)
    return num_correct, num_total


def _multilabel_accuracy_param_check(
    criteria: str,
) -> None:
    criteria_options = ("exact_match", "hamming", "overlap", "contain", "belong")
    if criteria not in criteria_options:
        raise ValueError(
            f"`criteria` was not in the allowed value of {criteria_options}, got {criteria}."
        )


def _topk_multilabel_accuracy_param_check(
    criteria: str,
    k: int,
) -> None:
    _multilabel_accuracy_param_check(criteria)
    if type(k) != int:
        raise TypeError(f"Expected `k` to be an integer, but {type(k)} was provided.")
    if k == 1:
        raise ValueError(
            f"Expected `k` to be an integer greater than 1, but {k} was provided. In such case, please use multilabel_accuracy metric."
        )
    if k < 1:
        raise ValueError(
            f"Expected `k` to be an integer greater than 1, but {k} was provided."
        )


def _multilabel_accuracy_update_input_check(
    input: torch.Tensor,
    target: torch.Tensor,
) -> None:
    if input.shape != target.shape:
        raise ValueError(
            "The `input` and `target` should have the same dimensions, "
            f"got shapes {input.shape} and {target.shape}."
        )


def _topk_multilabel_accuracy_update_input_check(
    input: torch.Tensor,
    target: torch.Tensor,
    k: int,
) -> None:
    if input.shape != target.shape:
        raise ValueError(
            "The `input` and `target` should have the same dimensions, "
            f"got shapes {input.shape} and {target.shape}."
        )

    if input.ndim != 2:
        raise ValueError(
            "input should have shape (num_sample, num_classes) for k > 1, "
            f"got shape {input.shape}."
        )
