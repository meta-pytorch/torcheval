# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# pyre-ignore-all-errors[16]: Undefined attribute of metric states.


from collections.abc import Iterable
from typing import TypeVar

import torch
from torcheval.metrics.functional.classification.binned_auprc import (
    _binary_binned_auprc_param_check,
    _binary_binned_auprc_update_input_check,
    _compute_riemann_integrals,
    _multiclass_binned_auprc_param_check,
    _multilabel_binned_auprc_param_check,
    DEFAULT_NUM_THRESHOLD,
)
from torcheval.metrics.functional.classification.binned_precision_recall_curve import (
    _binary_binned_precision_recall_curve_compute,
    _create_threshold_tensor,
    _multiclass_binned_precision_recall_curve_compute,
    _multiclass_binned_precision_recall_curve_update,
    _multilabel_binned_precision_recall_curve_compute,
    _multilabel_binned_precision_recall_curve_update,
    _optimization_param_check,
    _update,
)
from torcheval.metrics.functional.tensor_utils import _riemann_integral
from torcheval.metrics.metric import Metric


TBinaryBinnedAUPRC = TypeVar("TBinaryBinnedAUPRC")
TMulticlassBinnedAUPRC = TypeVar("TMulticlassBinnedAUPRC")
TMultilabelBinnedAUPRC = TypeVar("TMultilabelBinnedAUPRC")


class BinaryBinnedAUPRC(Metric[torch.Tensor]):
    """
    Compute Binned AUPRC, which is the area under the binned version of the Precision Recall Curve, for binary classification.
    Its functional version is :func:`torcheval.metrics.functional.binary_binned_auprc`.

    Args:
        num_tasks (int):  Number of tasks that need binary_binned_auprc calculation. Default value
                    is 1. binary_binned_auprc for each task will be calculated independently.
        threshold: A integer representing number of bins, a list of thresholds, or a tensor of thresholds.


    Examples::

        >>> import torch
        >>> from torcheval.metrics import BinaryBinnedAUPRC
        >>> input = torch.tensor([0.1, 0.5, 0.7, 0.8])
        >>> target = torch.tensor([1, 0, 1, 1])
        >>> metric = BinaryBinnedAUPRC(threshold=5)
        >>> metric.update(input, target)
        >>> metric.compute()
        (tensor([0.8056]),
        tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000])
        )
        >>> input = torch.tensor([1, 1, 1, 0])
        >>> target = torch.tensor([1, 1, 1, 0])
        >>> metric.update(input, target)
        >>> metric.compute()
        (tensor([0.9306]),
        tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000])
        )
        >>> metric = BinaryBinnedAUPRC(num_tasks=2, threshold=[0, 0.2, 0.4, 0.6, 0.8, 1])
        >>> input = torch.tensor([[1, 1, 1, 0], [0.1, 0.5, 0.7, 0.8]])
        >>> target = torch.tensor([[1, 0, 1, 0], [1, 0, 1, 1]])
        >>> metric.update(input, target)
        >>> metric.compute()
        (tensor([0.6667, 0.8056]),
        tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000]))
    """

    def __init__(
        self: TBinaryBinnedAUPRC,
        *,
        num_tasks: int = 1,
        threshold: int | list[float] | torch.Tensor = DEFAULT_NUM_THRESHOLD,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(device=device)
        threshold = _create_threshold_tensor(
            threshold,
            self.device,
        )
        _binary_binned_auprc_param_check(num_tasks, threshold)
        self.num_tasks = num_tasks
        self.threshold = threshold
        self._add_state(
            "num_tp",
            torch.zeros(num_tasks, len(threshold), device=self.device),
        )
        self._add_state(
            "num_fp",
            torch.zeros(num_tasks, len(threshold), device=self.device),
        )
        self._add_state(
            "num_fn",
            torch.zeros(num_tasks, len(threshold), device=self.device),
        )

    @torch.inference_mode()
    # pyre-ignore[14]: inconsistent override on *_:Any, **__:Any
    def update(
        self: TBinaryBinnedAUPRC,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> TBinaryBinnedAUPRC:
        """
        Update states with the ground truth labels and predictions.

        Args:
            input (Tensor): Tensor of label predictions
                It should be predicted label, probabilities or logits with shape of (num_tasks, n_sample) or (n_sample, ).
            target (Tensor): Tensor of ground truth labels with shape of (num_tasks, n_sample) or (n_sample, ).
        """
        input = input.to(self.device)
        target = target.to(self.device)

        _binary_binned_auprc_update_input_check(
            input, target, self.num_tasks, self.threshold
        )

        if self.num_tasks == 1:
            if input.ndim == 1:
                input = input[None, :]
                target = target[None, :]

        for i in range(self.num_tasks):
            num_tp, num_fp, num_fn = _update(input[i], target[i], self.threshold)
            self.num_tp[i] += num_tp.to(self.device)
            self.num_fp[i] += num_fp.to(self.device)
            self.num_fn[i] += num_fn.to(self.device)

        return self

    @torch.inference_mode()
    def compute(
        self: TBinaryBinnedAUPRC,
    ) -> torch.Tensor:
        """
        Return Binned_AUPRC.  If no ``update()`` calls are made before
        ``compute()`` is called, return an empty tensor.

        Returns:
            - Binned_AUPRC (Tensor): The return value of Binned_AUPRC for each task (num_tasks,) - except if num_tasks = 1,
            in which case we simply return the value as a scalar.
        """

        result = []
        for i in range(self.num_tasks):
            prec, recall, thresh = _binary_binned_precision_recall_curve_compute(
                self.num_tp[i],
                self.num_fp[i],
                self.num_fn[i],
                threshold=self.threshold,
            )
            result.append(_riemann_integral(recall, prec))

        if self.num_tasks == 1:
            result = result[0]
        return torch.tensor(result, device=self.device).nan_to_num(nan=0.0)

    @torch.inference_mode()
    def merge_state(
        self: TBinaryBinnedAUPRC, metrics: Iterable[TBinaryBinnedAUPRC]
    ) -> TBinaryBinnedAUPRC:
        for metric in metrics:
            self.num_tp += metric.num_tp.to(self.device)
            self.num_fp += metric.num_fp.to(self.device)
            self.num_fn += metric.num_fn.to(self.device)
        return self


class MulticlassBinnedAUPRC(Metric[torch.Tensor]):
    """
    Compute Binned AUPRC, which is the area under the binned version of the Precision Recall Curve, for multiclass classification.
    Its functional version is :func:`torcheval.metrics.functional.multiclass_binned_auprc`.

    Args:
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
        >>> from torcheval.metrics import MulticlassBinnedAUPRC
        >>> input = torch.tensor([[0.1, 0.2, 0.1], [0.4, 0.2, 0.1], [0.6, 0.1, 0.2], [0.4, 0.2, 0.3], [0.6, 0.2, 0.4]])
        >>> target = torch.tensor([0, 1, 2, 1, 0])
        >>> metric = MulticlassBinnedAUPRC(num_classes=3, threshold=5, average='macro')
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor(0.35)
        >>> metric = MulticlassBinnedAUPRC(num_classes=3, threshold=5, average=None)
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor([0.4500, 0.4000, 0.2000])
        >>> input = torch.tensor([[0.1, 0.2, 0.1, 0.4], [0.4, 0.2, 0.1, 0.7], [0.6, 0.1, 0.2, 0.4], [0.4, 0.2, 0.3, 0.2], [0.6, 0.2, 0.4, 0.5]])
        >>> target = torch.tensor([0, 1, 2, 1, 0])
        >>> threshold = torch.tensor([0.0, 0.1, 0.4, 0.7, 0.8, 1.0])
        >>> metric = MulticlassBinnedAUPRC(input, target, num_classes=4, threshold=threshold, average='macro')
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor(0.24375)
        >>> metric = MulticlassBinnedAUPRC(input, target, num_classes=4, threshold=threshold, average='macro')
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor([0.3250, 0.2000, 0.2000, 0.2500])
    """

    def __init__(
        self: TMulticlassBinnedAUPRC,
        *,
        num_classes: int,
        threshold: int | list[float] | torch.Tensor = DEFAULT_NUM_THRESHOLD,
        average: str | None = "macro",
        optimization: str = "vectorized",
        device: torch.device | None = None,
    ) -> None:
        super().__init__(device=device)
        _optimization_param_check(optimization)
        threshold = _create_threshold_tensor(
            threshold,
            self.device,
        )
        _multiclass_binned_auprc_param_check(num_classes, threshold, average)
        self.num_classes = num_classes
        self.threshold = threshold
        self.average = average
        self.optimization = optimization
        self._add_state(
            "num_tp",
            torch.zeros(len(threshold), self.num_classes, device=self.device),
        )
        self._add_state(
            "num_fp",
            torch.zeros(len(threshold), self.num_classes, device=self.device),
        )
        self._add_state(
            "num_fn",
            torch.zeros(len(threshold), self.num_classes, device=self.device),
        )

    @torch.inference_mode()
    # pyre-ignore[14]: inconsistent override on *_:Any, **__:Any
    def update(
        self: TMulticlassBinnedAUPRC,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> TMulticlassBinnedAUPRC:
        """
        Update states with the ground truth labels and predictions.

        Args:
            input (Tensor): Tensor of label predictions
                It should be predicted label, probabilities or logits with shape of (num_tasks, n_sample) or (n_sample, ).
            target (Tensor): Tensor of ground truth labels with shape of (num_tasks, n_sample) or (n_sample, ).
        """
        input = input.to(self.device)
        target = target.to(self.device)

        num_tp, num_fp, num_fn = _multiclass_binned_precision_recall_curve_update(
            input,
            target,
            num_classes=self.num_classes,
            threshold=self.threshold,
            optimization=self.optimization,
        )
        self.num_tp += num_tp
        self.num_fp += num_fp
        self.num_fn += num_fn
        return self

    @torch.inference_mode()
    def compute(
        self: TMulticlassBinnedAUPRC,
    ) -> torch.Tensor:
        """
        Return Binned_AUPRC.  If no ``update()`` calls are made before
        ``compute()`` is called, return an empty tensor.

        Returns:
            - Binned_AUPRC (Tensor): The return value of Binned_AUPRC for each task (num_tasks,).
        """
        prec, recall, thresh = _multiclass_binned_precision_recall_curve_compute(
            self.num_tp,
            self.num_fp,
            self.num_fn,
            num_classes=self.num_classes,
            threshold=self.threshold,
        )
        return _compute_riemann_integrals(prec, recall, self.average, self.device)

    @torch.inference_mode()
    def merge_state(
        self: TMultilabelBinnedAUPRC,
        metrics: Iterable[TMultilabelBinnedAUPRC],
    ) -> TMultilabelBinnedAUPRC:
        for metric in metrics:
            self.num_tp += metric.num_tp.to(self.device)
            self.num_fp += metric.num_fp.to(self.device)
            self.num_fn += metric.num_fn.to(self.device)
        return self


class MultilabelBinnedAUPRC(Metric[torch.Tensor]):
    """
    Compute Binned AUPRC, which is the area under the binned version of the Precision Recall Curve, for multilabel classification.
    Its functional version is :func:`torcheval.metrics.functional.multilabel_binned_auprc`.

    Args:
        num_labels (int): Number of labels.
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
            Choose the optimization to use. Accepted values: "vectorized" and "memory". Here are the tradeoffs between these two options:
            - "vectorized": consumes more memory but is faster on some hardware, e.g. modern GPUs.
            - "memory": consumes less memory but can be significantly slower on some hardware, e.g. modern GPUs
            Generally, on GPUs, the "vectorized" optimization requires more memory but is faster; the "memory" optimization requires less memory but is slower.
            On CPUs, the "memory" optimization is recommended in all cases; it uses less memory and is faster.

    Examples::

        >>> import torch
        >>> from torcheval.metrics import MultilabelBinnedAUPRC
        >>> input = torch.tensor([[0.75, 0.05, 0.35], [0.45, 0.75, 0.05], [0.05, 0.55, 0.75], [0.05, 0.65, 0.05]])
        >>> target = torch.tensor([[1, 0, 1], [0, 0, 0], [0, 1, 1], [1, 1, 1]])
        >>> threshold = torch.tensor([0.0, 0.1, 0.4, 0.7, 0.8, 0.9, 1.0])
        >>> metric = MultilabelBinnedAUPRC(num_labels=3, threshold=threshold, average='none')
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor([0.7500, 0.6667, 0.9167])
        >>> metric = MultilabelBinnedAUPRC(num_labels=3, threshold=threshold, average='macro')
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor(0.7778)
        >>> threshold = 5
        >>> metric = MultilabelBinnedAUPRC(num_labels=3, threshold=threshold, average='macro')
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor(0.7778)
        >>> threshold = 100
        >>> metric = MultilabelBinnedAUPRC(num_labels=3, threshold=threshold, average='none')
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor([0.7500, 0.5833, 0.9167])
        >>> metric = MultilabelBinnedAUPRC(num_labels=3, threshold=threshold, average='macro')
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor(0.7500)
    """

    def __init__(
        self: TMultilabelBinnedAUPRC,
        *,
        num_labels: int,
        threshold: int | list[float] | torch.Tensor = DEFAULT_NUM_THRESHOLD,
        average: str | None = "macro",
        optimization: str = "vectorized",
        device: torch.device | None = None,
    ) -> None:
        super().__init__(device=device)
        _optimization_param_check(optimization)
        threshold = _create_threshold_tensor(
            threshold,
            self.device,
        )
        _multilabel_binned_auprc_param_check(num_labels, threshold, average)
        self.num_labels = num_labels
        self.threshold = threshold
        self.average = average
        self.optimization = optimization
        self._add_state(
            "num_tp",
            torch.zeros(len(threshold), self.num_labels, device=self.device),
        )
        self._add_state(
            "num_fp",
            torch.zeros(len(threshold), self.num_labels, device=self.device),
        )
        self._add_state(
            "num_fn",
            torch.zeros(len(threshold), self.num_labels, device=self.device),
        )

    @torch.inference_mode()
    # pyre-ignore[14]: inconsistent override on *_:Any, **__:Any
    def update(
        self: TMultilabelBinnedAUPRC,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> TMultilabelBinnedAUPRC:
        """
        Update states with the ground truth labels and predictions.

        Args:
            input: Tensor of label predictions
                It should be probabilities or logits with shape of (n_sample, ).
            target: Tensor of ground truth labels with shape of (n_samples, ).
        """
        input = input.to(self.device)
        target = target.to(self.device)
        self.threshold = self.threshold.to(self.device)

        num_tp, num_fp, num_fn = _multilabel_binned_precision_recall_curve_update(
            input,
            target,
            num_labels=self.num_labels,
            threshold=self.threshold,
            optimization=self.optimization,
        )
        self.num_tp += num_tp
        self.num_fp += num_fp
        self.num_fn += num_fn
        return self

    @torch.inference_mode()
    def compute(
        self: TMultilabelBinnedAUPRC,
    ) -> torch.Tensor:
        """
        Return Binned_AUPRC.  If no ``update()`` calls are made before
        ``compute()`` is called, return an empty tensor.

        Returns:
            - Binned_AUPRC (Tensor): The return value of Binned_AUPRC for each task (num_tasks,).
        """
        prec, recall, thresh = _multilabel_binned_precision_recall_curve_compute(
            self.num_tp,
            self.num_fp,
            self.num_fn,
            num_labels=self.num_labels,
            threshold=self.threshold,
        )
        return _compute_riemann_integrals(prec, recall, self.average, self.device)

    @torch.inference_mode()
    def merge_state(
        self: TMultilabelBinnedAUPRC,
        metrics: Iterable[TMultilabelBinnedAUPRC],
    ) -> TMultilabelBinnedAUPRC:
        for metric in metrics:
            self.num_tp += metric.num_tp.to(self.device)
            self.num_fp += metric.num_fp.to(self.device)
            self.num_fn += metric.num_fn.to(self.device)
        return self
