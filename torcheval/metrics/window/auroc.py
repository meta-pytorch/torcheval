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

from torcheval.metrics.functional.classification.auroc import (
    _binary_auroc_compute,
    _binary_auroc_update_input_check,
)
from torcheval.metrics.metric import Metric
from torcheval.utils.device import largest_float


TAUROC = TypeVar("TAUROC")


class WindowedBinaryAUROC(Metric[torch.Tensor]):
    """
    The windowed version of BinaryAUROC.
    It is calculated from the input and target of the last window_size number of samples.
    Compute AUROC, which is the area under the ROC Curve, for binary classification.

    Examples::

        >>> import torch
        >>> from torcheval.metrics import WindowedBinaryAUROC
        >>> metric = WindowedBinaryAUROC(max_num_samples=4)
        >>> input = torch.tensor([0.2, 0.5, 0.1, 0.5, 0.7, 0.8])
        >>> target = torch.tensor([0, 1, 1, 0, 1, 1])
        >>> metric.update(input, target)
        >>> metric.inputs
        tensor([0.1, 0.5, 0.7, 0.8])
        >>> metric.targets
        tensor([1, 0, 1, 1])
        >>> metric.compute()
        tensor(0.6667)

        >>> metric = WindowedBinaryAUROC(max_num_samples=5, num_tasks=2)
        >>> metric.update(torch.tensor([[0.2, 0.3], [0.5, 0.1]]), torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
        >>> metric.update(torch.tensor([[0.8, 0.3], [0.6, 0.1]]), torch.tensor([[1.0, 1.0], [1.0, 0.0]]))
        >>> metric.update(torch.tensor([[0.5, 0.1], [0.3, 0.9]]), torch.tensor([[0.0, 1.0], [0.0, 0.0]]))
        >>> metric.inputs
        tensor([[0.1000, 0.3000, 0.8000, 0.3000, 0.5000],
                [0.9000, 0.1000, 0.6000, 0.1000, 0.3000]])
        >>> metric.targets
        tensor([[1., 0., 1., 1., 0.],
                [0., 1., 1., 0., 0.]])
        >>> metric.compute()
        tensor([0.4167, 0.5000])

    """

    def __init__(
        self: TAUROC,
        *,
        num_tasks: int = 1,
        max_num_samples: int = 100,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(device=device)
        if num_tasks < 1:
            raise ValueError(
                "`num_tasks` value should be greater than and equal to 1, but received {num_tasks}. "
            )
        if max_num_samples < 1:
            raise ValueError(
                "`max_num_samples` value should be greater than and equal to 1, but received {max_num_samples}. "
            )
        self.num_tasks = num_tasks
        self._add_state("max_num_samples", max_num_samples)
        self.next_inserted = 0
        self._add_state("total_samples", 0)
        self._add_state(
            "inputs",
            torch.zeros(self.num_tasks, self.max_num_samples, device=self.device),
        )
        self._add_state(
            "targets",
            torch.zeros(self.num_tasks, self.max_num_samples, device=self.device),
        )
        self._add_state(
            "weights",
            torch.zeros(self.num_tasks, self.max_num_samples, device=self.device),
        )

    @torch.inference_mode()
    # pyre-ignore[14]: inconsistent override on *_:Any, **__:Any
    def update(
        self: TAUROC,
        input: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor | None = None,
    ) -> TAUROC:
        """
        Update states with the ground truth labels and predictions.

        Args:
            input (Tensor): Tensor of label predictions
                It should be predicted label, probabilities or logits with shape of (num_samples, )
                or (num_tasks, num_samples).
            target (Tensor): Tensor of ground truth labels with shape of (num_samples, ) or
                or (num_tasks, num_samples).
        """
        if weight is None:
            weight = torch.ones_like(input, dtype=largest_float(input.device))
        _binary_auroc_update_input_check(input, target, self.num_tasks, weight)
        if input.ndim == 1:
            input = input.reshape(1, -1)
            target = target.reshape(1, -1)
            weight = weight.reshape(1, -1)
        # If input size is greater than or equal to window size, replace it with the last max_num_samples size of input.
        if input.shape[1] >= self.max_num_samples:
            self.inputs.copy_(input[:, -self.max_num_samples :].detach())
            self.targets.copy_(target[:, -self.max_num_samples :].detach())
            self.weights.copy_(weight[:, -self.max_num_samples :].detach())
            self.next_inserted = 0
        else:
            rest_window_size = self.max_num_samples - self.next_inserted
            # If input size can fit in the rest of window, replace it.
            if input.shape[1] <= rest_window_size:
                self.inputs[
                    :, self.next_inserted : self.next_inserted + input.shape[1]
                ] = input.detach()
                self.targets[
                    :, self.next_inserted : self.next_inserted + input.shape[1]
                ] = target.detach()
                self.weights[
                    :, self.next_inserted : self.next_inserted + input.shape[1]
                ] = weight.detach()
                self.next_inserted += input.shape[1]
            else:
                # Otherwise, replace with the first half and the second half of input respectively.
                # Put the first half of input to the end
                self.inputs[
                    :, self.next_inserted : self.next_inserted + rest_window_size
                ] = input[:, :rest_window_size].detach()
                self.targets[
                    :, self.next_inserted : self.next_inserted + rest_window_size
                ] = target[:, :rest_window_size].detach()
                self.weights[
                    :, self.next_inserted : self.next_inserted + rest_window_size
                ] = weight[:, :rest_window_size].detach()

                # Put the second half of input to the front
                rest_window_size = input.shape[1] - rest_window_size
                self.inputs[:, :rest_window_size] = input[
                    :, -rest_window_size:
                ].detach()
                self.targets[:, :rest_window_size] = target[
                    :, -rest_window_size:
                ].detach()
                self.weights[:, :rest_window_size] = weight[
                    :, -rest_window_size:
                ].detach()
                self.next_inserted = rest_window_size

        self.next_inserted %= self.max_num_samples
        self.total_samples += input.shape[1]
        return self

    @torch.inference_mode()
    def compute(
        self: TAUROC,
    ) -> torch.Tensor:
        """
        Return AUROC.  If no ``update()`` calls are made before
        ``compute()`` is called, return an empty tensor.

        Returns:
            Tensor: The return value of AUROC for each task (num_tasks,).
        """

        if torch.all(self.inputs[:, self.next_inserted :] == 0):
            return _binary_auroc_compute(
                self.inputs[:, : self.next_inserted].squeeze(),
                self.targets[:, : self.next_inserted].squeeze(),
                self.weights[:, : self.next_inserted].squeeze(),
            )
        else:
            return _binary_auroc_compute(
                self.inputs.squeeze(), self.targets.squeeze(), self.weights.squeeze()
            )

    @torch.inference_mode()
    def merge_state(self: TAUROC, metrics: Iterable[TAUROC]) -> TAUROC:
        """
        Merge the metric state with its counterparts from other metric instances.
        First create tensors of size equal to the sum of all metrics' window sizes.
        Then, put all tensors to the front and leave the remaining indices zeros.

        Args:
            metrics (Iterable[Metric]): metric instances whose states are to be merged.
        """

        merge_max_num_samples = self.max_num_samples
        for metric in metrics:
            merge_max_num_samples += metric.max_num_samples
        cur_inputs = self.inputs
        cur_targets = self.targets
        cur_weights = self.weights
        self.inputs = torch.zeros(
            self.num_tasks,
            merge_max_num_samples,
            device=self.device,
        )
        self.targets = torch.zeros(
            self.num_tasks,
            merge_max_num_samples,
            device=self.device,
        )
        self.weights = torch.zeros(
            self.num_tasks,
            merge_max_num_samples,
            device=self.device,
        )

        cur_size = min(self.total_samples, self.max_num_samples)
        self.inputs[:, :cur_size] = cur_inputs[:, :cur_size]
        self.targets[:, :cur_size] = cur_targets[:, :cur_size]
        self.weights[:, :cur_size] = cur_weights[:, :cur_size]
        idx = cur_size
        for metric in metrics:
            cur_size = min(metric.total_samples, metric.max_num_samples)
            self.inputs[:, idx : idx + cur_size] = metric.inputs[:, :cur_size]
            self.targets[:, idx : idx + cur_size] = metric.targets[:, :cur_size]
            self.weights[:, idx : idx + cur_size] = metric.weights[:, :cur_size]
            self.total_samples += metric.total_samples
            idx += cur_size

        self.max_num_samples = merge_max_num_samples
        self.next_inserted = idx
        self.next_inserted %= self.max_num_samples
        return self
