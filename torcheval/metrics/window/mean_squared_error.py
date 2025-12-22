# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# pyre-ignore-all-errors[16]: Undefined attribute of metric states.

from collections.abc import Iterable
from typing import TypeVar, Union

import torch

from torcheval.metrics.functional.regression.mean_squared_error import (
    _mean_squared_error_compute,
    _mean_squared_error_param_check,
    _mean_squared_error_update,
)
from torcheval.metrics.metric import Metric

TWindowedMeanSquaredError = TypeVar("TWindowedMeanSquaredError")


class WindowedMeanSquaredError(
    Metric[Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]]
):
    r"""
    The windowed version of Mean Squared Error that provides both windowed and liftime values.
    Windowed value is calculated from the input and target of the last window_size number of `update()` calls.
    Lifetime value is calculated from all past input and target of `update()` calls.

    .. math:: \text{MSE} = \frac{1}{N}\sum_i^N(y_i - \hat{y_i})^2

    Where :math:`y` is a tensor of target values, and :math:`\hat{y}` is a tensor of input values.

    Args:
        num_tasks (int): Number of tasks that need WindowedMeanSquaredError calculation. Default value
                    is 1. WindowedMeanSquaredError for each task will be calculated independently.
        max_num_updates (int): The max window size that can accommodate the number of updates.
        enable_lifetime (bool): A boolean indicator whether to calculate lifetime values.
        multioutput (str, Optional)
            - ``'uniform_average'`` [default]: Return scores of all outputs are averaged with uniform weight.
            - ``'raw_values'``: Return a full set of scores.
    Raises:
        ValueError:
            - If value of multioutput is not one of (``raw_values``, ``uniform_average``).
            - If the dimension of `input` or `target` is not 1D or 2D.
            - If the `input` and `target` do not have the same size.
            - If the first dimension of `input`, `target` and `sample_weight` are not the same.

    Examples::
        >>> metric = MeanSquaredError(max_num_updates=1, enable_lifetime=False)
        >>> metric.update(torch.tensor([[0.2, 0.3], [0.4, 0.6]]), torch.tensor([[0.1, 0.3], [0.6, 0.7]]))
        >>> metric.update(torch.tensor([[0.9, 0.5], [0.3, 0.5]]), torch.tensor([[0.5, 0.8], [0.2, 0.8]]))
        >>> metric.compute()
        tensor(0.0875)

        >>> metric = MeanSquaredError(max_num_updates=1, enable_lifetime=True)
        >>> metric.update(torch.tensor([[0.2, 0.3], [0.4, 0.6]]), torch.tensor([[0.1, 0.3], [0.6, 0.7]]))
        >>> metric.update(torch.tensor([[0.9, 0.5], [0.3, 0.5]]), torch.tensor([[0.5, 0.8], [0.2, 0.8]]))
        >>> metric.compute()
        (tensor(0.0512), tensor(0.0875))

        >>> metric = MeanSquaredError(max_num_updates=1, enable_lifetime=False, multioutput="raw_values")
        >>> metric.update(torch.tensor([[0.2, 0.3], [0.4, 0.6]]), torch.tensor([[0.1, 0.3], [0.6, 0.7]]))
        >>> metric.update(torch.tensor([[0.9, 0.5], [0.3, 0.5]]), torch.tensor([[0.5, 0.8], [0.2, 0.8]]))
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor([0.0850, 0.0900])
    """

    def __init__(
        self: TWindowedMeanSquaredError,
        *,
        num_tasks: int = 1,
        max_num_updates: int = 100,
        enable_lifetime: bool = True,
        multioutput: str = "uniform_average",
        device: torch.device | None = None,
    ) -> None:
        super().__init__(device=device)
        _mean_squared_error_param_check(multioutput)
        if num_tasks < 1:
            raise ValueError(
                "`num_tasks` value should be greater than and equal to 1, but received {num_tasks}. "
            )
        if max_num_updates < 1:
            raise ValueError(
                "`max_num_updates` value should be greater than and equal to 1, but received {max_num_updates}. "
            )
        self.num_tasks = num_tasks
        self._add_state("max_num_updates", max_num_updates)
        self.enable_lifetime = enable_lifetime
        self.multioutput = multioutput
        self.next_inserted = 0
        self._add_state("total_updates", 0)

        if self.enable_lifetime:
            self._add_state(
                "sum_squared_error",
                torch.tensor(0.0, device=self.device),
            )
            self._add_state("sum_weight", torch.tensor(0.0, device=self.device))
        self._add_state(
            "windowed_sum_squared_error",
            torch.zeros(
                self.num_tasks,
                self.max_num_updates,
                device=self.device,
            ),
        )
        self._add_state(
            "windowed_sum_weight",
            torch.zeros(
                self.num_tasks,
                self.max_num_updates,
                device=self.device,
            ),
        )

    @torch.inference_mode()
    # pyre-ignore[14]: inconsistent override on *_:Any, **__:Any
    def update(
        self: TWindowedMeanSquaredError,
        input: torch.Tensor,
        target: torch.Tensor,
        *,
        sample_weight: torch.Tensor | None = None,
    ) -> TWindowedMeanSquaredError:
        """
        Update states with the ground truth values and predictions.

        Args:
            input (Tensor): Tensor of predicted values with shape of (n_sample, n_output).
            target (Tensor): Tensor of ground truth values with shape of (n_sample, n_output).
            sample_weight (Optional):
                Tensor of sample weights with shape of (n_sample, ). Defaults to None.
        """

        (
            sum_squared_error,
            sum_weight,
        ) = _mean_squared_error_update(input, target, sample_weight)
        self._window_mean_squared_error_update_input_check(
            input, target, sample_weight, self.num_tasks
        )
        if self.enable_lifetime:
            if self.sum_squared_error.ndim == 0 and sum_squared_error.ndim == 1:
                self.sum_squared_error = sum_squared_error
            else:
                self.sum_squared_error += sum_squared_error
            self.sum_weight += sum_weight
        self.windowed_sum_squared_error[:, self.next_inserted] = sum_squared_error
        self.windowed_sum_weight[:, self.next_inserted] = sum_weight
        self.next_inserted += 1
        self.next_inserted %= self.max_num_updates
        self.total_updates += 1
        return self

    @torch.inference_mode()
    def compute(
        self: TWindowedMeanSquaredError,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Return the Mean Squared Error.

        Empty tensor is returned if no calls to ``update()`` are made before ``compute()`` is called.
        """
        if self.total_updates == 0:
            naught = torch.empty(0, device=self.device)
            if self.enable_lifetime:
                return naught, naught
            else:
                return naught

        windowed_sum_squared_error = self.windowed_sum_squared_error.sum(dim=1)
        windowed_sum_weight = self.windowed_sum_weight.sum(dim=1)
        windowed_mse = _mean_squared_error_compute(
            windowed_sum_squared_error,
            self.multioutput,
            windowed_sum_weight,
        )

        if self.enable_lifetime:
            lifetime_mse = _mean_squared_error_compute(
                self.sum_squared_error,
                self.multioutput,
                self.sum_weight,
            )
            return (
                lifetime_mse.squeeze(),
                windowed_mse.squeeze(),
            )
        else:
            return windowed_mse.squeeze()

    @torch.inference_mode()
    def merge_state(
        self: TWindowedMeanSquaredError, metrics: Iterable[TWindowedMeanSquaredError]
    ) -> TWindowedMeanSquaredError:
        merge_max_num_updates = self.max_num_updates
        for metric in metrics:
            merge_max_num_updates += metric.max_num_updates
        cur_sum_squared_error = self.windowed_sum_squared_error
        cur_sum_weight = self.windowed_sum_weight
        self.windowed_sum_squared_error = torch.zeros(
            self.num_tasks,
            merge_max_num_updates,
            dtype=torch.float32,
            device=self.device,
        )
        self.windowed_sum_weight = torch.zeros(
            self.num_tasks,
            merge_max_num_updates,
            dtype=torch.float32,
            device=self.device,
        )

        cur_size = min(self.total_updates, self.max_num_updates)
        self.windowed_sum_squared_error[:, :cur_size] = cur_sum_squared_error[
            :, :cur_size
        ]
        self.windowed_sum_weight[:, :cur_size] = cur_sum_weight[:, :cur_size]
        idx = cur_size

        for metric in metrics:
            if self.enable_lifetime:
                if (
                    self.sum_squared_error.ndim == 0
                    and metric.sum_squared_error.ndim == 1
                ):
                    self.sum_squared_error = metric.sum_squared_error.to(self.device)
                else:
                    self.sum_squared_error += metric.sum_squared_error.to(self.device)
                self.sum_weight += metric.sum_weight.to(self.device)
            cur_size = min(metric.total_updates, metric.max_num_updates)
            self.windowed_sum_squared_error[:, idx : idx + cur_size] = (
                metric.windowed_sum_squared_error[:, :cur_size]
            )
            self.windowed_sum_weight[:, idx : idx + cur_size] = (
                metric.windowed_sum_weight[:, :cur_size]
            )
            idx += cur_size
            self.total_updates += metric.total_updates

        self.next_inserted = idx
        self.next_inserted %= self.max_num_updates
        return self

    def _window_mean_squared_error_update_input_check(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        sample_weight: torch.Tensor | None,
        num_tasks: int = 1,
    ) -> None:
        if num_tasks == 1:
            if len(input.shape) > 1:
                raise ValueError(
                    f"`num_tasks = 1`, `input` is expected to be one-dimensional tensor, but got shape ({input.shape})."
                )
        elif len(input.shape) == 1 or input.shape[1] != num_tasks:
            raise ValueError(
                f"`num_tasks = {num_tasks}`, `input`'s shape is expected to be (num_samples, {num_tasks}), but got shape ({input.shape})."
            )
