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
from torcheval.metrics.functional.regression.mean_squared_error import (
    _mean_squared_error_compute,
    _mean_squared_error_param_check,
    _mean_squared_error_update,
)
from torcheval.metrics.metric import Metric

TMeanSquaredError = TypeVar("TMeanSquaredError")


class MeanSquaredError(Metric[torch.Tensor]):
    """
    Compute Mean Squared Error, which is the mean of squared error of `input` and `target`.
    Its functional version is :func:`torcheval.metrics.functional.mean_squared_error`.

    Args:
        multioutput (str, Optional)
            - ``'uniform_average'`` [default]: Return scores of all outputs are averaged with uniform weight.
            - ``'raw_values'``: Return a full set of scores.
    Raises:
        ValueError:
            - If value of multioutput does not exist in (``raw_values``, ``uniform_average``).
            - If the dimension of `input` or `target` is not 1D or 2D.
            - If the `input` and `target` do not have the same size.
            - If the first dimension of `input`, `target` and `sample_weight` are not the same.

    Examples::

        >>> import torch
        >>> from torcheval.metrics import MeanSquaredError
        >>> metric = MeanSquaredError()
        >>> input = torch.tensor([0.9, 0.5, 0.3, 0.5])
        >>> target = torch.tensor([0.5, 0.8, 0.2, 0.8])
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor(0.0875)

        >>> metric = MeanSquaredError()
        >>> input = torch.tensor([[0.9, 0.5], [0.3, 0.5]])
        >>> target = torch.tensor([[0.5, 0.8], [0.2, 0.8]])
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor(0.0875)

        >>> metric = MeanSquaredError(multioutput="raw_values")
        >>> input = torch.tensor([[0.9, 0.5], [0.3, 0.5]])
        >>> target = torch.tensor([[0.5, 0.8], [0.2, 0.8]])
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor([0.0850, 0.0900])

        >>> input = torch.tensor([[0.9, 0.5], [0.3, 0.5]])
        >>> target = torch.tensor([[0.5, 0.8], [0.2, 0.8]])
        >>> metric.update(input, target, sample_weight=torch.tensor([0.2, 0.8]))
        >>> metric.compute()
        tensor(0.0650)
    """

    def __init__(
        self: TMeanSquaredError,
        *,
        multioutput: str = "uniform_average",
        device: torch.device | None = None,
    ) -> None:
        super().__init__(device=device)
        _mean_squared_error_param_check(multioutput)
        self.multioutput = multioutput
        self._add_state(
            "sum_squared_error",
            torch.tensor(0.0, device=self.device),
        )
        self._add_state("sum_weight", torch.tensor(0.0, device=self.device))

    @torch.inference_mode()
    # pyre-ignore[14]: inconsistent override on *_:Any, **__:Any
    def update(
        self: TMeanSquaredError,
        input: torch.Tensor,
        target: torch.Tensor,
        *,
        sample_weight: torch.Tensor | None = None,
    ) -> TMeanSquaredError:
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
        if self.sum_squared_error.ndim == 0 and sum_squared_error.ndim == 1:
            self.sum_squared_error = sum_squared_error
        else:
            self.sum_squared_error += sum_squared_error
        self.sum_weight += sum_weight
        return self

    @torch.inference_mode()
    def compute(self: TMeanSquaredError) -> torch.Tensor:
        """
        Return the Mean Squared Error.

        NaN is returned if no calls to ``update()`` are made before ``compute()`` is called.
        """
        return _mean_squared_error_compute(
            self.sum_squared_error,
            self.multioutput,
            self.sum_weight,
        )

    @torch.inference_mode()
    def merge_state(
        self: TMeanSquaredError, metrics: Iterable[TMeanSquaredError]
    ) -> TMeanSquaredError:
        for metric in metrics:
            if self.sum_squared_error.ndim == 0 and metric.sum_squared_error.ndim == 1:
                self.sum_squared_error = metric.sum_squared_error.to(self.device)
            else:
                self.sum_squared_error += metric.sum_squared_error.to(self.device)
            self.sum_weight += metric.sum_weight.to(self.device)
        return self
