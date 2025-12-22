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

from torcheval.metrics.functional.regression.pearson_correlation_coefficient import (
    _pearson_correlation_coefficient_compute, _pearson_correlation_coefficient_param_check,
    _pearson_correlation_coefficient_update)
from torcheval.metrics.metric import Metric

TPearsonCorrelationCoefficient = TypeVar("TPearsonCorrelationCoefficient")


class PearsonCorrelationCoefficient(Metric[torch.Tensor]):
    """
    Compute Pearson Correlation Coefficient.
    Its functional version is :func:`torcheval.metrics.functional.pearson_correlation`.

    Args:
        num_regressors (int, Optional):
            Number of independent variables used, applied to adjusted R-squared score. Defaults to zero (standard R-squared score).

    Raises:
        ValueError:
            - If value of multioutput does not exist in (``raw_values``, ``uniform_average``).
            - If value of num_regressors is not an ``integer`` in the range of [0, n_samples - A1].

    Examples::

        >>> import torch
        >>> from torcheval.metrics import PearsonCorrelationCoefficient
        >>> metric = PearsonCorrelationCoefficient()
        >>> input = torch.tensor([0.9, 0.5])
        >>> target = torch.tensor([0.5, 0.8])
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor(-1.0000)
        >>> input = torch.tensor([0.3, 0.5])
        >>> target = torch.tensor([0.2, 0.8])
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor(0.2075)

        >>> metric = PearsonCorrelationCoefficient()
        >>> input = torch.tensor([[0.9, 0.5], [0.3, 0.5]])
        >>> target = torch.tensor([[0.5, 0.8], [0.2, 0.8]])
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor(0.5000)

        >>> metric = PearsonCorrelationCoefficient(multioutput="raw_values")
        >>> input = torch.tensor([[0.9, 0.5], [0.3, 0.5]])
        >>> target = torch.tensor([[0.5, 0.8], [0.2, 0.8]])
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor([1.0000, 0.0000])
    """

    def __init__(
        self: TPearsonCorrelationCoefficient,
        *,
        multioutput: str = "uniform_average",
        device: torch.device | None = None,
    ) -> None:
        super().__init__(device=device)
        _pearson_correlation_coefficient_param_check(multioutput)
        self.multioutput = multioutput
        self._add_state("sum_input", torch.tensor(0.0, device=self.device))
        self._add_state("sum_target", torch.tensor(0.0, device=self.device))
        self._add_state("sum_input_target", torch.tensor(0.0, device=self.device))
        self._add_state("sum_input_squared", torch.tensor(0.0, device=self.device))
        self._add_state("sum_target_squared", torch.tensor(0.0, device=self.device))
        self._add_state("num_samples", torch.tensor(0, device=self.device))

    @torch.inference_mode()
    # pyre-ignore[14]: inconsistent override on *_:Any, **__:Any
    def update(
        self: TPearsonCorrelationCoefficient,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> TPearsonCorrelationCoefficient:
        """
        Update states with the ground truth values and predictions.

        Args:
            input (Tensor): Tensor of predicted values with shape of (n_sample, n_output).
            target (Tensor): Tensor of ground truth values with shape of (n_sample, n_output).
        """
        (
            sum_input,
            sum_target,
            sum_input_target,
            sum_input_squared,
            sum_target_squared,
            num_samples,
        ) = _pearson_correlation_coefficient_update(input, target)
        if self.num_samples == 0:
            self.sum_input = sum_input
            self.sum_target = sum_target
            self.sum_input_target = sum_input_target
            self.sum_input_squared = sum_input_squared
            self.sum_target_squared = sum_target_squared
        else:
            self.sum_input += sum_input
            self.sum_target += sum_target
            self.sum_input_target += sum_input_target
            self.sum_input_squared += sum_input_squared
            self.sum_target_squared += sum_target_squared
        self.num_samples += num_samples
        return self

    @torch.inference_mode()
    def compute(self: TPearsonCorrelationCoefficient) -> torch.Tensor:
        """
        Return the Pearson Correlation Coefficient.

        NaN is returned if no calls to ``update()`` are made before ``compute()`` is called.
        """
        return _pearson_correlation_coefficient_compute(
            self.sum_input,
            self.sum_target,
            self.sum_input_target,
            self.sum_input_squared,
            self.sum_target_squared,
            self.num_samples,
            self.multioutput,
        )

    @torch.inference_mode()
    def merge_state(
        self: TPearsonCorrelationCoefficient,
        metrics: Iterable[TPearsonCorrelationCoefficient],
    ) -> TPearsonCorrelationCoefficient:
        for metric in metrics:
            if self.num_samples == 0:
                self.sum_input = metric.sum_input.to(self.device)
                self.sum_target = metric.sum_target.to(self.device)
                self.sum_input_target = metric.sum_input_target.to(self.device)
                self.sum_input_squared = metric.sum_input_squared.to(self.device)
                self.sum_target_squared = metric.sum_target_squared.to(self.device)
            else:
                self.sum_input += metric.sum_input.to(self.device)
                self.sum_target += metric.sum_target.to(self.device)
                self.sum_input_target += metric.sum_input_target.to(self.device)
                self.sum_input_squared += metric.sum_input_squared.to(self.device)
                self.sum_target_squared += metric.sum_target_squared.to(self.device)
            self.num_samples += metric.num_samples.to(self.device)
        return self
