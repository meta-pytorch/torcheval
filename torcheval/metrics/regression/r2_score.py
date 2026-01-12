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
from torcheval.metrics.functional.regression.r2_score import (
    _r2_score_compute,
    _r2_score_param_check,
    _r2_score_update,
)
from torcheval.metrics.metric import Metric

TR2Score = TypeVar("TR2Score")


class R2Score(Metric[torch.Tensor]):
    """
    Compute R-squared score, which is the proportion of variance in the dependent variable that can be explained by the independent variable.
    Its functional version is :func:`torcheval.metrics.functional.r2_score`.

    Args:
        multioutput (str, Optional):
            - ``'uniform_average'`` [default]:
              Return scores of all outputs are averaged with uniform weight.
            - ``'raw_values'``:
              Return a full set of scores.
            - ``variance_weighted``:
              Return scores of all outputs are averaged with weighted by the variances of each individual output.
        num_regressors (int, Optional):
            Number of independent variables used, applied to adjusted R-squared score. Defaults to zero (standard R-squared score).

    Raises:
        ValueError:
            - If value of multioutput does not exist in (``raw_values``, ``uniform_average``, ``variance_weighted``).
            - If value of num_regressors is not an ``integer`` in the range of [0, n_samples - 1].

    Examples::

        >>> import torch
        >>> from torcheval.metrics import R2Score
        >>> metric = R2Score()
        >>> input = torch.tensor([0, 2, 1, 3])
        >>> target = torch.tensor([0, 1, 2, 3])
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor(0.6)

        >>> metric = R2Score()
        >>> input = torch.tensor([[0, 2], [1, 6]])
        >>> target = torch.tensor([[0, 1], [2, 5]])
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor(0.6250)

        >>> metric = R2Score(multioutput="raw_values")
        >>> input = torch.tensor([[0, 2], [1, 6]])
        >>> target = torch.tensor([[0, 1], [2, 5]])
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor([0.5000, 0.7500])

        >>> metric = R2Score(multioutput="variance_weighted")
        >>> input = torch.tensor([[0, 2], [1, 6]])
        >>> target = torch.tensor([[0, 1], [2, 5]])
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor(0.7000)

        >>> metric = R2Score(multioutput="raw_values", num_regressors=2)
        >>> input = torch.tensor([1.2, 2.5, 3.6, 4.5, 6])
        >>> target = torch.tensor([1, 2, 3, 4, 5])
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor(0.6200)
    """

    def __init__(
        self: TR2Score,
        *,
        multioutput: str = "uniform_average",
        num_regressors: int = 0,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(device=device)
        _r2_score_param_check(multioutput, num_regressors)
        self.multioutput = multioutput
        self.num_regressors = num_regressors
        self._add_state("sum_squared_obs", torch.tensor(0.0, device=self.device))
        self._add_state("sum_obs", torch.tensor(0.0, device=self.device))
        self._add_state(
            "sum_squared_residual",
            torch.tensor(0.0, device=self.device),
        )
        self._add_state("num_obs", torch.tensor(0.0, device=self.device))

    @torch.inference_mode()
    # pyre-ignore[14]: inconsistent override on *_:Any, **__:Any
    def update(
        self: TR2Score,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> TR2Score:
        """
        Update states with the ground truth values and predictions.

        Args:
            input (Tensor): Tensor of predicted values with shape of (n_sample, n_output).
            target (Tensor): Tensor of ground truth values with shape of (n_sample, n_output).
        """
        (
            sum_squared_obs,
            sum_obs,
            sum_squared_residual,
            num_obs,
        ) = _r2_score_update(input, target)
        if self.sum_squared_obs.ndim == 0 and sum_squared_obs.ndim == 1:
            self.sum_squared_obs = sum_squared_obs
            self.sum_obs = sum_obs
            self.sum_squared_residual = sum_squared_residual
        else:
            self.sum_squared_obs += sum_squared_obs
            self.sum_obs += sum_obs
            self.sum_squared_residual += sum_squared_residual
        self.num_obs += num_obs
        return self

    @torch.inference_mode()
    def compute(self: TR2Score) -> torch.Tensor:
        """
        Return the R-squared score.

        NaN is returned if no calls to ``update()`` are made before ``compute()`` is called.
        """
        return _r2_score_compute(
            self.sum_squared_obs,
            self.sum_obs,
            self.sum_squared_residual,
            self.num_obs,
            self.multioutput,
            self.num_regressors,
        )

    @torch.inference_mode()
    def merge_state(self: TR2Score, metrics: Iterable[TR2Score]) -> TR2Score:
        for metric in metrics:
            if self.sum_squared_obs.ndim == 0 and metric.sum_squared_obs.ndim == 1:
                self.sum_squared_obs = metric.sum_squared_obs.to(self.device)
                self.sum_obs = metric.sum_obs.to(self.device)
                self.sum_squared_residual = metric.sum_squared_residual.to(self.device)
            else:
                self.sum_squared_obs += metric.sum_squared_obs.to(self.device)
                self.sum_obs += metric.sum_obs.to(self.device)
                self.sum_squared_residual += metric.sum_squared_residual.to(self.device)
            self.num_obs += metric.num_obs.to(self.device)
        return self
