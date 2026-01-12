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
from torcheval.metrics.functional.image.psnr import (
    _psnr_compute,
    _psnr_param_check,
    _psnr_update,
)
from torcheval.metrics.metric import Metric


TPeakSignalNoiseRatio = TypeVar("TPeakSignalNoiseRatio")


class PeakSignalNoiseRatio(Metric[torch.Tensor]):
    """
    Compute the PSNR (Peak Signal to Noise Ratio) between two images.
    Its functional version is `torcheval.metrics.functional.peak_signal_noise_ratio`

    Args:
        data_range (float): the range of the input images. Default: None.
            If ``None``, the range computed from the target data ``(target.max() - targert.min())``.

    Examples::

        >>> import torch
        >>> from torcheval.metrics import PeakSignalNoiseRatio
        >>> metric = PeakSignalNoiseRatio()
        >>> input = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
        >>> target = input * 0.9
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor(19.8767)

    """

    def __init__(
        self: TPeakSignalNoiseRatio,
        data_range: float | None = None,
        *,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(device=device)

        _psnr_param_check(data_range=data_range)
        if data_range is None:
            self.auto_range = True
            data_range = 0.0
        else:
            self.auto_range = False

        self._add_state("data_range", torch.tensor(data_range, device=self.device))
        self._add_state("num_observations", torch.tensor(0.0, device=self.device))
        self._add_state("sum_squared_error", torch.tensor(0.0, device=self.device))
        self._add_state("min_target", torch.tensor(torch.inf, device=self.device))
        self._add_state("max_target", torch.tensor(-torch.inf, device=self.device))

    @torch.inference_mode()
    # pyre-ignore[14]: `update` overrides method defined in `Metric` inconsistently.
    def update(
        self: TPeakSignalNoiseRatio,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> TPeakSignalNoiseRatio:
        """
        Update the metric state with new input.

        Args:
            input (Tensor): Input image ``(N, C, H, W)``.
            target (Tensor): Target image ``(N, C, H, W)``.
        """
        sum_squared_error, num_observations = _psnr_update(input, target)

        self.sum_squared_error = self.sum_squared_error + sum_squared_error
        self.num_observations = self.num_observations + num_observations
        if self.auto_range:
            self._update_range(target)

        return self

    def _update_range(self, target: torch.Tensor) -> None:
        """
        updates data_range in cases where auto_range is True
        """
        self.min_target = torch.minimum(target.min(), self.min_target)
        self.max_target = torch.maximum(target.max(), self.max_target)
        self.data_range = self.max_target - self.min_target

    @torch.inference_mode()
    def compute(self: TPeakSignalNoiseRatio) -> torch.Tensor:
        """
        Return the peak signal-to-noise ratio.
        """
        return _psnr_compute(
            self.sum_squared_error, self.num_observations, self.data_range
        )

    @torch.inference_mode()
    def merge_state(
        self: TPeakSignalNoiseRatio, metrics: Iterable[TPeakSignalNoiseRatio]
    ) -> TPeakSignalNoiseRatio:
        """
        Merge the metric state with its counterparts from other metric instances.

        Args:
            metrics (Iterable[Metric]): metric instances whose states are to be merged.
        """
        for metric in metrics:
            self.num_observations = self.num_observations + metric.num_observations.to(
                self.device
            )
            self.sum_squared_error = (
                self.sum_squared_error + metric.sum_squared_error.to(self.device)
            )
            if self.auto_range:
                self.min_target = torch.minimum(self.min_target, metric.min_target)
                self.max_target = torch.maximum(self.max_target, metric.max_target)

        if self.auto_range:
            self.data_range = self.max_target - self.min_target

        return self
