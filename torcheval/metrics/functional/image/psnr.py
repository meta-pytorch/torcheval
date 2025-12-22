# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import torch


@torch.inference_mode()
def peak_signal_noise_ratio(
    input: torch.Tensor,
    target: torch.Tensor,
    data_range: float | None = None,
) -> torch.Tensor:
    """
    Compute the peak signal-to-noise ratio between two images.
    It's class version is :obj:`torcheval.metrics.PeakSignalNoiseRatio`.

    Args:
        input (Tensor): Input image ``(N, C, H, W)``.
        target (Tensor): Target image ``(N, C, H, W)``.
        data_range (float): the range of the input images. Default: None.
            If None, the input range computed from the target data ``(target.max() - targert.min())``.
    Examples::

        >>> import torch
        >>> from torcheval.metrics.functional import peak_signal_noise_ratio
        >>> input = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
        >>> target = input * 0.9
        >>> peak_signal_noise_ratio(input, target)
        tensor(19.8767)
    """
    _psnr_param_check(data_range)

    if data_range is None:
        data_range_tensor = torch.max(target) - torch.min(target)
    else:
        data_range_tensor = torch.tensor(data=data_range, device=target.device)

    sum_square_error, num_observations = _psnr_update(input, target)
    psnr = _psnr_compute(sum_square_error, num_observations, data_range_tensor)
    return psnr


def _psnr_param_check(data_range: float | None) -> None:
    # Check matching shapes
    if data_range is not None:
        if type(data_range) is not float:
            raise ValueError("`data_range needs to be either `None` or `float`.")
        if data_range <= 0:
            raise ValueError("`data_range` needs to be positive.")


def _psnr_input_check(input: torch.Tensor, target: torch.Tensor) -> None:
    # Check matching shapes
    if input.shape != target.shape:
        raise ValueError(
            "The `input` and `target` must have the same shape, "
            f"got shapes {input.shape} and {target.shape}."
        )


def _psnr_update(
    input: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    _psnr_input_check(input, target)
    sum_squared_error = torch.sum(torch.pow(input - target, 2))
    num_observations = torch.tensor(target.numel(), device=target.device)
    return sum_squared_error, num_observations


def _psnr_compute(
    sum_square_error: torch.Tensor,
    num_observations: torch.Tensor,
    data_range: torch.Tensor,
) -> torch.Tensor:
    mse = sum_square_error / num_observations
    psnr = 10 * torch.log10(torch.pow(data_range, 2) / mse)

    return psnr
