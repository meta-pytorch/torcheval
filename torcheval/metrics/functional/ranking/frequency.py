# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch


@torch.inference_mode()
def frequency_at_k(
    input: torch.Tensor,
    k: float,
) -> torch.Tensor:
    """
    Calculate the frequency given a list of frequencies and threshold k.
    Generate a binary list to indicate if frequencies is less than k.

    Args:
        input (Tensor): Predicted unnormalized scores (often referred to as logits).
        k (float): Threshold of the frequency. k should not negative value.

    Example:
        >>> import torch
        >>> from torcheval.metrics.functional import frequency
        >>> input = torch.tensor([0.3, 0.1, 0.6])
        >>> frequency(input, k=0.5)
        tensor([1.0000, 1.0000, 0.0000])
    """
    _frequency_input_check(input, k)

    return (input < k).float()


def _frequency_input_check(input: torch.Tensor, k: float) -> None:
    if input.ndim != 1:
        raise ValueError(
            f"input should be a one-dimensional tensor, got shape {input.shape}."
        )
    if k < 0:
        raise ValueError(f"k should not be negative, got {k}.")
