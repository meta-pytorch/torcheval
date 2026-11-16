# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch


@torch.inference_mode()
def num_collisions(input: torch.Tensor) -> torch.Tensor:
    """
    Compute the number of collisions given a list of input(ids).

    Args:
        input (Tensor): a tensor of input ids (num_samples, ).
            class probabilities of shape (num_samples, num_classes).

    Examples::

        >>> import torch
        >>> from torcheval.metrics.functional import num_collisions
        >>> input = torch.tensor([3, 4, 2, 3])
        >>> num_collisions(input)
        tensor([1, 0, 0, 1])
        >>> input = torch.tensor([3, 4, 1, 3, 1, 1, 5])
        >>> num_collisions(input)
        tensor([1, 0, 2, 1, 2, 2, 0])
    """
    _num_collisions_input_check(input)

    input_for_logits = input.view(1, -1).repeat_interleave(torch.numel(input), dim=0)
    num_collisions = (input_for_logits == input.view(-1, 1)).sum(
        dim=1, keepdim=True
    ) - 1
    return num_collisions.view(-1)


def _num_collisions_input_check(input: torch.Tensor) -> None:
    if input.ndim != 1:
        raise ValueError(
            f"input should be a one-dimensional tensor, got shape {input.shape}."
        )

    if input.dtype not in (
        torch.int,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        raise ValueError(f"input should be an integer tensor, got {input.dtype}.")
