# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch


@torch.inference_mode()
def hit_rate(
    input: torch.Tensor,
    target: torch.Tensor,
    *,
    k: int | None = None,
) -> torch.Tensor:
    """
    Compute the hit rate of the correct class among the top predicted classes.
    Its class version is ``torcheval.metrics.HitRate``.

    Args:
        input (Tensor): Predicted unnormalized scores (often referred to as logits) or
            class probabilities of shape (num_samples, num_classes).
        target (Tensor): Ground truth class indices of shape (num_samples,).
        k (int, optional): Number of top predicted classes to be considered.
            If k is None, all classes are considered and a hit rate of 1.0 is returned.

    Examples::

        >>> import torch
        >>> from torcheval.metrics.functional import hit_rate
        >>> input = torch.tensor([[0.3, 0.1, 0.6], [0.5, 0.2, 0.3], [0.2, 0.1, 0.7], [0.3, 0.3, 0.4]])
        >>> target = torch.tensor([2, 1, 1, 0])
        >>> hit_rate(input, target, k=2)
        tensor([1.0000, 0.0000, 0.0000, 1.0000])
    """
    _hit_rate_input_check(input, target, k)
    if k is None or k >= input.size(dim=-1):
        return input.new_ones(target.size())

    y_score = torch.gather(input, dim=-1, index=target.unsqueeze(dim=-1))
    rank = torch.gt(input, y_score).sum(dim=-1)
    return (rank < k).float()


def _hit_rate_input_check(
    input: torch.Tensor, target: torch.Tensor, k: int | None = None
) -> None:
    if target.ndim != 1:
        raise ValueError(
            f"target should be a one-dimensional tensor, got shape {target.shape}."
        )
    if input.ndim != 2:
        raise ValueError(
            f"input should be a two-dimensional tensor, got shape {input.shape}."
        )
    if input.shape[0] != target.shape[0]:
        raise ValueError(
            "`input` and `target` should have the same minibatch dimension, ",
            f"got shapes {input.shape} and {target.shape}, respectively.",
        )
    if k is not None and k <= 0:
        raise ValueError(f"k should be None or positive, got {k}.")
