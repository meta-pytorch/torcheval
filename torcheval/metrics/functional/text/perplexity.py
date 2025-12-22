# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from copy import deepcopy

import torch
import torch.nn.functional as F

from torcheval.utils.device import largest_float


@torch.inference_mode()
def perplexity(
    input: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int | None = None,
) -> torch.Tensor:
    """
    Perplexity measures how well a model predicts sample data. It is calculated by:

    perplexity = exp (sum of negative log likelihood / number of tokens)

    Its class version is :obj:`torcheval.metrics.text.Perplexity`.

    Args:
        input (Tensor):
            Predicted unnormalized scores (i.e., logits) for each token with shape
            of (n_samples, seq_len, vocab_size)
        target (Tensor):
            Tensor of ground truth vocab index with shape of (n_samples, seq_len).
        ignore_index (Tensor):
            if specified, the target class with 'ignore_index' will be ignored when
            calculating perplexity. The default value is None.

    Returns:
       (Tensor): perplexity for the input and target.

    Examples:
        >>> import torch
        >>> from torcheval.metrics.functional.text import perplexity

        >>> input = torch.tensor([[[0.3659, 0.7025, 0.3104], [0.0097, 0.6577, 0.1947]]])
        >>> target = torch.tensor([[2, 1]])
        >>> perplexity(input, target)
        tensor(2.7593, dtype=torch.float64)

        >>> input = torch.tensor([[[0.3, 0.7, 0.3, 0.1], [0.5, 0.4, 0.1, 0.4],[0.1, 0.1, 0.2, 0.5]], [[0.1, 0.6, 0.1, 0.5], [0.3, 0.7, 0.3, 0.4], [0.3, 0.7, 0.3, 0.4]]])
        >>> target = torch.tensor([[2, 1, 3],  [1, 0, 1]])
        >>> perplexity(input, target)
        tensor(3.6216, dtype=torch.float64)

        >>> input = torch.tensor([[[0.3659, 0.7025, 0.3104], [0.0097, 0.6577, 0.1947]]])
        >>> target = torch.tensor([[2, 1]])
        >>> perplexity(input, target, ignore_index = 1)
        tensor(3.5372, dtype=torch.float64)

    """

    sum_log_probs, num_total = _perplexity_update(input, target, ignore_index)

    return _perplexity_compute(sum_log_probs, num_total)


def _perplexity_update(
    input: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sums the log probabilities of the inputs tokens given the target tokens, and counts
    the total number of tokens.

    Args:
        input (Tensor):
            Predicted unnormalized scores (i.e., logits) for each token with shape
            of (n_samples, seq_len, vocab_size).
        target (Tensor):
            Tensor of ground truth vocab index with shape of (n_samples, seq_len).
        ignore_index (Tensor):
            if specified, the target class with 'ignore_index' will be ignored when
            calculating perplexity.

    Returns:
        Tuple(Tensor, Tensor):
            the summed log propabilities 'sum_log_probs' and the number of tokens 'num_total'.
    """

    _perplexity_input_check(input, target, ignore_index)

    dtype = largest_float(target.device)
    probs = input.reshape(-1, input.shape[-1]).type(dtype)
    probs = F.softmax(probs, dim=1)

    target = target.reshape(-1)

    if ignore_index is not None:
        mask = target.ne(ignore_index)
        probs = probs[mask]
        target = target[mask]

    probs = probs[:, target].diagonal()

    sum_log_probs = -probs.log().sum()
    num_total = torch.tensor(target.size(0), device=target.device)

    return sum_log_probs, num_total


def _perplexity_compute(
    sum_log_probs: torch.Tensor,
    num_total: torch.Tensor,
) -> torch.Tensor:
    return torch.exp(sum_log_probs / num_total)


def _perplexity_input_check(
    input: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int | None = None,
) -> None:
    if target.ndim != 2:
        raise ValueError(
            f"target should be a two-dimensional tensor, got shape {target.shape}."
        )

    if input.ndim != 3:
        raise ValueError(
            f"input should be a three-dimensional tensor, got shape {input.shape}."
        )

    if input.size(0) != target.size(0):
        raise ValueError(
            "The `input` and `target` should have the same first dimension (i.e., batch size), "
            f"got shapes {input.shape} and {target.shape} instead."
        )

    if input.size(1) != target.size(1):
        raise ValueError(
            "The `input` and `target` should have the same second dimension (i.e., sequence length), "
            f"got shapes {input.shape} and {target.shape} instead."
        )

    if ignore_index:
        _target = deepcopy(target)
        mask = _target.ne(ignore_index)
        _target = _target[mask]
    else:
        _target = target

    if input.size(2) <= torch.max(_target):
        raise ValueError(
            "Class labels in `target` tensor cannot be larger than vocab_size minus one, got "
            f"vocab size of {input.size(2)} and target label of {int(torch.max(_target))}."
        )
