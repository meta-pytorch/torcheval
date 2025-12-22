# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import List, Optional, Tuple, Union

import torch

from torcheval.metrics.functional.text.helper import _edit_distance
from torcheval.utils.device import largest_float


@torch.inference_mode()
def word_error_rate(
    input: Union[str, List[str]],
    target: Union[str, List[str]],
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Compute the word error rate of the predicted word sequence(s) with the reference word sequence(s).
    Its class version is :obj:`torcheval.metrics.text.WordErrorRate`.

    Args:
        input (str, List[str]): Predicted word sequence(s) to score as a string or list of strings.
        target (str, List[str]): Reference word sequence(s) as a string or list of strings.
        device: The device to allocate tensors on

    Examples:

        >>> import torch
        >>> from torcheval.metrics.functional import word_error_rate
        >>> input = ["hello world", "welcome to the facebook"]
        >>> target = ["hello metaverse", "welcome to meta"]
        >>> word_error_rate(input, target)
        tensor(0.6)
        >>> input = ["this is the prediction", "there is an other sample"]
        >>> target = ["this is the reference", "there is another one"]
        >>> word_error_rate(input, target)
        tensor(0.5)
    """
    errors, total = _word_error_rate_update(input, target, device)
    return _word_error_rate_compute(errors, total)


def _word_error_rate_update(
    input: Union[str, List[str]],
    target: Union[str, List[str]],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Update the metric state with edit distance and the length of the reference sequence.

    Args:
        input (str, List[str]): Predicted word sequence(s) to score as a string or list of strings.
        target (str, List[str]): Reference word sequence(s) as a string or list of strings.
        device: The device to allocate Tensors on
    """
    _word_error_rate_input_check(input, target)
    if isinstance(input, str):
        input = [input]
    if isinstance(target, str):
        target = [target]
    dtype = largest_float(device)
    errors = torch.tensor(0, dtype=dtype, device=device)
    total = torch.tensor(0, dtype=dtype, device=device)
    for ipt, tgt in zip(input, target):
        ipt_tokens = ipt.split()
        tgt_tokens = tgt.split()
        errors += _edit_distance(ipt_tokens, tgt_tokens)
        total += len(tgt_tokens)
    return errors, total


def _word_error_rate_compute(
    errors: torch.Tensor,
    total: torch.Tensor,
) -> torch.Tensor:
    """
    Return the word error rate score

    Args:
        errors (Tensor): edit distance from the reference sequence to the predicted sequence, summed over all samples
        total (Tensor): length of reference sequence, summed over all samples.
    """
    return errors / total


def _word_error_rate_input_check(
    input: Union[str, List[str]],
    target: Union[str, List[str]],
) -> None:
    if (isinstance(input, str) and not isinstance(target, str)) or (
        isinstance(input, list) and not isinstance(target, list)
    ):
        raise ValueError(
            f"input and target should have the same type, got {type(input)} and {type(target)}."
        )
    if isinstance(input, list):
        if len(input) != len(target):
            raise ValueError(
                f"input and target lists should have the same length, got {len(input)} and {len(target)}",
            )
