# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import List, Optional, Tuple, Union

import torch

from torcheval.metrics.functional.text.helper import _get_errors_and_totals


def _wil_update(
    input: Union[str, List[str]],
    target: Union[str, List[str]],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Update the wil score with the current set of references and predictions.
    Args:
        input: Transcription(s) to score as a string or list of strings
        target: Reference(s) for each speech input as a string or list of strings
        device: The device to allocate tensors on
    Returns:
        Number of correct words
        Number of words overall references
        Number of words overall predictions
    """
    if isinstance(input, str):
        input = [input]
    if isinstance(target, str):
        target = [target]
    assert len(input) == len(
        target
    ), f"Arguments must contain the same number of strings, but got len(input)={len(input)} and len(target)={len(target)}"
    errors, max_total, target_total, input_total = _get_errors_and_totals(
        input, target, device
    )
    return errors - max_total, target_total, input_total


def _wil_compute(
    correct_total: torch.Tensor, target_total: torch.Tensor, preds_total: torch.Tensor
) -> torch.Tensor:
    """Compute the Word Information Lost.
    Args:
        correct_total: Number of correct words
        target_total: Number of words overall references
        preds_total: Number of words overall prediction
    Returns:
        Word Information Lost score
    """
    return 1 - ((correct_total / target_total) * (correct_total / preds_total))


@torch.inference_mode()
def word_information_lost(
    input: Union[str, List[str]],
    target: Union[str, List[str]],
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Word Information Lost rate is a metric of the performance of an automatic speech recognition system. This
    value indicates the percentage of characters that were incorrectly predicted. The lower the value, the better
    the performance of the ASR system with a Word Information Lost rate of 0 being a perfect score.

    Its class version is :obj:`torcheval.metrics.text.WordInformationLost`.

    Args:
        input: Transcription(s) to score as a string or list of strings
        target: Reference(s) for each speech input as a string or list of strings
        device: The device to allocate Tensors on
    Returns:
        Word Information Lost rate
    Examples:
        >>> from torcheval.metrics.functional import word_information_lost
        >>> input = ["this is the prediction", "there is an other sample"]
        >>> target = ["this is the reference", "there is another one"]
        >>> word_information_lost(input, target)
        tensor(0.6528)
    """
    correct_total, target_total, preds_total = _wil_update(input, target, device)
    return _wil_compute(correct_total, target_total, preds_total)
