# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import torch
from torcheval.metrics.functional.text.helper import _get_errors_and_totals


@torch.inference_mode()
def word_information_preserved(
    input: str | list[str],
    target: str | list[str],
) -> torch.Tensor:
    """
    Compute the word information preserved score of the predicted word sequence(s) against the reference word sequence(s).
    Its class version is ``torcheval.metrics.WordInformationPreserved``.

    Args:
        input (str, List[str]): Predicted word sequence(s) to score as a string or list of strings.
        target (str, List[str]): Reference word sequence(s) as a string or list of strings.

    Examples:

        >>> import torch
        >>> from torcheval.metrics.functional import word_information_preserved
        >>> input = ["hello world", "welcome to the facebook"]
        >>> target = ["hello metaverse", "welcome to meta"]
        >>> word_information_preserved(input, target)
        tensor(0.3)
        >>> input = ["this is the prediction", "there is an other sample"]
        >>> target = ["this is the reference", "there is another one"]
        >>> word_information_preserved(input, target)
        tensor(0.3472)
    """
    correct_total, target_total, input_total = _word_information_preserved_update(
        input, target
    )
    return _word_information_preserved_compute(correct_total, target_total, input_total)


def _word_information_preserved_update(
    input: str | list[str],
    target: str | list[str],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Update the word information preserved score with current set of predictions and references.

        Args:
            input (str, List[str]): Predicted word sequence(s) to score as a string or list of strings.
            target (str, List[str]): Reference word sequence(s) as a string or list of strings.
    """
    _word_information_preserved_input_check(input, target)
    errors, max_total, target_total, input_total = _get_errors_and_totals(input, target)

    return max_total - errors, target_total, input_total


def _word_information_preserved_compute(
    correct_total: torch.Tensor, target_total: torch.Tensor, input_total: torch.Tensor
) -> torch.Tensor:
    """
    Return the word information preserved score

    Args:
        correct_total (Tensor): number of words that are correctly predicted, summed over all samples
        target_total (Tensor): length of reference sequence, summed over all samples.
        input_total (Tensor): length of predicted sequence, summed over all samples.
    """
    return (correct_total / target_total) * (correct_total / input_total)


def _word_information_preserved_input_check(
    input: str | list[str],
    target: str | list[str],
) -> None:
    if type(input) != type(target):
        raise ValueError(
            f"input and target should have the same type, got {type(input)} and {type(target)}."
        )
    if type(input) == list:
        if len(input) != len(target):
            raise ValueError(
                f"input and target lists should have the same length, got {len(input)} and {len(target)}",
            )
