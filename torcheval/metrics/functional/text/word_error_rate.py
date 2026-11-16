# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch


@torch.inference_mode()
def word_error_rate(
    input: str | list[str],
    target: str | list[str],
) -> torch.Tensor:
    """
    Compute the word error rate of the predicted word sequence(s) with the reference word sequence(s).
    Its class version is ``torcheval.metrics.WordErrorRate``.

    Args:
        input (str, List[str]): Predicted word sequence(s) to score as a string or list of strings.
        target (str, List[str]): Reference word sequence(s) as a string or list of strings.

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
    errors, total = _word_error_rate_update(input, target)
    return _word_error_rate_compute(errors, total)


def _word_error_rate_update(
    input: str | list[str],
    target: str | list[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Update the metric state with edit distance and the length of the reference sequence.

    Args:
        input (str, List[str]): Predicted word sequence(s) to score as a string or list of strings.
        target (str, List[str]): Reference word sequence(s) as a string or list of strings.
    """
    _word_error_rate_input_check(input, target)
    if isinstance(input, str):
        input = [input]
    if isinstance(target, str):
        target = [target]
    errors = torch.tensor(0, dtype=torch.float64)
    total = torch.tensor(0, dtype=torch.float64)
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


def _edit_distance(
    prediction_tokens: list[str],
    reference_tokens: list[str],
) -> int:
    """
    Dynamic programming algorithm to compute the edit distance between two word sequences.

    Args:
        prediction_tokens (List[str]): A tokenized predicted sentence
        reference_tokens (List[str]): A tokenized reference sentence
    """
    dp = [[0] * (len(reference_tokens) + 1) for _ in range(len(prediction_tokens) + 1)]
    for i in range(len(prediction_tokens) + 1):
        dp[i][0] = i
    for j in range(len(reference_tokens) + 1):
        dp[0][j] = j
    for i in range(1, len(prediction_tokens) + 1):
        for j in range(1, len(reference_tokens) + 1):
            if prediction_tokens[i - 1] == reference_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
    return dp[-1][-1]


def _word_error_rate_input_check(
    input: str | list[str],
    target: str | list[str],
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
