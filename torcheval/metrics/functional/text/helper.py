# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import torch


def _count_hits(
    prediction_tokens: list[str],
    reference_tokens: list[str],
) -> int:
    """
    Count the words that match in an optimal (minimum edit distance) alignment of two word sequences.

    Args:
        prediction_tokens (List[str]): A tokenized predicted sentence
        reference_tokens (List[str]): A tokenized reference sentence
    """
    n, m = len(prediction_tokens), len(reference_tokens)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if prediction_tokens[i - 1] == reference_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
    hits = 0
    i, j = n, m
    while i > 0 and j > 0:
        if (
            prediction_tokens[i - 1] == reference_tokens[j - 1]
            and dp[i][j] == dp[i - 1][j - 1]
        ):
            hits += 1
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i - 1][j - 1] + 1:
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
    return hits


def _get_hits_and_totals(
    input: str | list[str],
    target: str | list[str],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate the number of matching words and lengths of predicted and reference word sequences.

        Args:
            input (str, List[str]): Predicted word sequence(s) to score as a string or list of strings.
            target (str, List[str]): Reference word sequence(s) as a string or list of strings.
    """
    if isinstance(input, str):
        input = [input]
    if isinstance(target, str):
        target = [target]
    hits = torch.tensor(0.0, dtype=torch.float64)
    target_total = torch.tensor(0.0, dtype=torch.float64)
    input_total = torch.tensor(0.0, dtype=torch.float64)
    for ipt, tgt in zip(input, target):
        input_tokens = ipt.split()
        target_tokens = tgt.split()
        hits += _count_hits(input_tokens, target_tokens)
        target_total += len(target_tokens)
        input_total += len(input_tokens)

    return hits, target_total, input_total
