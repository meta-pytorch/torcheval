# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from typing import List, Tuple, Union

import torch

from torcheval.utils.device import largest_float


def _edit_distance(
    prediction_tokens: List[str],
    reference_tokens: List[str],
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


def _get_errors_and_totals(
    input: Union[str, List[str]],
    target: Union[str, List[str]],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate the edit distance, max length and lengths of predicted and reference word sequences.

        Args:
            input (str, List[str]): Predicted word sequence(s) to score as a string or list of strings.
            target (str, List[str]): Reference word sequence(s) as a string or list of strings.
            device: The device to allocate tensors on
    """
    if isinstance(input, str):
        input = [input]
    if isinstance(target, str):
        target = [target]
    dtype = largest_float(device)
    max_total = torch.tensor(0.0, dtype=dtype, device=device)
    errors = torch.tensor(0.0, dtype=dtype, device=device)
    target_total = torch.tensor(0.0, dtype=dtype, device=device)
    input_total = torch.tensor(0.0, dtype=dtype, device=device)
    for ipt, tgt in zip(input, target):
        input_tokens = ipt.split()
        target_tokens = tgt.split()
        errors += _edit_distance(input_tokens, target_tokens)
        target_total += len(target_tokens)
        input_total += len(input_tokens)
        max_total += max(len(target_tokens), len(input_tokens))

    return errors, max_total, target_total, input_total
