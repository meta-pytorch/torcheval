# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import torch


def _count_matches(
    prediction_tokens: list[str],
    reference_tokens: list[str],
) -> int:
    """
    Count matching words in a minimum-edit alignment of two word sequences.
    Among alignments with the same edit distance, prefer the most matches.

    Args:
        prediction_tokens (List[str]): A tokenized predicted sentence
        reference_tokens (List[str]): A tokenized reference sentence
    """
    # Store (edit distance, negative match count) to minimize edits first,
    # then maximize matches. Counting a common subsequence alone could choose
    # an alignment with more edits than the Levenshtein distance.
    previous_row = [(j, 0) for j in range(len(reference_tokens) + 1)]
    for i, prediction_token in enumerate(prediction_tokens, 1):
        current_row = [(i, 0)]
        for j, reference_token in enumerate(reference_tokens, 1):
            match = prediction_token == reference_token
            current_row.append(
                min(
                    (previous_row[j][0] + 1, previous_row[j][1]),
                    (current_row[j - 1][0] + 1, current_row[j - 1][1]),
                    (
                        previous_row[j - 1][0] + (not match),
                        previous_row[j - 1][1] - match,
                    ),
                )
            )
        previous_row = current_row
    return -previous_row[-1][1]


def _get_correct_and_totals(
    input: str | list[str],
    target: str | list[str],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate the matches and lengths of predicted and reference word sequences.

        Args:
            input (str, List[str]): Predicted word sequence(s) to score as a string or list of strings.
            target (str, List[str]): Reference word sequence(s) as a string or list of strings.
    """
    if isinstance(input, str):
        input = [input]
    if isinstance(target, str):
        target = [target]
    correct_total = torch.tensor(0.0, dtype=torch.float64)
    target_total = torch.tensor(0.0, dtype=torch.float64)
    input_total = torch.tensor(0.0, dtype=torch.float64)
    for ipt, tgt in zip(input, target):
        input_tokens = ipt.split()
        target_tokens = tgt.split()
        correct_total += _count_matches(input_tokens, target_tokens)
        target_total += len(target_tokens)
        input_total += len(input_tokens)

    return correct_total, target_total, input_total
