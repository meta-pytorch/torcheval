# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import torch
from torcheval.metrics.functional import word_information_lost


class TestWordInformationLost(unittest.TestCase):
    def test_word_information_lost_alignment(self) -> None:
        cases = [
            ("sat the cat", "the cat sat", 5 / 9),
            ("the cat sat", "sat the cat", 5 / 9),
            ("sat the cat", "the cat sat extra", 2 / 3),
            ("the cat sat extra", "sat the cat", 2 / 3),
            ("the cat sat", "the cat", 1 / 3),
            ("the cat", "the cat sat", 1 / 3),
            ("the dog sat", "the cat sat", 5 / 9),
            ("a b", "c d", 1.0),
            ("a a b", "b c c", 1.0),
            ("a b", "b a", 3 / 4),
            ("a a b", "a b a", 5 / 9),
            ("  the\tcat\nsat  ", "the cat sat", 0.0),
        ]
        for input, target, expected in cases:
            with self.subTest(input=input, target=target):
                torch.testing.assert_close(
                    word_information_lost(input, target),
                    torch.tensor(expected, dtype=torch.float64),
                )

    def test_word_information_lost_aggregates_counts(self) -> None:
        # Sum the three matching words before applying the WIL formula.
        torch.testing.assert_close(
            word_information_lost(
                ["sat the cat", "hello"], ["the cat sat", "hello world"]
            ),
            torch.tensor(11 / 20, dtype=torch.float64),
        )

    def test_word_information_lost(self) -> None:
        input = ["hello world", "welcome to the facebook"]
        target = ["hello metaverse", "welcome to meta"]
        torch.testing.assert_close(
            word_information_lost(input, target),
            torch.tensor(0.7, dtype=torch.float64),
        )

        input = ["this is the prediction", "there is an other sample"]
        target = ["this is the reference", "there is another one"]
        torch.testing.assert_close(
            word_information_lost(input, target),
            torch.tensor(0.6527777, dtype=torch.float64),
        )

    def test_word_information_lost_with_invalid_input(self) -> None:
        with self.assertRaisesRegex(
            AssertionError,
            "Arguments must contain the same number of strings.",
        ):
            word_information_lost(
                ["hello metaverse", "welcome to meta"],
                [
                    "welcome to meta",
                    "this is the prediction",
                    "there is an other sample",
                ],
            )
