# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import torch
from torcheval.metrics.functional import word_information_preserved


class TestWordInformationPreserved(unittest.TestCase):
    def test_word_information_preserved_alignment(self) -> None:
        cases = [
            ("sat the cat", "the cat sat", 4 / 9),
            ("the cat sat", "sat the cat", 4 / 9),
            ("sat the cat", "the cat sat extra", 1 / 3),
            ("the cat sat extra", "sat the cat", 1 / 3),
            ("the cat sat", "the cat", 2 / 3),
            ("the cat", "the cat sat", 2 / 3),
            ("the dog sat", "the cat sat", 4 / 9),
            ("a b", "c d", 0.0),
            ("a a b", "b c c", 0.0),
            ("a b", "b a", 1 / 4),
            ("a a b", "a b a", 4 / 9),
            ("  the\tcat\nsat  ", "the cat sat", 1.0),
        ]
        for input, target, expected in cases:
            with self.subTest(input=input, target=target):
                torch.testing.assert_close(
                    word_information_preserved(input, target),
                    torch.tensor(expected, dtype=torch.float64),
                )

    def test_word_information_preserved_aggregates_counts(self) -> None:
        # Sum the three matching words before applying the WIP formula.
        torch.testing.assert_close(
            word_information_preserved(
                ["sat the cat", "hello"], ["the cat sat", "hello world"]
            ),
            torch.tensor(9 / 20, dtype=torch.float64),
        )

    def test_word_information_preserved_with_valid_input(self) -> None:
        torch.testing.assert_close(
            word_information_preserved("hello meta", "hi metaverse"),
            torch.tensor(0.0, dtype=torch.float64),
        )
        torch.testing.assert_close(
            word_information_preserved("hello meta", "hello meta"),
            torch.tensor(1.0, dtype=torch.float64),
        )
        torch.testing.assert_close(
            word_information_preserved(
                "this is the prediction", "this is the reference"
            ),
            torch.tensor(0.5625, dtype=torch.float64),
        )
        torch.testing.assert_close(
            word_information_preserved(
                ["hello world", "welcome to the facebook"],
                ["hello metaverse", "welcome to meta"],
            ),
            torch.tensor(0.3, dtype=torch.float64),
        )
        torch.testing.assert_close(
            word_information_preserved(
                [
                    "hello metaverse",
                    "come to the facebook",
                    "this is reference",
                    "there is the other one",
                ],
                [
                    "hello world",
                    "welcome to meta",
                    "this is reference",
                    "there is another one",
                ],
            ),
            torch.tensor(0.38095238, dtype=torch.float64),
        )

    def test_word_information_preserved_with_invalid_input(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "input and target should have the same type"
        ):
            word_information_preserved(
                ["hello metaverse", "welcome to meta"], "hello world"
            )

        with self.assertRaisesRegex(
            ValueError, "input and target lists should have the same length"
        ):
            word_information_preserved(
                ["hello metaverse", "welcome to meta"],
                [
                    "welcome to meta",
                    "this is the prediction",
                    "there is an other sample",
                ],
            )
