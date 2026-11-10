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

        torch.testing.assert_close(
            word_information_lost("sat the cat", "the cat sat"),
            torch.tensor(5 / 9, dtype=torch.float64),
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
