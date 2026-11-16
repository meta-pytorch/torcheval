# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest

import torch
from torcheval.metrics.functional import word_error_rate


class TestWordErrorRate(unittest.TestCase):
    def test_word_error_rate_with_valid_input(self) -> None:
        torch.testing.assert_close(
            word_error_rate("hello meta", "hello metaverse"),
            torch.tensor(0.5, dtype=torch.float64),
        )
        torch.testing.assert_close(
            word_error_rate("hello meta", "hello meta"),
            torch.tensor(0.0, dtype=torch.float64),
        )
        torch.testing.assert_close(
            word_error_rate("this is the prediction", "this is the reference"),
            torch.tensor(0.25, dtype=torch.float64),
        )
        torch.testing.assert_close(
            word_error_rate(
                ["hello world", "welcome to the facebook"],
                ["hello metaverse", "welcome to meta"],
            ),
            torch.tensor(0.6, dtype=torch.float64),
        )
        torch.testing.assert_close(
            word_error_rate(
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
            torch.tensor(0.5, dtype=torch.float64),
        )

    def test_word_error_rate_with_invalid_input(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "input and target should have the same type"
        ):
            word_error_rate(["hello metaverse", "welcome to meta"], "hello world")

        with self.assertRaisesRegex(
            ValueError, "input and target lists should have the same length"
        ):
            word_error_rate(
                ["hello metaverse", "welcome to meta"],
                [
                    "welcome to meta",
                    "this is the prediction",
                    "there is an other sample",
                ],
            )
