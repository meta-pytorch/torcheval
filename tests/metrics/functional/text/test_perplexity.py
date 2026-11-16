# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest

import torch
from torcheval.metrics.functional import perplexity


class Perplexity(unittest.TestCase):
    def test_perplexity(self) -> None:
        input = torch.tensor([[[0.3659, 0.7025, 0.3104]], [[0.0097, 0.6577, 0.1947]]])
        target = torch.tensor([[2], [1]])
        torch.testing.assert_close(
            perplexity(input, target),
            torch.tensor(2.759287357330, dtype=torch.float64),
        )

        input = torch.tensor(
            [
                [
                    [0.3633, 0.7777, 0.3111, 0.1134],
                    [0.5543, 0.4387, 0.8176, 0.4565],
                    [0.1324, 0.1465, 0.2822, 0.6654],
                ],
                [
                    [0.0956, 0.6523, 0.8914, 0.3653],
                    [0.3289, 0.7254, 0.8121, 0.4196],
                    [0.3233, 0.7775, 0.8731, 0.4001],
                ],
            ]
        )
        target = torch.tensor([[2, 1, 3], [1, 0, 1]])
        torch.testing.assert_close(
            perplexity(input, target),
            torch.tensor(3.944433212280, dtype=torch.float64),
        )

    def test_perplexity_with_ignore_index(self) -> None:
        input = torch.tensor([[[0.3659, 0.7025, 0.3104]], [[0.0097, 0.6577, 0.1947]]])
        target = torch.tensor([[2], [100]])
        torch.testing.assert_close(
            perplexity(input, target, ignore_index=100),
            torch.tensor(3.537154912949, dtype=torch.float64),
        )

        input = torch.tensor(
            [
                [
                    [0.3633, 0.7777, 0.3111, 0.1134],
                    [0.5543, 0.4387, 0.8176, 0.4565],
                    [0.1324, 0.1465, 0.2822, 0.6654],
                ],
                [
                    [0.0956, 0.6523, 0.8914, 0.3653],
                    [0.3289, 0.7254, 0.8121, 0.4196],
                    [0.3233, 0.7775, 0.8731, 0.4001],
                ],
            ]
        )
        target = torch.tensor([[2, 1, 3], [1, 0, 1]])
        torch.testing.assert_close(
            perplexity(input, target, ignore_index=1),
            torch.tensor(4.053068637848, dtype=torch.float64),
        )

    def test_perplexity_with_invalid_input(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "target should be a two-dimensional tensor"
        ):
            perplexity(torch.rand(4, 2, 3), torch.randint(3, (4, 2, 2)))

        with self.assertRaisesRegex(
            ValueError, "input should be a three-dimensional tensor"
        ):
            perplexity(torch.rand(3, 2), torch.randint(3, (4, 2)))

        with self.assertRaisesRegex(
            ValueError, "The `input` and `target` should have the same second dimension"
        ):
            perplexity(torch.rand(3, 2, 2), torch.randint(2, (3, 9)))

        with self.assertRaisesRegex(
            ValueError, "The `input` and `target` should have the same first dimension"
        ):
            perplexity(torch.rand(3, 2, 3), torch.randint(3, (2, 2)))

        with self.assertRaisesRegex(
            ValueError,
            "Class labels in `target` tensor cannot be larger than vocab_size minus one",
        ):
            perplexity(
                torch.rand(3, 2, 3), target=torch.tensor([[4, 2], [1, 0], [0, 0]])
            )

        with self.assertRaisesRegex(
            ValueError,
            "Class labels in `target` tensor cannot be larger than vocab_size minus one",
        ):
            perplexity(
                torch.rand(3, 2, 3),
                torch.tensor([[4, 2], [1, 0], [0, 0]]),
                ignore_index=1,
            )

        with self.assertRaisesRegex(
            ValueError,
            "Class labels in `target` tensor cannot be larger than vocab_size minus one",
        ):
            perplexity(
                torch.rand(3, 2, 3),
                torch.tensor([[4, 2], [1, 0], [100, 0]]),
                ignore_index=100,
            )
