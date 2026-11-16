# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest

import torch
from torcheval.metrics.functional import frequency_at_k


class TestFrequency(unittest.TestCase):
    def test_frequency_with_valid_input(self) -> None:
        input = torch.tensor(
            [0.4826, 0.9517, 0.8967, 0.8995, 0.1584, 0.9445, 0.9700],
        )

        torch.testing.assert_close(
            frequency_at_k(input, k=0.5),
            torch.tensor([1.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000]),
        )
        torch.testing.assert_close(
            frequency_at_k(input, k=0.9),
            torch.tensor([1.0000, 0.0000, 1.0000, 1.0000, 1.0000, 0.0000, 0.0000]),
        )
        torch.testing.assert_close(
            frequency_at_k(input, k=0.95),
            torch.tensor([1.0000, 0.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.0000]),
        )
        torch.testing.assert_close(
            frequency_at_k(input, k=1.0),
            torch.tensor([1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000]),
        )

    def test_frequency_with_invalid_input(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "input should be a one-dimensional tensor"
        ):
            frequency_at_k(torch.rand(3, 2, 2), k=1)
        with self.assertRaisesRegex(ValueError, "k should not be negative"):
            frequency_at_k(torch.rand(3), k=-1)
