# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest

import torch
from torcheval.metrics.functional import reciprocal_rank


class TestReciprocalRank(unittest.TestCase):
    def test_reciprocal_rank_with_valid_input(self) -> None:
        input = torch.tensor(
            [
                [0.4826, 0.9517, 0.8967, 0.8995, 0.1584, 0.9445, 0.9700],
                [0.4938, 0.7517, 0.8039, 0.7167, 0.9488, 0.9607, 0.7091],
                [0.5127, 0.4732, 0.5461, 0.5617, 0.9198, 0.0847, 0.2337],
                [0.4175, 0.9452, 0.9852, 0.2131, 0.5016, 0.7305, 0.0516],
            ]
        )
        target = torch.tensor([3, 5, 2, 1])

        torch.testing.assert_close(
            reciprocal_rank(input, target, k=None),
            torch.tensor([0.2500, 1.0000, 1.0000 / 3, 0.5000]),
        )
        torch.testing.assert_close(
            reciprocal_rank(input, target, k=1),
            torch.tensor([0.0000, 1.0000, 0.0000, 0.0000]),
        )
        torch.testing.assert_close(
            reciprocal_rank(input, target, k=3),
            torch.tensor([0.0000, 1.0000, 1.0000 / 3, 0.5000]),
        )
        torch.testing.assert_close(
            reciprocal_rank(input, target, k=5),
            torch.tensor([0.2500, 1.0000, 1.0000 / 3, 0.5000]),
        )
        torch.testing.assert_close(
            reciprocal_rank(input, target, k=20),
            torch.tensor([0.2500, 1.0000, 1.0000 / 3, 0.5000]),
        )
        torch.testing.assert_close(
            reciprocal_rank(input, target, k=100),
            torch.tensor([0.2500, 1.0000, 1.0000 / 3, 0.5000]),
        )

    def test_reciprocal_rank_with_invalid_input(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "target should be a one-dimensional tensor"
        ):
            reciprocal_rank(torch.rand(3, 2), torch.rand(3, 2))

        with self.assertRaisesRegex(
            ValueError, "input should be a two-dimensional tensor"
        ):
            reciprocal_rank(torch.rand(3, 2, 2), torch.rand(3))
        with self.assertRaisesRegex(
            ValueError, "`input` and `target` should have the same minibatch dimension"
        ):
            reciprocal_rank(torch.rand(4, 2), torch.rand(3))
