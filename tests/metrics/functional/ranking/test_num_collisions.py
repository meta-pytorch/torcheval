# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest

import torch
from torcheval.metrics.functional import num_collisions


class TestNumCollisions(unittest.TestCase):
    def test_num_collisions_with_valid_input(self) -> None:
        input_test_1 = torch.tensor([3, 4, 2, 3])
        torch.testing.assert_close(
            num_collisions(input_test_1),
            torch.tensor([1, 0, 0, 1]),
        )

        input_test_2 = torch.tensor([3, 4, 1, 3, 1, 1, 5])
        torch.testing.assert_close(
            num_collisions(input_test_2),
            torch.tensor([1, 0, 2, 1, 2, 2, 0]),
        )

    def test_num_collisions_with_invalid_input(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "input should be a one-dimensional tensor"
        ):
            num_collisions(torch.randint(10, (3, 2)))

        with self.assertRaisesRegex(ValueError, "input should be an integer tensor"):
            num_collisions(torch.rand(3))
