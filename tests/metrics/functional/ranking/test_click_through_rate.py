# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest

import torch
from torcheval.metrics.functional import click_through_rate


class TestClickThroughRate(unittest.TestCase):
    def test_click_through_rate_with_valid_input(self) -> None:
        input = torch.tensor([0, 1, 0, 1, 1, 0, 0, 1])
        weights = torch.tensor([1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0])
        torch.testing.assert_close(click_through_rate(input), torch.tensor(0.5))
        torch.testing.assert_close(
            click_through_rate(input, weights), torch.tensor(0.58333334)
        )

        input = torch.tensor([[0, 1, 0, 1], [1, 0, 0, 1]])
        weights = torch.tensor([[1.0, 2.0, 1.0, 2.0], [1.0, 2.0, 1.0, 1.0]])
        torch.testing.assert_close(
            click_through_rate(input, num_tasks=2), torch.tensor([0.5, 0.5])
        )
        torch.testing.assert_close(
            click_through_rate(input, weights, num_tasks=2),
            torch.tensor([0.66666667, 0.4]),
        )

    def test_click_through_rate_with_invalid_input(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "^`input` should be a one or two dimensional tensor",
        ):
            click_through_rate(torch.rand(3, 2, 2))
        with self.assertRaisesRegex(
            ValueError,
            "^tensor `weights` should have the same shape as tensor `input`",
        ):
            click_through_rate(torch.rand(4, 2), torch.rand(3))
        with self.assertRaisesRegex(
            ValueError,
            r"`num_tasks = 1`, `input` is expected to be one-dimensional tensor,",
        ):
            click_through_rate(
                torch.tensor([[1, 1], [0, 1]]),
            )
        with self.assertRaisesRegex(
            ValueError,
            r"`num_tasks = 2`, `input`'s shape is expected to be",
        ):
            click_through_rate(
                torch.tensor([1, 0, 0, 1]),
                num_tasks=2,
            )
