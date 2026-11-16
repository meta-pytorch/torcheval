# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest

import torch
from torcheval.metrics.functional import sum
from torcheval.utils.test_utils.metric_class_tester import BATCH_SIZE, NUM_TOTAL_UPDATES


class TestSum(unittest.TestCase):
    def _test_sum_with_input(
        self,
        val: torch.Tensor,
        weight: float | torch.Tensor = 1.0,
    ) -> None:
        torch.testing.assert_close(
            sum(val),
            torch.sum(val),
            equal_nan=True,
            atol=1e-8,
            rtol=1e-5,
        )

    def test_sum_base(self) -> None:
        input_val_tensor = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE)
        self._test_sum_with_input(input_val_tensor)
        input_val_tensor = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE, 4)
        self._test_sum_with_input(input_val_tensor)
        input_val_tensor = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE, 3, 4)
        self._test_sum_with_input(input_val_tensor)

    def test_sum_input_valid_weight(self) -> None:
        def _compute_result(
            val: torch.Tensor, weight: float | torch.Tensor
        ) -> torch.Tensor:
            weighted_sum = torch.tensor(0.0)
            if isinstance(weight, torch.Tensor):
                weight = weight.numpy().flatten()
            weighted_sum += val.numpy().flatten().dot(weight).sum()

            return weighted_sum

        inputs = [
            torch.rand(1),
            torch.rand(BATCH_SIZE, 4),
            torch.rand(BATCH_SIZE, 3, 4),
            torch.rand(5),
            torch.rand(10),
        ]
        weights = [
            torch.rand(1),
            torch.rand(BATCH_SIZE, 4),
            torch.rand(BATCH_SIZE, 3, 4),
            0.8,
            2,
        ]

        for input, weight in zip(inputs, weights):
            torch.testing.assert_close(
                sum(input, weight),
                _compute_result(input, weight),
                equal_nan=True,
                atol=1e-8,
                rtol=1e-5,
            )

    def test_sum_input_invalid_weight(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            r"Weight must be either a float value or an int value or a tensor that matches the input tensor size.",
        ):
            sum(torch.tensor([2.0, 3.0]), torch.tensor([0.5]))
