# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import numpy as np
import torch
from torcheval.metrics.functional.aggregation import mean
from torcheval.utils.test_utils.metric_class_tester import BATCH_SIZE, NUM_TOTAL_UPDATES


class TestMean(unittest.TestCase):
    def _test_mean_with_input(
        self,
        val: torch.Tensor,
        weight: float | torch.Tensor = 1.0,
    ) -> None:
        torch.testing.assert_close(
            mean(val),
            torch.mean(val),
            equal_nan=True,
            atol=1e-8,
            rtol=1e-5,
        )

    def test_mean_base(self) -> None:
        input_val_tensor = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE)
        self._test_mean_with_input(input_val_tensor)
        input_val_tensor = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE, 4)
        self._test_mean_with_input(input_val_tensor)
        input_val_tensor = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE, 3, 4)
        self._test_mean_with_input(input_val_tensor)

    def test_mean_input_valid_weight(self) -> None:
        def _compute_result(
            val: torch.Tensor, weights: float | torch.Tensor
        ) -> torch.Tensor:
            val = val.numpy().flatten()
            if isinstance(weights, torch.Tensor):
                weights = weights.numpy().flatten()
            else:
                weights = weights * np.ones_like(val)
            weighted_mean = np.average(val, weights=weights)
            return torch.tensor(weighted_mean, dtype=torch.float32)

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
            1,
        ]

        for input, weight in zip(inputs, weights):
            print(input)
            print(weight)
            torch.testing.assert_close(
                mean(input, weight),
                _compute_result(input, weight),
                equal_nan=True,
                atol=1e-8,
                rtol=1e-5,
            )

    def test_mean_input_invalid_weight(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            r"Weight must be either a float value or a tensor that matches the input tensor size.",
        ):
            mean(torch.tensor([2.0, 3.0]), torch.tensor([0.5]))
