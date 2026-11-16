# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import torch
from torcheval.metrics import Mean
from torcheval.utils.test_utils.metric_class_tester import (
    BATCH_SIZE,
    MetricClassTester,
    NUM_TOTAL_UPDATES,
)


class TestMean(MetricClassTester):
    def _test_mean_class_with_input(self, input_val_tensor: torch.Tensor) -> None:
        self.run_class_implementation_tests(
            metric=Mean(),
            state_names={"weighted_sum", "weights"},
            update_kwargs={"input": input_val_tensor},
            compute_result=torch.mean(input_val_tensor).to(torch.float64),
        )

    def test_mean_class_base(self) -> None:
        input_val_tensor = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE)
        self._test_mean_class_with_input(input_val_tensor)
        input_val_tensor = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE, 4)
        self._test_mean_class_with_input(input_val_tensor)
        input_val_tensor = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE, 3, 4)
        self._test_mean_class_with_input(input_val_tensor)

    def test_mean_class_update_input_dimension_different(self) -> None:
        self.run_class_implementation_tests(
            metric=Mean(),
            state_names={"weighted_sum", "weights"},
            update_kwargs={
                "input": [
                    torch.tensor(1.0),
                    torch.tensor([2.0, 3.0, 5.0]),
                    torch.tensor([-1.0, 2.0]),
                    torch.tensor([[1.0, 6.0], [2.0, -4.0]]),
                ]
            },
            compute_result=torch.tensor(1.700, dtype=torch.float64),
            num_total_updates=4,
            num_processes=2,
        )

    def test_mean_class_compute_without_update(self) -> None:
        metric = Mean()
        self.assertEqual(metric.compute(), torch.tensor(0.0, dtype=torch.float64))

        metric = Mean()
        metric.update(torch.tensor([0.0, 0.0]), weight=0)
        self.assertEqual(metric.compute(), torch.tensor(0.0, dtype=torch.float64))

    def test_mean_class_update_input_valid_weight(self) -> None:
        update_value = [
            torch.rand(BATCH_SIZE),
            torch.rand(BATCH_SIZE, 4),
            torch.rand(BATCH_SIZE, 3, 4),
            torch.rand(5),
            torch.rand(10),
        ]
        update_weight = [
            torch.rand(BATCH_SIZE),
            torch.rand(BATCH_SIZE, 4),
            torch.rand(BATCH_SIZE, 3, 4),
            0.8,
            4,
        ]

        def _compute_result(
            update_value: list[torch.Tensor],
            update_weight: list[float | int | torch.Tensor],
        ) -> torch.Tensor:
            weighted_sum = 0.0
            weights = 0.0
            for v, w in zip(update_value, update_weight):
                v = v.numpy().flatten()
                if isinstance(w, torch.Tensor):
                    w = w.numpy().flatten()
                else:
                    w = np.ones_like(v) * w
                average, sum_weights = np.average(v, weights=w, returned=True)
                weights += sum_weights
                weighted_sum += average * sum_weights
            weighted_mean = weighted_sum / weights
            return torch.tensor(weighted_mean, dtype=torch.float64)

        self.run_class_implementation_tests(
            metric=Mean(),
            state_names={"weighted_sum", "weights"},
            update_kwargs={
                "input": update_value,
                "weight": update_weight,
            },
            compute_result=_compute_result(update_value, update_weight),
            num_total_updates=5,
            num_processes=5,
        )

    def test_mean_class_update_input_invalid_weight(self) -> None:
        metric = Mean()
        with self.assertRaisesRegex(
            ValueError,
            r"Weight must be either a float value or a tensor that matches the input tensor size.",
        ):
            metric.update(torch.tensor([2.0, 3.0]), weight=torch.tensor([0.5]))
