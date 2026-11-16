# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch
from torcheval.metrics import Sum
from torcheval.utils.test_utils.metric_class_tester import (
    BATCH_SIZE,
    MetricClassTester,
    NUM_TOTAL_UPDATES,
)


class TestSum(MetricClassTester):
    def _test_sum_class_with_input(self, input_val_tensor: torch.Tensor) -> None:
        self.run_class_implementation_tests(
            metric=Sum(),
            state_names={"weighted_sum"},
            update_kwargs={"input": input_val_tensor},
            compute_result=torch.sum(input_val_tensor).to(torch.float64),
        )

    def test_sum_class_base(self) -> None:
        input_val_tensor = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE)
        self._test_sum_class_with_input(input_val_tensor)
        input_val_tensor = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE, 4)
        self._test_sum_class_with_input(input_val_tensor)
        input_val_tensor = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE, 3, 4)
        self._test_sum_class_with_input(input_val_tensor)

    def test_sum_class_update_input_dimension_different(self) -> None:
        self.run_class_implementation_tests(
            metric=Sum(),
            state_names={"weighted_sum"},
            update_kwargs={
                "input": [
                    torch.tensor(1.0),
                    torch.tensor([2.0, 3.0, 5.0]),
                    torch.tensor([-1.0, 2.0]),
                    torch.tensor([[1.0, 6.0], [2.0, -4.0]]),
                ]
            },
            compute_result=torch.tensor(17.0, dtype=torch.float64),
            num_total_updates=4,
            num_processes=2,
        )

    def test_sum_class_update_input_valid_weight(self) -> None:
        update_inputs = [
            torch.rand(BATCH_SIZE),
            torch.rand(BATCH_SIZE, 4),
            torch.rand(BATCH_SIZE, 3, 4),
            torch.rand(5),
            torch.rand(10),
        ]
        update_weights = [
            torch.rand(BATCH_SIZE),
            torch.rand(BATCH_SIZE, 4),
            torch.rand(BATCH_SIZE, 3, 4),
            0.8,
            2,
        ]

        def _compute_result(
            update_inputs: list[torch.Tensor],
            update_weights: list[float | torch.Tensor],
        ) -> torch.Tensor:
            weighted_sum = torch.tensor(0.0, dtype=torch.float64)
            for v, w in zip(update_inputs, update_weights):
                if isinstance(w, torch.Tensor):
                    w = w.numpy().flatten()
                weighted_sum += v.numpy().flatten().dot(w).sum()
            return weighted_sum

        self.run_class_implementation_tests(
            metric=Sum(),
            state_names={"weighted_sum"},
            update_kwargs={
                "input": update_inputs,
                "weight": update_weights,
            },
            compute_result=_compute_result(update_inputs, update_weights),
            num_total_updates=5,
            num_processes=5,
        )

    def test_sum_class_update_input_invalid_weight(self) -> None:
        metric = Sum()
        with self.assertRaisesRegex(
            ValueError,
            r"Weight must be either a float value or an int value or a tensor that matches the input tensor size.",
        ):
            metric.update(torch.tensor([2.0, 3.0]), weight=torch.tensor([0.5]))

    def test_sum_class_compute_without_update(self) -> None:
        metric = Sum()
        self.assertEqual(metric.compute(), torch.tensor(0.0, dtype=torch.float64))
