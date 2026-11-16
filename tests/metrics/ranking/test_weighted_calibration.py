# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch
from torcheval.metrics import WeightedCalibration
from torcheval.utils.test_utils.metric_class_tester import MetricClassTester


class TestWeightedCalibration(MetricClassTester):
    def test_weighted_calibration_with_valid_input(self) -> None:
        input = torch.tensor([[0.2, 0.3], [0.5, 0.1], [0.3, 0.5], [0.2, 0.4]])
        target = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
        weight = torch.tensor([[5.0, 1.0], [2.0, 3.0], [4.0, 7.0], [1.0, 1.0]])

        # without weight and num_tasks=1
        self.run_class_implementation_tests(
            metric=WeightedCalibration(),
            state_names={"weighted_input_sum", "weighted_target_sum"},
            update_kwargs={"input": input, "target": target},
            compute_result=torch.tensor([0.6250000596], dtype=torch.float64),
            num_total_updates=4,
            num_processes=2,
        )

        # with weight and num_tasks=1
        self.run_class_implementation_tests(
            metric=WeightedCalibration(),
            state_names={"weighted_input_sum", "weighted_target_sum"},
            update_kwargs={"input": input, "target": target, "weight": weight},
            compute_result=torch.tensor([0.9875000119], dtype=torch.float64),
            num_total_updates=4,
            num_processes=2,
        )

        # multi-task
        input_multi_tasks = torch.tensor(
            [
                [[0.2, 0.3], [0.5, 0.1]],
                [[0.3, 0.5], [0.2, 0.4]],
                [[0.9, 0.3], [0.3, 0.7]],
                [[0.1, 0.4], [0.7, 0.5]],
            ]
        )
        target_multi_tasks = torch.tensor(
            [
                [[0.0, 0.0], [1.0, 0.0]],
                [[1.0, 0.0], [1.0, 1.0]],
                [[0.0, 0.0], [1.0, 0.0]],
                [[1.0, 0.0], [1.0, 1.0]],
            ]
        )
        weight_multi_tasks = torch.tensor(
            [
                [[5.0, 1.0], [2.0, 3.0]],
                [[4.0, 7.0], [1.0, 1.0]],
                [[6.0, 2.0], [1.0, 1.0]],
                [[2.0, 1.0], [6.0, 2.0]],
            ]
        )

        # without weight and num_tasks=2
        self.run_class_implementation_tests(
            metric=WeightedCalibration(num_tasks=2),
            state_names={"weighted_input_sum", "weighted_target_sum"},
            update_kwargs={"input": input_multi_tasks, "target": target_multi_tasks},
            compute_result=torch.tensor(
                [1.5000000298023224, 0.5666666825612386], dtype=torch.float64
            ),
            num_total_updates=4,
            num_processes=2,
        )

        # with weight and num_tasks=2
        self.run_class_implementation_tests(
            metric=WeightedCalibration(num_tasks=2),
            state_names={"weighted_input_sum", "weighted_target_sum"},
            update_kwargs={
                "input": input_multi_tasks,
                "target": target_multi_tasks,
                "weight": weight_multi_tasks,
            },
            compute_result=torch.tensor(
                [2.0999998847643533, 0.6230769065710214], dtype=torch.float64
            ),
            num_total_updates=4,
            num_processes=2,
        )

    def test_weighted_calibration_with_invalid_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "`num_tasks` value should be greater"):
            metric = WeightedCalibration(num_tasks=-1)

        metric = WeightedCalibration()
        with self.assertRaisesRegex(
            ValueError,
            r"Weight must be either a float value or a tensor that matches the input tensor size.",
        ):
            metric.update(
                torch.tensor([0.8, 0.4, 0.8, 0.7]),
                torch.tensor([1, 1, 0, 1]),
                torch.tensor([1, 1.5]),
            )

        with self.assertRaisesRegex(
            ValueError,
            r"is different from `target` shape",
        ):
            metric.update(
                torch.tensor([0.8, 0.4, 0.8, 0.7]),
                torch.tensor([[1, 1, 0], [0, 1, 1]]),
            )

        with self.assertRaisesRegex(
            ValueError,
            r"`num_tasks = 1`, `input` is expected to be one-dimensional tensor,",
        ):
            metric.update(
                torch.tensor([[0.8, 0.4], [0.8, 0.7]]),
                torch.tensor([[1, 1], [0, 1]]),
            )

        metric = WeightedCalibration(num_tasks=2)
        with self.assertRaisesRegex(
            ValueError,
            r"`num_tasks = 2`, `input`'s shape is expected to be",
        ):
            metric.update(
                torch.tensor([0.8, 0.4, 0.8, 0.7]),
                torch.tensor([1, 0, 0, 1]),
            )
