# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch
from torcheval.metrics import WindowedClickThroughRate
from torcheval.utils.test_utils.metric_class_tester import MetricClassTester


class TestClickThroughRate(MetricClassTester):
    def test_ctr_with_valid_input(self) -> None:
        input = torch.tensor([[1, 0, 0, 1], [0, 0, 0, 0], [1, 1, 1, 1], [0, 1, 1, 1]])

        self.run_class_implementation_tests(
            metric=WindowedClickThroughRate(
                num_tasks=1, max_num_updates=2, enable_lifetime=True
            ),
            state_names={
                "max_num_updates",
                "total_updates",
                "click_total",
                "weight_total",
                "windowed_click_total",
                "windowed_weight_total",
            },
            update_kwargs={"input": input},
            compute_result=(
                torch.tensor([0.5625], dtype=torch.float64),
                torch.tensor([0.875], dtype=torch.float64),
            ),
            merge_and_compute_result=(
                torch.tensor([0.5625], dtype=torch.float64),
                torch.tensor([0.5625], dtype=torch.float64),
            ),
            num_total_updates=4,
            num_processes=2,
        )

        self.run_class_implementation_tests(
            metric=WindowedClickThroughRate(
                num_tasks=1, max_num_updates=2, enable_lifetime=False
            ),
            state_names={
                "max_num_updates",
                "total_updates",
                "windowed_click_total",
                "windowed_weight_total",
            },
            update_kwargs={"input": input},
            compute_result=torch.tensor([0.875], dtype=torch.float64),
            merge_and_compute_result=torch.tensor([0.5625], dtype=torch.float64),
            num_total_updates=4,
            num_processes=2,
        )

        input = torch.tensor(
            [
                [[1, 0, 0, 1], [1, 1, 1, 1]],
                [[0, 0, 0, 0], [1, 1, 1, 1]],
                [[0, 1, 0, 1], [0, 1, 0, 1]],
                [[1, 1, 1, 1], [0, 1, 1, 1]],
            ]
        )
        weights = torch.tensor(
            [
                [[1, 2, 3, 4], [0, 0, 0, 0]],
                [[1, 2, 1, 2], [1, 2, 1, 2]],
                [[1, 1, 1, 1], [1, 1, 3, 1]],
                [[1, 1, 1, 1], [1, 1, 1, 1]],
            ]
        )

        self.run_class_implementation_tests(
            metric=WindowedClickThroughRate(
                num_tasks=2, max_num_updates=2, enable_lifetime=True
            ),
            state_names={
                "max_num_updates",
                "total_updates",
                "click_total",
                "weight_total",
                "windowed_click_total",
                "windowed_weight_total",
            },
            update_kwargs={"input": input, "weights": weights},
            compute_result=(
                torch.tensor([0.4583333, 0.6875], dtype=torch.float64),
                torch.tensor([0.75, 0.5], dtype=torch.float64),
            ),
            merge_and_compute_result=(
                torch.tensor([0.4583333, 0.6875], dtype=torch.float64),
                torch.tensor([0.4583333, 0.6875], dtype=torch.float64),
            ),
            num_total_updates=4,
            num_processes=2,
        )

        weights = [4.0, 1, torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]]), 0.0]

        self.run_class_implementation_tests(
            metric=WindowedClickThroughRate(
                num_tasks=2, max_num_updates=2, enable_lifetime=True
            ),
            state_names={
                "max_num_updates",
                "total_updates",
                "click_total",
                "weight_total",
                "windowed_click_total",
                "windowed_weight_total",
            },
            update_kwargs={"input": input, "weights": weights},
            compute_result=(
                torch.tensor([0.46666667, 0.86666667], dtype=torch.float64),
                torch.tensor([0.6, 0.6], dtype=torch.float64),
            ),
            merge_and_compute_result=(
                torch.tensor([0.46666667, 0.86666667], dtype=torch.float64),
                torch.tensor([0.46666667, 0.86666667], dtype=torch.float64),
            ),
            num_total_updates=4,
            num_processes=2,
        )

    def test_ctr_with_invalid_input(self) -> None:
        metric = WindowedClickThroughRate()
        with self.assertRaisesRegex(
            ValueError,
            "^`input` should be a one or two dimensional tensor",
        ):
            metric.update(torch.rand(3, 2, 2))

        metric = WindowedClickThroughRate()
        with self.assertRaisesRegex(
            ValueError,
            "^tensor `weights` should have the same shape as tensor `input`",
        ):
            metric.update(torch.rand(4, 2), torch.rand(3))
        with self.assertRaisesRegex(
            ValueError,
            r"`num_tasks = 1`, `input` is expected to be one-dimensional tensor,",
        ):
            metric.update(
                torch.tensor([[1, 1], [0, 1]]),
            )

        metric = WindowedClickThroughRate(num_tasks=2)
        with self.assertRaisesRegex(
            ValueError,
            r"`num_tasks = 2`, `input`'s shape is expected to be",
        ):
            metric.update(
                torch.tensor([1, 0, 0, 1]),
            )

        with self.assertRaisesRegex(
            ValueError,
            r"`num_tasks` value should be greater than and equal to 1,",
        ):
            metric = WindowedClickThroughRate(num_tasks=0)

        with self.assertRaisesRegex(
            ValueError,
            r"`max_num_updates` value should be greater than and equal to 1, ",
        ):
            metric = WindowedClickThroughRate(max_num_updates=0)
