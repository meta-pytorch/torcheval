# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch
from sklearn.metrics import mean_squared_error
from torcheval.metrics import WindowedMeanSquaredError
from torcheval.utils.test_utils.metric_class_tester import (
    BATCH_SIZE,
    MetricClassTester,
    NUM_TOTAL_UPDATES,
)


class TestMeanSquaredError(MetricClassTester):
    def _test_mean_squared_error_class_with_input(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        num_tasks: int,
        max_num_updates: int,
        enable_lifetime: bool,
        sample_weight: torch.Tensor | None = None,
        multioutput: str = "uniform_average",
    ) -> None:
        target_np = target.reshape(
            -1, int(torch.numel(target) / (NUM_TOTAL_UPDATES * BATCH_SIZE))
        ).squeeze()
        input_np = input.reshape(
            -1, int(torch.numel(target) / (NUM_TOTAL_UPDATES * BATCH_SIZE))
        ).squeeze()

        target_window = target[(-1) * max_num_updates :,]
        input_window = input[(-1) * max_num_updates :,]

        target_window_np = target_window.reshape(
            -1, int(torch.numel(target_window) / (max_num_updates * BATCH_SIZE))
        ).squeeze()
        input_window_np = input_window.reshape(
            -1, int(torch.numel(input_window) / (max_num_updates * BATCH_SIZE))
        ).squeeze()

        sample_weight_np = None
        sample_weight_window_np = None
        if sample_weight is not None:
            sample_weight_np = sample_weight.reshape(
                -1, int(torch.numel(sample_weight) / (NUM_TOTAL_UPDATES * BATCH_SIZE))
            ).squeeze()
            sample_weight_window = sample_weight[(-1) * max_num_updates :,]
            sample_weight_window_np = sample_weight_window.reshape(
                -1,
                int(torch.numel(sample_weight_window) / (max_num_updates * BATCH_SIZE)),
            ).squeeze()

        lifetime_compute_result = torch.tensor(
            mean_squared_error(
                target_np,
                input_np,
                sample_weight=sample_weight_np,
                multioutput=multioutput,
            ),
            dtype=torch.float32,
        ).squeeze()
        window_compute_result = torch.tensor(
            mean_squared_error(
                target_window_np,
                input_window_np,
                sample_weight=sample_weight_window_np,
                multioutput=multioutput,
            ),
            dtype=torch.float32,
        ).squeeze()
        state_names = (
            {
                "max_num_updates",
                "total_updates",
                "sum_squared_error",
                "sum_weight",
                "windowed_sum_squared_error",
                "windowed_sum_weight",
            }
            if enable_lifetime
            else {
                "max_num_updates",
                "total_updates",
                "windowed_sum_squared_error",
                "windowed_sum_weight",
            }
        )

        update_kwargs = (
            {
                "input": input,
                "target": target,
            }
            if sample_weight is None
            else {
                "input": input,
                "target": target,
                "sample_weight": sample_weight,
            }
        )
        compute_result = (
            (lifetime_compute_result, window_compute_result)
            if enable_lifetime
            else window_compute_result
        )
        merge_and_compute_result = (
            (lifetime_compute_result, lifetime_compute_result)
            if enable_lifetime
            else lifetime_compute_result
        )
        self.run_class_implementation_tests(
            metric=WindowedMeanSquaredError(
                num_tasks=num_tasks,
                max_num_updates=max_num_updates,
                enable_lifetime=enable_lifetime,
                multioutput=multioutput,
            ),
            state_names=state_names,
            update_kwargs=update_kwargs,
            compute_result=compute_result,
            merge_and_compute_result=merge_and_compute_result,
        )

    def test_mean_squared_error_class_base(self) -> None:
        input = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE)
        target = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE)
        self._test_mean_squared_error_class_with_input(
            input,
            target,
            num_tasks=1,
            max_num_updates=2,
            enable_lifetime=False,
        )

        self._test_mean_squared_error_class_with_input(
            input,
            target,
            num_tasks=1,
            max_num_updates=2,
            enable_lifetime=True,
        )

        input = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE)
        target = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE)
        self._test_mean_squared_error_class_with_input(
            input,
            target,
            num_tasks=1,
            max_num_updates=2,
            enable_lifetime=False,
            multioutput="raw_values",
        )
        self._test_mean_squared_error_class_with_input(
            input,
            target,
            num_tasks=1,
            max_num_updates=2,
            enable_lifetime=True,
            multioutput="raw_values",
        )

        input = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE, 2)
        target = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE, 2)

        self._test_mean_squared_error_class_with_input(
            input,
            target,
            num_tasks=2,
            max_num_updates=2,
            enable_lifetime=False,
        )
        self._test_mean_squared_error_class_with_input(
            input,
            target,
            num_tasks=2,
            max_num_updates=2,
            enable_lifetime=True,
        )

        input = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE, 3)
        target = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE, 3)
        self._test_mean_squared_error_class_with_input(
            input,
            target,
            num_tasks=3,
            max_num_updates=2,
            enable_lifetime=False,
            multioutput="raw_values",
        )

    def test_mean_squared_error_class_valid_weight(self) -> None:
        sample_weight = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE)
        input = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE)
        target = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE)
        self._test_mean_squared_error_class_with_input(
            input,
            target,
            sample_weight=sample_weight,
            num_tasks=1,
            max_num_updates=2,
            enable_lifetime=False,
        )
        self._test_mean_squared_error_class_with_input(
            input,
            target,
            sample_weight=sample_weight,
            num_tasks=1,
            max_num_updates=2,
            enable_lifetime=True,
        )

        input = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE, 5)
        target = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE, 5)
        self._test_mean_squared_error_class_with_input(
            input,
            target,
            sample_weight=sample_weight,
            num_tasks=5,
            max_num_updates=2,
            enable_lifetime=False,
        )

        input = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE)
        target = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE)
        self._test_mean_squared_error_class_with_input(
            input,
            target,
            sample_weight=sample_weight,
            num_tasks=1,
            max_num_updates=2,
            enable_lifetime=False,
            multioutput="raw_values",
        )
        self._test_mean_squared_error_class_with_input(
            input,
            target,
            sample_weight=sample_weight,
            num_tasks=1,
            max_num_updates=2,
            enable_lifetime=True,
            multioutput="raw_values",
        )
        input = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE, 5)
        target = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE, 5)
        self._test_mean_squared_error_class_with_input(
            input,
            target,
            sample_weight=sample_weight,
            num_tasks=5,
            max_num_updates=2,
            enable_lifetime=False,
            multioutput="raw_values",
        )
        self._test_mean_squared_error_class_with_input(
            input,
            target,
            sample_weight=sample_weight,
            num_tasks=5,
            max_num_updates=2,
            enable_lifetime=True,
            multioutput="raw_values",
        )

    def test_mean_squared_error_class_update_input_shape_different(self) -> None:
        update_input = [
            torch.randn(5, 2),
            torch.randn(8, 2),
            torch.randn(2, 2),
            torch.randn(5, 2),
        ]

        update_target = [
            torch.randn(5, 2),
            torch.randn(8, 2),
            torch.randn(2, 2),
            torch.randn(5, 2),
        ]

        num_tasks = 2
        max_num_updates = 2
        enable_lifetime = False
        multioutput = "uniform_average"

        self.run_class_implementation_tests(
            metric=WindowedMeanSquaredError(
                num_tasks=num_tasks,
                max_num_updates=max_num_updates,
                enable_lifetime=enable_lifetime,
                multioutput=multioutput,
            ),
            state_names={
                "max_num_updates",
                "total_updates",
                "windowed_sum_squared_error",
                "windowed_sum_weight",
            },
            update_kwargs={
                "input": update_input,
                "target": update_target,
            },
            compute_result=(
                torch.tensor(
                    mean_squared_error(
                        torch.cat(update_target[-2:], dim=0),
                        torch.cat(update_input[-2:], dim=0),
                        multioutput=multioutput,
                    )
                )
                .to(torch.float32)
                .squeeze()
            ),
            merge_and_compute_result=(
                torch.tensor(
                    mean_squared_error(
                        torch.cat(update_target, dim=0),
                        torch.cat(update_input, dim=0),
                        multioutput=multioutput,
                    )
                )
                .to(torch.float32)
                .squeeze()
            ),
            num_total_updates=4,
            num_processes=2,
        )

        update_input = [
            torch.randn(5, 2),
            torch.randn(8, 2),
            torch.randn(2, 2),
            torch.randn(5, 2),
        ]

        update_target = [
            torch.randn(5, 2),
            torch.randn(8, 2),
            torch.randn(2, 2),
            torch.randn(5, 2),
        ]

        update_weight = [
            torch.randn(5),
            torch.randn(8),
            torch.randn(2),
            torch.randn(5),
        ]

        num_tasks = 2
        max_num_updates = 2
        enable_lifetime = False
        multioutput = "uniform_average"

        self.run_class_implementation_tests(
            metric=WindowedMeanSquaredError(
                num_tasks=num_tasks,
                max_num_updates=max_num_updates,
                enable_lifetime=enable_lifetime,
                multioutput=multioutput,
            ),
            state_names={
                "max_num_updates",
                "total_updates",
                "windowed_sum_squared_error",
                "windowed_sum_weight",
            },
            update_kwargs={
                "input": update_input,
                "target": update_target,
                "sample_weight": update_weight,
            },
            compute_result=(
                torch.tensor(
                    mean_squared_error(
                        torch.cat(update_target[-2:], dim=0),
                        torch.cat(update_input[-2:], dim=0),
                        sample_weight=torch.cat(update_weight[-2:], dim=0),
                        multioutput=multioutput,
                    )
                )
                .to(torch.float32)
                .squeeze()
            ),
            merge_and_compute_result=(
                torch.tensor(
                    mean_squared_error(
                        torch.cat(update_target, dim=0),
                        torch.cat(update_input, dim=0),
                        sample_weight=torch.cat(update_weight, dim=0),
                        multioutput=multioutput,
                    )
                )
                .to(torch.float32)
                .squeeze()
            ),
            num_total_updates=4,
            num_processes=2,
        )

    def test_mean_squared_error_class_invalid_input(self) -> None:
        metric = WindowedMeanSquaredError()
        with self.assertRaisesRegex(
            ValueError,
            "The dimension `input` and `target` should be 1D or 2D, "
            r"got shapes torch.Size\(\[3, 2, 2\]\) and torch.Size\(\[3, 2, 2\]\).",
        ):
            metric.update(torch.rand(3, 2, 2), torch.rand(3, 2, 2))

        with self.assertRaisesRegex(
            ValueError,
            "The `input` and `target` should have the same size, "
            r"got shapes torch.Size\(\[3, 2\]\) and torch.Size\(\[3, 5\]\).",
        ):
            metric.update(torch.rand(3, 2), torch.rand(3, 5))

        with self.assertRaisesRegex(
            ValueError,
            "The first dimension of `input`, `target` and `sample_weight` should be the same size, "
            r"got shapes torch.Size\(\[3, 2\]\), torch.Size\(\[3, 2\]\) and torch.Size\(\[5\]\).",
        ):
            metric.update(
                torch.rand(3, 2), torch.rand(3, 2), sample_weight=torch.rand(5)
            )

        with self.assertRaisesRegex(
            ValueError,
            "The `multioutput` must be either `raw_values` or `uniform_average`, "
            r"got multioutput=gaussian.",
        ):
            WindowedMeanSquaredError(multioutput="gaussian")

        with self.assertRaisesRegex(
            ValueError,
            r"`num_tasks` value should be greater than and equal to 1,",
        ):
            metric = WindowedMeanSquaredError(num_tasks=0)

        with self.assertRaisesRegex(
            ValueError,
            r"`num_tasks = 1`, `input` is expected to be one-dimensional tensor,",
        ):
            metric = WindowedMeanSquaredError()
            metric.update(torch.rand(3, 2), torch.rand(3, 2))

        with self.assertRaisesRegex(
            ValueError,
            r"`num_tasks = 2`, `input`'s shape is expected to be",
        ):
            metric = WindowedMeanSquaredError(num_tasks=2)
            metric.update(torch.rand(3, 3), torch.rand(3, 3))

        with self.assertRaisesRegex(
            ValueError,
            r"`max_num_updates` value should be greater than and equal to 1, ",
        ):
            metric = WindowedMeanSquaredError(max_num_updates=0)
