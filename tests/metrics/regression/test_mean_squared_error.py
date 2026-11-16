# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch
from sklearn.metrics import mean_squared_error
from torcheval.metrics import MeanSquaredError
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
        sample_weight: torch.Tensor | None = None,
        multioutput: str = "uniform_average",
    ) -> None:
        target_np = target.reshape(
            -1, int(torch.numel(target) / (NUM_TOTAL_UPDATES * BATCH_SIZE))
        ).squeeze()
        input_np = input.reshape(
            -1, int(torch.numel(target) / (NUM_TOTAL_UPDATES * BATCH_SIZE))
        ).squeeze()

        sample_weight_np = None
        if sample_weight is not None:
            sample_weight_np = sample_weight.reshape(
                -1, int(torch.numel(sample_weight) / (NUM_TOTAL_UPDATES * BATCH_SIZE))
            ).squeeze()

        compute_result = torch.tensor(
            mean_squared_error(
                target_np,
                input_np,
                sample_weight=sample_weight_np,
                multioutput=multioutput,
            ),
            dtype=torch.float32,
        ).squeeze()
        if sample_weight is None:
            self.run_class_implementation_tests(
                metric=MeanSquaredError(
                    multioutput=multioutput,
                ),
                state_names={
                    "sum_squared_error",
                    "sum_weight",
                },
                update_kwargs={
                    "input": input,
                    "target": target,
                },
                compute_result=compute_result,
            )
        else:
            self.run_class_implementation_tests(
                metric=MeanSquaredError(
                    multioutput=multioutput,
                ),
                state_names={
                    "sum_squared_error",
                    "sum_weight",
                },
                update_kwargs={
                    "input": input,
                    "target": target,
                    "sample_weight": sample_weight,
                },
                compute_result=compute_result,
            )

    def test_mean_squared_error_class_base(self) -> None:
        input = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE)
        target = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE)
        self._test_mean_squared_error_class_with_input(input, target)

        input = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE)
        target = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE)
        self._test_mean_squared_error_class_with_input(
            input, target, multioutput="raw_values"
        )

        input = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE, 2)
        target = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE, 2)
        self._test_mean_squared_error_class_with_input(input, target)

        input = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE, 3)
        target = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE, 3)
        self._test_mean_squared_error_class_with_input(
            input, target, multioutput="raw_values"
        )

    def test_mean_squared_error_class_valid_weight(self) -> None:
        sample_weight = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE)
        input = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE)
        target = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE)
        self._test_mean_squared_error_class_with_input(
            input, target, sample_weight=sample_weight
        )
        input = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE, 5)
        target = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE, 5)
        self._test_mean_squared_error_class_with_input(
            input, target, sample_weight=sample_weight
        )
        input = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE)
        target = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE)
        self._test_mean_squared_error_class_with_input(
            input, target, sample_weight=sample_weight, multioutput="raw_values"
        )
        input = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE, 5)
        target = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE, 5)
        self._test_mean_squared_error_class_with_input(
            input, target, sample_weight=sample_weight, multioutput="raw_values"
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

        self.run_class_implementation_tests(
            metric=MeanSquaredError(),
            state_names={
                "sum_squared_error",
                "sum_weight",
            },
            update_kwargs={
                "input": update_input,
                "target": update_target,
            },
            compute_result=(
                torch.tensor(
                    mean_squared_error(
                        torch.cat(update_target, dim=0),
                        torch.cat(update_input, dim=0),
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

        self.run_class_implementation_tests(
            metric=MeanSquaredError(),
            state_names={
                "sum_squared_error",
                "sum_weight",
            },
            update_kwargs={
                "input": update_input,
                "target": update_target,
                "sample_weight": update_weight,
            },
            compute_result=(
                torch.tensor(
                    mean_squared_error(
                        torch.cat(update_target, dim=0),
                        torch.cat(update_input, dim=0),
                        sample_weight=torch.cat(update_weight, dim=0),
                    )
                )
                .to(torch.float32)
                .squeeze()
            ),
            num_total_updates=4,
            num_processes=2,
        )

    def test_mean_squared_error_class_invalid_input(self) -> None:
        metric = MeanSquaredError()
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
            MeanSquaredError(multioutput="gaussian")
