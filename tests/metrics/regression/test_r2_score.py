# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch
from sklearn.metrics import r2_score
from torcheval.metrics import R2Score
from torcheval.utils.test_utils.metric_class_tester import (
    BATCH_SIZE,
    MetricClassTester,
    NUM_TOTAL_UPDATES,
)


class TestR2Score(MetricClassTester):
    def _test_r2_score_class_with_input(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        multioutput: str = "uniform_average",
        num_regressors: int = 0,
        n_samples: int = NUM_TOTAL_UPDATES * BATCH_SIZE,
    ) -> None:
        target_np = target.reshape(-1, int(torch.numel(target) / n_samples)).squeeze()
        input_np = input.reshape(-1, int(torch.numel(target) / n_samples)).squeeze()
        compute_result = (
            torch.tensor(r2_score(target_np, input_np, multioutput=multioutput))
            .to(torch.float32)
            .squeeze()
        )

        if num_regressors != 0:
            num_obs = target_np.shape[0]
            compute_result = 1 - (1 - compute_result) * (num_obs - 1) / (
                num_obs - num_regressors - 1
            )

        self.run_class_implementation_tests(
            metric=R2Score(multioutput=multioutput, num_regressors=num_regressors),
            state_names={
                "sum_squared_obs",
                "sum_obs",
                "sum_squared_residual",
                "num_obs",
            },
            update_kwargs={
                "input": input,
                "target": target,
            },
            compute_result=compute_result,
        )

    def test_r2_score_class_base(self) -> None:
        input = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE)
        target = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE)
        self._test_r2_score_class_with_input(input, target)
        input = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE)
        target = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE)
        self._test_r2_score_class_with_input(input, target, "raw_values")
        input = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE)
        target = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE)
        self._test_r2_score_class_with_input(input, target, "variance_weighted")
        input = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE, 2)
        target = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE, 2)
        self._test_r2_score_class_with_input(input, target)
        input = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE, 3)
        target = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE, 3)
        self._test_r2_score_class_with_input(input, target, "raw_values")
        input = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE, 3)
        target = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE, 3)
        self._test_r2_score_class_with_input(input, target, "variance_weighted")

    def test_r2_score_class_adjusted(self) -> None:
        input = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE)
        target = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE)
        self._test_r2_score_class_with_input(input, target, num_regressors=1)
        input = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE)
        target = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE)
        self._test_r2_score_class_with_input(
            input, target, "raw_values", num_regressors=1
        )
        input = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE)
        target = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE)
        self._test_r2_score_class_with_input(
            input, target, "variance_weighted", num_regressors=2
        )
        input = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE, 2)
        target = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE, 2)
        self._test_r2_score_class_with_input(input, target)
        input = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE, 3)
        target = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE, 3)
        self._test_r2_score_class_with_input(
            input, target, "raw_values", num_regressors=3
        )
        input = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE, 3)
        target = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE, 3)
        self._test_r2_score_class_with_input(
            input, target, "variance_weighted", num_regressors=4
        )

    def test_r2_score_class_update_input_shape_different(self) -> None:
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
            metric=R2Score(),
            state_names={
                "sum_squared_obs",
                "sum_obs",
                "sum_squared_residual",
                "num_obs",
            },
            update_kwargs={
                "input": update_input,
                "target": update_target,
            },
            compute_result=(
                torch.tensor(
                    r2_score(
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

    def test_r2_score_class_invalid_intialization(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "The `multioutput` must be either `raw_values` or `uniform_average` or `variance_weighted`, "
            r"got multioutput=gaussian_distribution.",
        ):
            R2Score(multioutput="gaussian_distribution")

        with self.assertRaisesRegex(
            ValueError,
            "The `num_regressors` must an integer larger or equal to zero, "
            r"got num_regressors=-1.",
        ):
            R2Score(num_regressors=-1)

    def test_r2_score_class_invalid_input(self) -> None:
        metric = R2Score()
        with self.assertRaisesRegex(
            ValueError,
            "The dimension `input` and `target` should be 1D or 2D, "
            r"got shapes torch.Size\(\[3, 2, 2\]\) and torch.Size\(\[3, 2, 2\]\).",
        ):
            metric.update(torch.rand(3, 2, 2), torch.rand(3, 2, 2))

        with self.assertRaisesRegex(
            ValueError,
            "The `input` and `target` should have the same size, "
            r"got shapes torch.Size\(\[4, 2\]\) and torch.Size\(\[3\]\).",
        ):
            metric.update(torch.rand(4, 2), torch.rand(3))

    def test_r2_score_class_invalid_compute(self) -> None:
        metric = R2Score()
        with self.assertRaisesRegex(
            ValueError,
            "There is no enough data for computing. Needs at least two samples to calculate r2 score.",
        ):
            metric.compute()

        with self.assertRaisesRegex(
            ValueError,
            "The `num_regressors` must be smaller than n_samples - 1, "
            r"got num_regressors=10, n_samples=3.",
        ):
            metric.num_regressors = 10
            metric.update(torch.rand(3, 3), torch.rand(3, 3)).compute()
