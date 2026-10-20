# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from copy import deepcopy

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

    def test_merge_state_preserves_independent_states(self) -> None:
        for outputs in (1, 2):
            with self.subTest(outputs=outputs):
                targets = [
                    torch.tensor([[1.0], [3.0]]).repeat(1, outputs) + offset
                    for offset in (0, 4, 8)
                ]
                inputs = [target - torch.tensor([[0.0], [1.0]]) for target in targets]
                donors = [
                    R2Score(multioutput="raw_values").update(pred, target)
                    for pred, target in zip(inputs[:2], targets[:2])
                ]
                states = [deepcopy(donor.state_dict()) for donor in donors]
                results = [donor.compute() for donor in donors]
                merged = R2Score(multioutput="raw_values").merge_state(donors)
                for donor, state, result in zip(donors, states, results):
                    torch.testing.assert_close(donor.state_dict(), state)
                    torch.testing.assert_close(donor.compute(), result)
                for count in (2, 3):
                    if count == 3:
                        merged.update(inputs[2], targets[2])
                    pred, target = torch.cat(inputs[:count]), torch.cat(targets[:count])
                    expected = 1 - (pred - target).square().sum(0) / (
                        target - target.mean(0)
                    ).square().sum(0)
                    torch.testing.assert_close(merged.compute(), expected)
                    for donor, state in zip(donors, states):
                        torch.testing.assert_close(donor.state_dict(), state)
                merged_state = deepcopy(merged.state_dict())
                donors[0].update(inputs[2], targets[2])
                torch.testing.assert_close(merged.state_dict(), merged_state)

    def test_merge_state_single_donor_then_update(self) -> None:
        target = torch.tensor([[1.0], [3.0]])
        pred = torch.tensor([[1.0], [2.0]])
        donor = R2Score().update(pred, target)
        state = deepcopy(donor.state_dict())
        merged = R2Score().merge_state([donor]).update(pred + 4, target + 4)
        torch.testing.assert_close(donor.state_dict(), state)
        pooled_pred, pooled_target = torch.cat([pred, pred + 4]), torch.cat(
            [target, target + 4]
        )
        expected = (
            1
            - (pooled_pred - pooled_target).square().sum()
            / (pooled_target - pooled_target.mean()).square().sum()
        )
        torch.testing.assert_close(merged.compute(), expected)
