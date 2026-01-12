# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch
from scipy.stats import wasserstein_distance as sp_wasserstein
from torcheval.metrics.statistical.wasserstein import Wasserstein1D
from torcheval.utils.random_data import get_rand_data_wasserstein1d
from torcheval.utils.test_utils.metric_class_tester import (
    BATCH_SIZE,
    MetricClassTester,
    NUM_PROCESSES,
)

NUM_TOTAL_UPDATES = 8


class TestWasserstein1D(MetricClassTester):
    def _get_scipy_equivalent(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_weights: torch.Tensor | None = None,
        y_weights: torch.Tensor | None = None,
        device: str = "cpu",
    ) -> torch.Tensor:
        # Convert inputs to scipy style inputs
        x_np = x.numpy().flatten()
        y_np = y.numpy().flatten()
        if x_weights is not None:
            x_weights_np = x_weights.numpy().flatten()
        if y_weights is not None:
            y_weights_np = y_weights.numpy().flatten()

        scipy_result = [sp_wasserstein(x_np, y_np, x_weights_np, y_weights_np)]

        return torch.tensor(scipy_result, device=device).to(torch.float)

    def _check_against_scipy(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_weights: torch.Tensor | None = None,
        y_weights: torch.Tensor | None = None,
        device: str = "cpu",
    ) -> None:
        compute_result = self._get_scipy_equivalent(
            x.to("cpu"), y.to("cpu"), x_weights.to("cpu"), y_weights.to("cpu"), device
        )

        self.run_class_implementation_tests(
            metric=Wasserstein1D(device=device),
            state_names={
                "dist_1_samples",
                "dist_2_samples",
                "dist_1_weights",
                "dist_2_weights",
            },
            update_kwargs={
                "new_samples_dist_1": x,
                "new_samples_dist_2": y,
                "new_weights_dist_1": x_weights,
                "new_weights_dist_2": y_weights,
            },
            compute_result=compute_result,
            num_total_updates=NUM_TOTAL_UPDATES,
            num_processes=NUM_PROCESSES,
        )

    def test_wasserstein1d_valid_input(self) -> None:
        # Checking with distribution values only
        metric = Wasserstein1D()
        x = torch.tensor([5, -5, -7, 9, -3])
        y = torch.tensor([9, -7, 5, -4, -2])
        metric.update(x, y)
        result = metric.compute()
        expected = torch.tensor([0.39999999999999997])
        torch.testing.assert_close(
            result,
            expected,
            equal_nan=True,
            atol=1e-4,
            rtol=1e-3,
        )

        # Checking with distribution and weight values
        metric = Wasserstein1D()
        x = torch.tensor([-13, -9, -19, 11, -18, -20, 8, 2, -8, -18])
        y = torch.tensor([9, 6, -5, -11, 9, -4, -13, -19, -14, 4])
        x_weights = torch.tensor([3, 3, 1, 2, 2, 3, 2, 2, 2, 3])
        y_weights = torch.tensor([2, 2, 1, 1, 2, 2, 1, 1, 1, 1])
        metric.update(x, y, x_weights, y_weights)
        result = metric.compute()
        expected = torch.tensor([8.149068322981368])
        torch.testing.assert_close(
            result,
            expected,
            equal_nan=True,
            atol=1e-4,
            rtol=1e-3,
        )

        # Checking with different distribution shapes
        metric = Wasserstein1D()
        x = torch.tensor([5, -5, -7, 9, -3])
        y = torch.tensor([9, -7, 5, -4, -2, 4, -1])
        metric.update(x, y)
        result = metric.compute()
        expected = torch.tensor([1.4571428571428569])
        torch.testing.assert_close(
            result,
            expected,
            equal_nan=True,
            atol=1e-4,
            rtol=1e-3,
        )

        # Checking with identical distributions
        metric = Wasserstein1D()
        x = torch.tensor([-13, -9, -19, 11, -18, -20, 8, 2, -8, -18])
        x_weights = torch.tensor([3, 3, 1, 2, 2, 3, 2, 2, 2, 3])
        metric.update(x, x, x_weights, x_weights)
        result = metric.compute()
        expected = torch.tensor([0.0])
        torch.testing.assert_close(
            result,
            expected,
            equal_nan=True,
            atol=1e-4,
            rtol=1e-3,
        )

    def test_wasserstein1d_random_data_getter(self) -> None:
        for _ in range(10):
            x, y, x_weights, y_weights = get_rand_data_wasserstein1d(
                num_updates=NUM_TOTAL_UPDATES, batch_size=BATCH_SIZE
            )

            self._check_against_scipy(x, y, x_weights, y_weights)

    def test_wasserstein1d_invalid_input(self) -> None:
        metric = Wasserstein1D()
        with self.assertRaisesRegex(
            ValueError, "Distribution has to be one dimensional."
        ):
            metric.update(torch.rand(4, 2), torch.rand(7))

        with self.assertRaisesRegex(
            ValueError, "Distribution has to be one dimensional."
        ):
            metric.update(torch.rand(4), torch.rand(7, 3))

        with self.assertRaisesRegex(ValueError, "Distribution cannot be empty."):
            metric.update(torch.rand(4), torch.tensor([]))

        with self.assertRaisesRegex(ValueError, "Distribution cannot be empty."):
            metric.update(torch.tensor([]), torch.rand(5))

        with self.assertRaisesRegex(
            ValueError, "Weight tensor sum must be positive-finite."
        ):
            metric.update(
                torch.rand(4), torch.rand(4), torch.tensor([torch.inf]), torch.rand(4)
            )

        with self.assertRaisesRegex(
            ValueError, "Weight tensor sum must be positive-finite."
        ):
            metric.update(
                torch.rand(4), torch.rand(4), torch.rand(4), torch.tensor([torch.inf])
            )

        with self.assertRaisesRegex(
            ValueError,
            "Distribution values and weight tensors must be of the same shape, "
            "got shapes "
            r"torch.Size\(\[4\]\) and torch.Size\(\[7\]\).",
        ):
            metric.update(torch.rand(4), torch.rand(4), torch.rand(7), torch.rand(4))

        with self.assertRaisesRegex(
            ValueError,
            "Distribution values and weight tensors must be of the same shape, "
            "got shapes "
            r"torch.Size\(\[6\]\) and torch.Size\(\[10\]\).",
        ):
            metric.update(torch.rand(6), torch.rand(6), torch.rand(6), torch.rand(10))

        with self.assertRaisesRegex(ValueError, "All weights must be non-negative."):
            metric.update(
                torch.rand(4), torch.rand(4), torch.tensor([1, -1, 2, 3]), torch.rand(4)
            )

        with self.assertRaisesRegex(ValueError, "All weights must be non-negative."):
            metric.update(
                torch.rand(4), torch.rand(4), torch.rand(4), torch.tensor([1, -1, 2, 3])
            )

        with self.assertRaisesRegex(ValueError, "All weights must be non-negative."):
            metric.update(
                torch.rand(4),
                torch.rand(4),
                torch.tensor([-1.0, -2.0, 0.0, 1.0]),
                torch.rand(4),
            )

        with self.assertRaisesRegex(ValueError, "All weights must be non-negative."):
            metric.update(
                torch.rand(4),
                torch.rand(4),
                torch.rand(4),
                torch.tensor([-1.5, -1.0, 0.5, 0.75]),
            )
