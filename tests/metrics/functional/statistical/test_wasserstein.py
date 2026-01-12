# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np
import torch
from scipy.stats import wasserstein_distance as sp_wasserstein
from torcheval.metrics.functional.statistical.wasserstein import wasserstein_1d
from torcheval.utils import random_data as rd


class TestWasserstein1D(unittest.TestCase):
    def _get_scipy_equivalent(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_weights: torch.Tensor | None = None,
        y_weights: torch.Tensor | None = None,
        device: str = "cpu",
    ) -> torch.Tensor:
        # Convert inputs to scipy style inputs
        x_np = x.numpy()
        y_np = y.numpy()
        if x_weights is not None:
            x_weights_np = x_weights.numpy()
        if y_weights is not None:
            y_weights_np = y_weights.numpy()

        if x.ndim == 1:
            scipy_result = [sp_wasserstein(x_np, y_np, x_weights_np, y_weights_np)]
        else:
            scipy_result = np.stack(
                [
                    sp_wasserstein(sp_x, sp_y, sp_x_w, sp_y_w)
                    for sp_x, sp_y, sp_x_w, sp_y_w in zip(
                        x_np, y_np, x_weights_np, y_weights_np
                    )
                ]
            )

        return torch.tensor(scipy_result, device=device).to(torch.float)

    def _test_wasserstein1d_with_input(
        self,
        compute_result: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        x_weights: torch.Tensor | None = None,
        y_weights: torch.Tensor | None = None,
    ) -> None:
        if x.ndim == 1:
            my_compute_result = wasserstein_1d(x, y, x_weights, y_weights)
            torch.testing.assert_close(
                my_compute_result,
                compute_result,
                equal_nan=True,
                atol=1e-8,
                rtol=1e-5,
            )

            # Also test for cuda
            if torch.cuda.is_available():
                compute_result_cuda = tuple(c.to("cuda") for c in compute_result)
                my_compute_result_cuda = tuple(c.to("cuda") for c in my_compute_result)

                torch.testing.assert_close(
                    my_compute_result_cuda,
                    compute_result_cuda,
                    equal_nan=True,
                    atol=1e-8,
                    rtol=1e-5,
                )
        else:
            my_compute_result = torch.tensor(
                [
                    wasserstein_1d(x, y, x_weights, y_weights)
                    for x, y, x_weights, y_weights in zip(x, y, x_weights, y_weights)
                ]
            ).to(x.device)
            torch.testing.assert_close(
                my_compute_result,
                compute_result,
                equal_nan=True,
                atol=1e-8,
                rtol=1e-5,
            )

            # Also test for cuda
            if torch.cuda.is_available():
                compute_result_cuda = tuple(c.to("cuda") for c in compute_result)
                my_compute_result_cuda = tuple(c.to("cuda") for c in my_compute_result)

                torch.testing.assert_close(
                    my_compute_result_cuda,
                    compute_result_cuda,
                    equal_nan=True,
                    atol=1e-8,
                    rtol=1e-5,
                )

    def test_wasserstein1d_distribution_values_only(self) -> None:
        x = torch.tensor([5, -5, -7, 9, -3])
        y = torch.tensor([9, -7, 5, -4, -2])
        self._test_wasserstein1d_with_input(torch.tensor([0.39999999999999997]), x, y)

    def test_wasserstein1d_distribution_and_weight_values(self) -> None:
        x = torch.tensor([-13, -9, -19, 11, -18, -20, 8, 2, -8, -18])
        y = torch.tensor([9, 6, -5, -11, 9, -4, -13, -19, -14, 4])
        x_weights = torch.tensor([3, 3, 1, 2, 2, 3, 2, 2, 2, 3])
        y_weights = torch.tensor([2, 2, 1, 1, 2, 2, 1, 1, 1, 1])
        self._test_wasserstein1d_with_input(
            torch.tensor([8.149068322981368]), x, y, x_weights, y_weights
        )

    def test_wasserstein1d_different_distribution_shape(self) -> None:
        x = torch.tensor([5, -5, -7, 9, -3])
        y = torch.tensor([9, -7, 5, -4, -2, 4, -1])
        self._test_wasserstein1d_with_input(torch.tensor([1.4571428571428569]), x, y)

    def test_wasserstein1d_identical_distributions(self) -> None:
        x = torch.tensor([-13, -9, -19, 11, -18, -20, 8, 2, -8, -18])
        x_weights = torch.tensor([3, 3, 1, 2, 2, 3, 2, 2, 2, 3])
        self._test_wasserstein1d_with_input(
            torch.tensor([0.0]), x, x, x_weights, x_weights
        )

    def test_wasserstein1d_randomized_data_getter(self) -> None:
        num_updates = 1
        batch_size = 32
        device = "cuda" if torch.cuda.is_available() else "cpu"

        for _ in range(10):
            x, y, x_weights, y_weights = rd.get_rand_data_wasserstein1d(
                num_updates, batch_size, device
            )

            compute_result = self._get_scipy_equivalent(
                x.to("cpu"),
                y.to("cpu"),
                x_weights.to("cpu"),
                y_weights.to("cpu"),
                device,
            )

            self._test_wasserstein1d_with_input(
                compute_result, x, y, x_weights, y_weights
            )

        num_updates = 8
        batch_size = 32
        device = "cuda" if torch.cuda.is_available() else "cpu"

        for _ in range(10):
            x, y, x_weights, y_weights = rd.get_rand_data_wasserstein1d(
                num_updates, batch_size, device
            )

            compute_result = self._get_scipy_equivalent(
                x.to("cpu"),
                y.to("cpu"),
                x_weights.to("cpu"),
                y_weights.to("cpu"),
                device,
            )

            self._test_wasserstein1d_with_input(
                compute_result, x, y, x_weights, y_weights
            )

    def test_wasserstein1d_invalid_input(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "Distribution has to be one dimensional."
        ):
            wasserstein_1d(torch.rand(4, 2), torch.rand(7))

        with self.assertRaisesRegex(
            ValueError, "Distribution has to be one dimensional."
        ):
            wasserstein_1d(torch.rand(4), torch.rand(7, 3))

        with self.assertRaisesRegex(ValueError, "Distribution cannot be empty."):
            wasserstein_1d(torch.rand(4), torch.tensor([]))

        with self.assertRaisesRegex(ValueError, "Distribution cannot be empty."):
            wasserstein_1d(torch.tensor([]), torch.rand(5))

        with self.assertRaisesRegex(
            ValueError, "Weight tensor sum must be positive-finite."
        ):
            wasserstein_1d(
                torch.rand(4), torch.rand(4), torch.tensor([torch.inf]), torch.rand(4)
            )

        with self.assertRaisesRegex(
            ValueError, "Weight tensor sum must be positive-finite."
        ):
            wasserstein_1d(
                torch.rand(4), torch.rand(4), torch.rand(4), torch.tensor([torch.inf])
            )

        with self.assertRaisesRegex(
            ValueError,
            "Distribution values and weight tensors must be of the same shape, "
            "got shapes "
            r"torch.Size\(\[4\]\) and torch.Size\(\[7\]\).",
        ):
            wasserstein_1d(torch.rand(4), torch.rand(4), torch.rand(7), torch.rand(4))

        with self.assertRaisesRegex(
            ValueError,
            "Distribution values and weight tensors must be of the same shape, "
            "got shapes "
            r"torch.Size\(\[6\]\) and torch.Size\(\[10\]\).",
        ):
            wasserstein_1d(torch.rand(6), torch.rand(6), torch.rand(6), torch.rand(10))

        with self.assertRaisesRegex(ValueError, "All weights must be non-negative."):
            wasserstein_1d(
                torch.rand(4), torch.rand(4), torch.tensor([1, -1, 2, 3]), torch.rand(4)
            )

        with self.assertRaisesRegex(ValueError, "All weights must be non-negative."):
            wasserstein_1d(
                torch.rand(4), torch.rand(4), torch.rand(4), torch.tensor([1, -1, 2, 3])
            )

        with self.assertRaisesRegex(ValueError, "All weights must be non-negative."):
            wasserstein_1d(
                torch.rand(4),
                torch.rand(4),
                torch.tensor([-1.0, -2.0, 0.0, 1.0]),
                torch.rand(4),
            )

        with self.assertRaisesRegex(ValueError, "All weights must be non-negative."):
            wasserstein_1d(
                torch.rand(4),
                torch.rand(4),
                torch.rand(4),
                torch.tensor([-1.5, -1.0, 0.5, 0.75]),
            )
