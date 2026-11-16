# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest

import torch
from sklearn.metrics import r2_score
from torcheval.metrics.functional import r2_score as my_r2_score
from torcheval.utils.test_utils.metric_class_tester import BATCH_SIZE


class TestR2Score(unittest.TestCase):
    def _test_r2_score_with_input(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        multioutput: str = "uniform_average",
        num_regressors: int = 0,
    ) -> None:
        compute_result = (
            torch.tensor(r2_score(target, input, multioutput=multioutput))
            .squeeze()
            .to(torch.float32)
        )
        if num_regressors != 0:
            num_obs = target.shape[0]
            compute_result = 1 - (1 - compute_result) * (num_obs - 1) / (
                num_obs - num_regressors - 1
            )
        my_compute_result = my_r2_score(
            input, target, multioutput=multioutput, num_regressors=num_regressors
        )
        torch.testing.assert_close(
            my_compute_result,
            compute_result,
            equal_nan=True,
            atol=1e-6,
            rtol=1e-4,
        )

    def test_r2_score_base(self) -> None:
        input = torch.rand(BATCH_SIZE)
        target = torch.rand(BATCH_SIZE)
        self._test_r2_score_with_input(input, target)
        input = torch.rand(BATCH_SIZE, 5)
        target = torch.rand(BATCH_SIZE, 5)
        self._test_r2_score_with_input(input, target)
        input = torch.rand(BATCH_SIZE)
        target = torch.rand(BATCH_SIZE)
        self._test_r2_score_with_input(input, target, "raw_values")
        input = torch.rand(BATCH_SIZE, 5)
        target = torch.rand(BATCH_SIZE, 5)
        self._test_r2_score_with_input(input, target, "raw_values")
        input = torch.rand(BATCH_SIZE)
        target = torch.rand(BATCH_SIZE)
        self._test_r2_score_with_input(input, target, "variance_weighted")
        input = torch.rand(BATCH_SIZE, 5)
        target = torch.rand(BATCH_SIZE, 5)
        self._test_r2_score_with_input(input, target, "variance_weighted")

    def test_r2_score_adjusted(self) -> None:
        input = torch.rand(BATCH_SIZE)
        target = torch.rand(BATCH_SIZE)
        self._test_r2_score_with_input(input, target, num_regressors=1)
        input = torch.rand(BATCH_SIZE)
        target = torch.rand(BATCH_SIZE)
        self._test_r2_score_with_input(input, target, "raw_values", num_regressors=1)
        input = torch.rand(BATCH_SIZE)
        target = torch.rand(BATCH_SIZE)
        self._test_r2_score_with_input(
            input, target, "variance_weighted", num_regressors=2
        )
        input = torch.rand(BATCH_SIZE, 2)
        target = torch.rand(BATCH_SIZE, 2)
        self._test_r2_score_with_input(input, target)
        input = torch.rand(BATCH_SIZE, 3)
        target = torch.rand(BATCH_SIZE, 3)
        self._test_r2_score_with_input(input, target, "raw_values", num_regressors=3)
        input = torch.rand(BATCH_SIZE, 3)
        target = torch.rand(BATCH_SIZE, 3)
        self._test_r2_score_with_input(
            input, target, "variance_weighted", num_regressors=4
        )

    def test_r2_score_invalid_intialization(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "The `multioutput` must be either `raw_values` or `uniform_average` or `variance_weighted`, "
            r"got multioutput=gaussian_distribution.",
        ):
            my_r2_score(
                torch.rand(BATCH_SIZE),
                torch.rand(BATCH_SIZE),
                multioutput="gaussian_distribution",
            )

        with self.assertRaisesRegex(
            ValueError,
            "The `num_regressors` must an integer larger or equal to zero, "
            r"got num_regressors=-1.",
        ):
            my_r2_score(
                torch.rand(BATCH_SIZE), torch.rand(BATCH_SIZE), num_regressors=-1
            )

    def test_r2_score_invalid_input(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "The dimension `input` and `target` should be 1D or 2D, "
            r"got shapes torch.Size\(\[3, 2, 2\]\) and torch.Size\(\[3, 2, 2\]\).",
        ):
            my_r2_score(torch.rand(3, 2, 2), torch.rand(3, 2, 2))

        with self.assertRaisesRegex(
            ValueError,
            "The `input` and `target` should have the same size, "
            r"got shapes torch.Size\(\[4, 2\]\) and torch.Size\(\[3\]\).",
        ):
            my_r2_score(torch.rand(4, 2), torch.rand(3))

    def test_r2_score_invalid_compute(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "The `num_regressors` must be smaller than n_samples - 1, "
            r"got num_regressors=10, n_samples=3.",
        ):
            my_r2_score(torch.rand(3, 3), torch.rand(3, 3), num_regressors=10)
