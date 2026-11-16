# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest

import torch
from sklearn.metrics import mean_squared_error
from torcheval.metrics.functional import mean_squared_error as my_mean_squared_error
from torcheval.utils.test_utils.metric_class_tester import BATCH_SIZE


class TestMeanSquaredError(unittest.TestCase):
    def _test_mean_squared_error_with_input(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        sample_weight: torch.Tensor | None = None,
        multioutput: str = "uniform_average",
    ) -> None:
        compute_result = torch.tensor(
            mean_squared_error(
                target, input, sample_weight=sample_weight, multioutput=multioutput
            ),
            dtype=torch.float32,
        ).squeeze()
        my_compute_result = my_mean_squared_error(
            input, target, sample_weight=sample_weight, multioutput=multioutput
        )
        torch.testing.assert_close(
            my_compute_result,
            compute_result,
            equal_nan=True,
            atol=1e-8,
            rtol=1e-5,
        )

    def test_mean_squared_error_base(self) -> None:
        input = torch.rand(32)
        target = torch.rand(32)
        self._test_mean_squared_error_with_input(input, target)
        input = torch.rand(32, 5)
        target = torch.rand(32, 5)
        self._test_mean_squared_error_with_input(input, target)
        input = torch.rand(BATCH_SIZE)
        target = torch.rand(BATCH_SIZE)
        self._test_mean_squared_error_with_input(input, target)
        input = torch.rand(BATCH_SIZE, 5)
        target = torch.rand(BATCH_SIZE, 5)
        self._test_mean_squared_error_with_input(input, target)
        input = torch.rand(BATCH_SIZE)
        target = torch.rand(BATCH_SIZE)
        self._test_mean_squared_error_with_input(
            input, target, multioutput="raw_values"
        )
        input = torch.rand(BATCH_SIZE, 5)
        target = torch.rand(BATCH_SIZE, 5)
        self._test_mean_squared_error_with_input(
            input, target, multioutput="raw_values"
        )

    def test_mean_squared_error_valid_weight(self) -> None:
        sample_weight = torch.rand(BATCH_SIZE)
        input = torch.rand(BATCH_SIZE)
        target = torch.rand(BATCH_SIZE)
        self._test_mean_squared_error_with_input(
            input, target, sample_weight=sample_weight
        )

        input = torch.rand(BATCH_SIZE, 5)
        target = torch.rand(BATCH_SIZE, 5)
        self._test_mean_squared_error_with_input(
            input, target, sample_weight=sample_weight
        )

        input = torch.rand(BATCH_SIZE)
        target = torch.rand(BATCH_SIZE)
        self._test_mean_squared_error_with_input(
            input, target, sample_weight=sample_weight, multioutput="raw_values"
        )
        input = torch.rand(BATCH_SIZE, 5)
        target = torch.rand(BATCH_SIZE, 5)
        self._test_mean_squared_error_with_input(
            input, target, sample_weight=sample_weight, multioutput="raw_values"
        )

    def test_mean_squared_error_invalid_input(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "The dimension `input` and `target` should be 1D or 2D, "
            r"got shapes torch.Size\(\[3, 2, 2\]\) and torch.Size\(\[3, 2, 2\]\).",
        ):
            my_mean_squared_error(torch.rand(3, 2, 2), torch.rand(3, 2, 2))

        with self.assertRaisesRegex(
            ValueError,
            "The `input` and `target` should have the same size, "
            r"got shapes torch.Size\(\[3, 2\]\) and torch.Size\(\[3, 5\]\).",
        ):
            my_mean_squared_error(torch.rand(3, 2), torch.rand(3, 5))

        with self.assertRaisesRegex(
            ValueError,
            "The first dimension of `input`, `target` and `sample_weight` should be the same size, "
            r"got shapes torch.Size\(\[3, 2\]\), torch.Size\(\[3, 2\]\) and torch.Size\(\[5\]\).",
        ):
            my_mean_squared_error(
                torch.rand(3, 2), torch.rand(3, 2), sample_weight=torch.rand(5)
            )

        with self.assertRaisesRegex(
            ValueError,
            "The `multioutput` must be either `raw_values` or `uniform_average`, "
            r"got multioutput=gaussian.",
        ):
            my_mean_squared_error(
                torch.rand(3, 2), torch.rand(3, 2), multioutput="gaussian"
            )
