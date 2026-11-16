# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest

import torch
from torcheval.metrics.functional import weighted_calibration


class TestWeightedCalibration(unittest.TestCase):
    def test_weighted_calibration_with_valid_input(self) -> None:
        torch.testing.assert_close(
            weighted_calibration(
                torch.tensor([0.8, 0.4, 0.3, 0.8, 0.7, 0.6]),
                torch.tensor([1, 1, 0, 0, 1, 0]),
            ),
            torch.tensor(1.2000),
        )

        torch.testing.assert_close(
            weighted_calibration(
                torch.tensor([0.8, 0.4, 0.3, 0.8, 0.7, 0.6]),
                torch.tensor([1, 1, 0, 0, 1, 0]),
                torch.tensor([0.5, 1.0, 2.0, 0.4, 1.3, 0.9]),
            ),
            torch.tensor(1.1321428185),
        )

        torch.testing.assert_close(
            weighted_calibration(
                torch.tensor([[0.8, 0.4], [0.8, 0.7]]),
                torch.tensor([[1, 1], [0, 1]]),
                num_tasks=2,
            ),
            torch.tensor([0.6000, 1.5000]),
        )

    def test_weighted_calibration_with_invalid_input(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            r"Weight must be either a float value or a tensor that matches the input tensor size.",
        ):
            weighted_calibration(
                torch.tensor([0.8, 0.4, 0.8, 0.7]),
                torch.tensor([1, 1, 0, 1]),
                torch.tensor([1, 1.5]),
            )

        with self.assertRaisesRegex(
            ValueError,
            r"is different from `target` shape",
        ):
            weighted_calibration(
                torch.tensor([0.8, 0.4, 0.8, 0.7]),
                torch.tensor([[1, 1, 0], [0, 1, 1]]),
            )

        with self.assertRaisesRegex(
            ValueError,
            r"`num_tasks = 1`, `input` is expected to be one-dimensional tensor,",
        ):
            weighted_calibration(
                torch.tensor([[0.8, 0.4], [0.8, 0.7]]),
                torch.tensor([[1, 1], [0, 1]]),
            )
        with self.assertRaisesRegex(
            ValueError,
            r"`num_tasks = 2`, `input`'s shape is expected to be",
        ):
            weighted_calibration(
                torch.tensor([0.8, 0.4, 0.8, 0.7]),
                torch.tensor([1, 0, 0, 1]),
                num_tasks=2,
            )
