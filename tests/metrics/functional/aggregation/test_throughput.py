# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import random
import unittest

import torch
from torcheval.metrics.functional import throughput
from torcheval.utils.test_utils.metric_class_tester import NUM_PROCESSES


class TestThroughput(unittest.TestCase):
    def _test_throughput_with_input(
        self,
        num_processed: int,
        elapsed_time_sec: float,
    ) -> None:
        torch.testing.assert_close(
            throughput(num_processed, elapsed_time_sec),
            torch.tensor(num_processed / elapsed_time_sec),
            equal_nan=True,
            atol=1e-8,
            rtol=1e-5,
        )

    def test_throughput_base(self) -> None:
        num_processed = NUM_PROCESSES
        elapsed_time_sec = random.random() * 20
        self._test_throughput_with_input(num_processed, elapsed_time_sec)

    def test_throughput_update_input_invalid_num_processed(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            r"Expected num_processed to be a non-negative number, but received",
        ):
            throughput(-1, 1.0)

    def test_throughput_update_input_invalid_elapsed_time_sec(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            r"Expected elapsed_time_sec to be a positive number, but received",
        ):
            throughput(42, -5.1)
        with self.assertRaisesRegex(
            ValueError,
            r"Expected elapsed_time_sec to be a positive number, but received",
        ):
            throughput(42, 0.0)
