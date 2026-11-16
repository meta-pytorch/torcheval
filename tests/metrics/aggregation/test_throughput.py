# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import random

from torcheval.metrics import Throughput
from torcheval.utils.test_utils.metric_class_tester import (
    MetricClassTester,
    NUM_PROCESSES,
    NUM_TOTAL_UPDATES,
)


class TestThroughput(MetricClassTester):
    def _test_throughput_class_with_input(
        self,
        num_processed: list[int],
        elapsed_time_sec: list[float],
    ) -> None:
        num_individual_update = NUM_TOTAL_UPDATES // NUM_PROCESSES
        expected_num_total = sum(num_processed)
        max_elapsed_time_sec = max(
            [
                sum(
                    elapsed_time_sec[
                        i * num_individual_update : (i + 1) * num_individual_update
                    ]
                )
                for i in range(NUM_PROCESSES)
            ]
        )
        total_elapsed_time_sec = sum(elapsed_time_sec)

        expected_compute_result = (1.0 * expected_num_total) / total_elapsed_time_sec
        expected_merge_and_compute_result = (
            1.0 * expected_num_total
        ) / max_elapsed_time_sec
        self.run_class_implementation_tests(
            metric=Throughput(),
            state_names={"num_total", "elapsed_time_sec"},
            update_kwargs={
                "num_processed": num_processed,
                "elapsed_time_sec": elapsed_time_sec,
            },
            compute_result=expected_compute_result,
            merge_and_compute_result=expected_merge_and_compute_result,
        )

    def test_throughput_class_base(self) -> None:
        num_processed = [random.randint(0, 40) for _ in range(NUM_TOTAL_UPDATES)]
        elapsed_time_sec = [random.uniform(0.1, 5.0) for _ in range(NUM_TOTAL_UPDATES)]
        self._test_throughput_class_with_input(num_processed, elapsed_time_sec)

    def test_throughput_class_update_input_invalid_num_processed(self) -> None:
        metric = Throughput()
        with self.assertRaisesRegex(
            ValueError,
            r"Expected num_processed to be a non-negative number, but received",
        ):
            metric.update(-1, 1.0)

    def test_throughput_class_update_input_invalid_elapsed_time_sec(self) -> None:
        metric = Throughput()
        with self.assertRaisesRegex(
            ValueError,
            r"Expected elapsed_time_sec to be a positive number, but received",
        ):
            metric.update(42, -5.1)
        with self.assertRaisesRegex(
            ValueError,
            r"Expected elapsed_time_sec to be a positive number, but received",
        ):
            metric.update(42, 0.0)

    def test_throughput_class_compute_without_update(self) -> None:
        metric = Throughput()
        self.assertEqual(metric.compute(), 0.0)
