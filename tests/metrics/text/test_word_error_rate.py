# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import torch
from torcheval.metrics.text import WordErrorRate
from torcheval.utils.test_utils.metric_class_tester import MetricClassTester


class TestWordErrorRate(MetricClassTester):
    def test_word_error_rate_with_valid_input(self) -> None:
        self.run_class_implementation_tests(
            metric=WordErrorRate(),
            state_names={"errors", "total"},
            update_kwargs={
                "input": [
                    ["hello world", "welcome to the facebook"],
                    ["hello world", "welcome to the facebook"],
                    ["hello world", "welcome to the facebook"],
                    ["hello world", "welcome to the facebook"],
                ],
                "target": [
                    ["hello metaverse", "welcome to meta"],
                    ["hello metaverse", "welcome to meta"],
                    ["hello metaverse", "welcome to meta"],
                    ["hello metaverse", "welcome to meta"],
                ],
            },
            compute_result=torch.tensor(0.6, dtype=torch.float64),
            num_total_updates=4,
        )

    def test_word_error_rate_with_invalid_input(self) -> None:
        metric = WordErrorRate()
        with self.assertRaisesRegex(
            ValueError, "input and target should have the same type"
        ):
            metric.update(["hello metaverse", "welcome to meta"], "hello world")

        with self.assertRaisesRegex(
            ValueError, "input and target lists should have the same length"
        ):
            metric.update(
                ["hello metaverse", "welcome to meta"],
                [
                    "welcome to meta",
                    "this is the prediction",
                    "there is an other sample",
                ],
            )
