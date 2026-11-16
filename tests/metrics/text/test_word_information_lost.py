# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch
from torcheval.metrics.text import WordInformationLost
from torcheval.utils.test_utils.metric_class_tester import MetricClassTester


class TestWordInformationLost(MetricClassTester):
    def test_word_information_lost(self) -> None:
        self.run_class_implementation_tests(
            metric=WordInformationLost(),
            state_names={"correct_total", "target_total", "preds_total"},
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
            compute_result=torch.tensor(0.7, dtype=torch.float64),
            num_total_updates=4,
        )

    def test_word_information_lost_with_invalid_input(self) -> None:
        metric = WordInformationLost()

        with self.assertRaisesRegex(
            AssertionError,
            "Arguments must contain the same number of strings.",
        ):
            metric.update(
                ["hello metaverse", "welcome to meta"],
                [
                    "welcome to meta",
                    "this is the prediction",
                    "there is an other sample",
                ],
            )
