# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import torch
from torcheval.metrics.text import WordInformationLost
from torcheval.utils.test_utils.metric_class_tester import MetricClassTester


class TestWordInformationLost(MetricClassTester):
    def test_word_information_lost_accumulates_correct_counts(self) -> None:
        metric = WordInformationLost()
        metric.update("sat the cat", "the cat sat")
        torch.testing.assert_close(
            metric.state_dict()["correct_total"],
            torch.tensor(-2.0, dtype=torch.float64),
        )
        torch.testing.assert_close(
            metric.compute(), torch.tensor(5 / 9, dtype=torch.float64)
        )

        metric.update("hello", "hello world")
        torch.testing.assert_close(
            metric.state_dict()["correct_total"],
            torch.tensor(-3.0, dtype=torch.float64),
        )
        torch.testing.assert_close(
            metric.compute(), torch.tensor(11 / 20, dtype=torch.float64)
        )

        first = WordInformationLost().update("sat the cat", "the cat sat")
        second = WordInformationLost().update("hello", "hello world")
        first.merge_state([second])
        torch.testing.assert_close(
            first.state_dict()["correct_total"],
            torch.tensor(-3.0, dtype=torch.float64),
        )
        torch.testing.assert_close(first.compute(), metric.compute())

    def test_word_information_lost_loads_legacy_state(self) -> None:
        metric = WordInformationLost()
        # Legacy state after update("hello", "hello world").
        metric.load_state_dict(
            {
                "correct_total": torch.tensor(-1.0, dtype=torch.float64),
                "target_total": torch.tensor(2.0, dtype=torch.float64),
                "preds_total": torch.tensor(1.0, dtype=torch.float64),
            }
        )
        torch.testing.assert_close(
            metric.compute(), torch.tensor(1 / 2, dtype=torch.float64)
        )

        metric.update("sat the cat", "the cat sat")
        torch.testing.assert_close(
            metric.state_dict()["correct_total"],
            torch.tensor(-3.0, dtype=torch.float64),
        )
        torch.testing.assert_close(
            metric.compute(), torch.tensor(11 / 20, dtype=torch.float64)
        )

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
