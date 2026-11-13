# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import torch
from torcheval.metrics.text import WordInformationPreserved
from torcheval.utils.test_utils.metric_class_tester import MetricClassTester


class TestWordInformationPreserved(MetricClassTester):
    def test_word_information_preserved_accumulates_correct_counts(self) -> None:
        metric = WordInformationPreserved()
        metric.update("sat the cat", "the cat sat")
        torch.testing.assert_close(
            metric.state_dict()["correct_total"],
            torch.tensor(2.0, dtype=torch.float64),
        )
        torch.testing.assert_close(
            metric.compute(), torch.tensor(4 / 9, dtype=torch.float64)
        )

        metric.update("hello", "hello world")
        torch.testing.assert_close(
            metric.state_dict()["correct_total"],
            torch.tensor(3.0, dtype=torch.float64),
        )
        torch.testing.assert_close(
            metric.compute(), torch.tensor(9 / 20, dtype=torch.float64)
        )

        first = WordInformationPreserved().update("sat the cat", "the cat sat")
        second = WordInformationPreserved().update("hello", "hello world")
        first.merge_state([second])
        torch.testing.assert_close(
            first.state_dict()["correct_total"],
            torch.tensor(3.0, dtype=torch.float64),
        )
        torch.testing.assert_close(first.compute(), metric.compute())

    def test_word_information_preserved_with_valid_input(self) -> None:
        self.run_class_implementation_tests(
            metric=WordInformationPreserved(),
            state_names={"correct_total", "input_total", "target_total"},
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
            compute_result=torch.tensor(0.3, dtype=torch.float64),
            num_total_updates=4,
        )

    def test_word_information_preserved_with_invalid_input(self) -> None:
        metric = WordInformationPreserved()
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
