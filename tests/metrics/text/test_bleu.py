#!/usr/bin/env fbpython
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch
from torcheval.metrics.text import BLEUScore
from torcheval.utils.test_utils.metric_class_tester import MetricClassTester


class TestBleu(MetricClassTester):
    def test_bleu_invalid_update(self) -> None:
        candidates = ["the squirrel is eating the nut"]
        references = [
            ["a squirrel is eating a nut", "the squirrel is eating a tasty nut"],
            ["there is a cat on the mat", "a cat is on the mat"],
        ]
        metric = BLEUScore(n_gram=4)
        with self.assertRaisesRegex(
            ValueError,
            "Input and target corpus should have same sizes",
        ):
            metric.update(candidates, references)

    def test_bleu_invalid_w(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "the length of weights should equal n_gram",
        ):
            BLEUScore(n_gram=4, weights=torch.tensor([0.3, 0.3, 0.4]))

    def test_bleu_invalid_n(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "n_gram should be 1, 2, 3, or 4",
        ):
            BLEUScore(n_gram=5)

    def test_bleu_single_example(self) -> None:
        candidate = ["the squirrel is eating the nut"]
        reference = [
            ["a squirrel is eating a nut", "the squirrel is eating a tasty nut"]
        ]
        metric = BLEUScore(n_gram=4)
        metric.update(candidate, reference)
        val = metric.compute()
        self.assertAlmostEqual(val.item(), 0.53728497)

    def test_bleu_multiple_updates(self) -> None:
        candidates = [["the squirrel is eating the nut"], ["the cat is on the mat"]]
        references = [
            [["a squirrel is eating a nut", "the squirrel is eating a tasty nut"]],
            [["there is a cat on the mat", "a cat is on the mat"]],
        ]
        self.run_class_implementation_tests(
            metric=BLEUScore(n_gram=4),
            state_names={
                "input_len",
                "target_len",
                "matches_by_order",
                "possible_matches_by_order",
            },
            update_kwargs={
                "input": candidates,
                "target": references,
            },
            compute_result=torch.tensor(0.65341892, dtype=torch.float64),
            num_total_updates=2,
            num_processes=2,
        )

    def test_bleu_multiple_examples_per_update(self) -> None:
        candidates = [
            ["the squirrel is eating the nut", "the cat is on the mat"],
            ["i like ice cream and apple pie"],
        ]
        references = [
            [
                ["a squirrel is eating a nut", "the squirrel is eating a tasty nut"],
                ["there is a cat on the mat", "a cat is on the mat"],
            ],
            [
                [
                    "i like apple pie with ice cream on top",
                    "i like ice cream with my apple pie",
                    "i enjoy my apple pie with ice cream",
                ]
            ],
        ]
        self.run_class_implementation_tests(
            metric=BLEUScore(n_gram=4),
            state_names={
                "input_len",
                "target_len",
                "matches_by_order",
                "possible_matches_by_order",
            },
            update_kwargs={
                "input": candidates,
                "target": references,
            },
            compute_result=torch.tensor(0.56377503, dtype=torch.float64),
            num_total_updates=2,
            num_processes=2,
        )

    def test_bleu_brevity(self) -> None:
        candidates = [["the squirrel is eating the nut"], ["the cat is on mat"]]
        references = [
            [
                [
                    "a squirrel is eating a nut",
                    "the squirrel is eating a tasty nut",
                    "hi",
                ]
            ],
            [["there is a cat on the mat", "a cat is on the mat"]],
        ]
        self.run_class_implementation_tests(
            metric=BLEUScore(n_gram=4),
            state_names={
                "input_len",
                "target_len",
                "matches_by_order",
                "possible_matches_by_order",
            },
            update_kwargs={
                "input": candidates,
                "target": references,
            },
            compute_result=torch.tensor(0.41650065, dtype=torch.float64),
            num_total_updates=2,
            num_processes=2,
        )
