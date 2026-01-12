#!/usr/bin/env fbpython
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from collections import Counter
from math import exp

import torch
from torcheval.metrics.functional.text.bleu import (
    _bleu_score_update,
    _calc_brevity_penalty,
    _get_ngrams,
    bleu_score,
)


class TestBleu(unittest.TestCase):
    def test_brevity_penalty_r_leq_c(self) -> None:
        input_len = torch.tensor(6)
        target_len = torch.tensor(7)
        bp = _calc_brevity_penalty(input_len, target_len)
        self.assertAlmostEqual(exp(-1 / 6), bp.item())

    def test_brevity_penalty_r_ge_c(self) -> None:
        input_len = torch.tensor(8)
        target_len = torch.tensor(7)
        bp = _calc_brevity_penalty(input_len, target_len)
        self.assertEqual(1, bp.item())

    def test_get_1grams(self) -> None:
        sentence = "a squirrel is eating a nut"
        sentence_tokenized = sentence.split()
        n = 1
        actual = _get_ngrams(sentence_tokenized, n)
        expected = Counter(
            {("a",): 2, ("squirrel",): 1, ("is",): 1, ("eating",): 1, ("nut",): 1}
        )
        self.assertEqual(actual, expected)

    def test_get_3grams(self) -> None:
        sentence = "a squirrel is eating a nut"
        sentence_tokenized = sentence.split()
        n = 3
        actual = _get_ngrams(sentence_tokenized, n)
        expected = Counter(
            {
                ("a",): 2,
                ("squirrel",): 1,
                ("is",): 1,
                ("eating",): 1,
                ("nut",): 1,
                ("a", "squirrel"): 1,
                ("squirrel", "is"): 1,
                ("is", "eating"): 1,
                ("eating", "a"): 1,
                ("a", "nut"): 1,
                ("a", "squirrel", "is"): 1,
                ("squirrel", "is", "eating"): 1,
                ("is", "eating", "a"): 1,
                ("eating", "a", "nut"): 1,
            },
        )
        self.assertEqual(actual, expected)

    def test_get_ngrams_invalid(self) -> None:
        sentence = "a squirrel is eating a nut"
        sentence_tokenized = sentence.split()
        n = 5
        with self.assertRaisesRegex(ValueError, "n_gram should be 1, 2, 3, or 4"):
            _get_ngrams(sentence_tokenized, n)

    def test_invalid_input_target(self) -> None:
        candidates = ["the squirrel is eating the nut", "the cat is on the mat"]
        references = [
            ["a squirrel is eating a nut", "the squirrel is eating a tasty nut"]
        ]
        n = 4
        with self.assertRaisesRegex(
            ValueError,
            "Input and target corpus should have same sizes",
        ):
            _bleu_score_update(candidates, references, n)

    def test_invalid_input_target_2(self) -> None:
        candidates = "the squirrel is eating the nut"
        references = [
            "a squirrel is eating a nut",
            "the squirrel is eating a tasty nut",
        ]
        n = 4
        with self.assertRaisesRegex(
            ValueError,
            "Input and target corpus should have same sizes",
        ):
            _bleu_score_update(candidates, references, n)

    def test_bleu(self) -> None:
        candidates = ["the squirrel is eating the nut", "the cat is on the mat"]
        references = [
            ["a squirrel is eating a nut", "the squirrel is eating a tasty nut"],
            ["there is a cat on the mat", "a cat is on the mat"],
        ]
        n = 4
        score = bleu_score(candidates, references, n)
        self.assertAlmostEqual(score.item(), 0.65341892)

    def test_bleu_single_candidate(self) -> None:
        candidate = "the squirrel is eating the nut"
        references = [
            [
                "a squirrel is eating a nut",
                "the squirrel is eating a tasty nut",
            ]
        ]
        n_gram = 3
        score = bleu_score(candidate, references, n_gram)
        self.assertAlmostEqual(score.item(), 0.62996054)

    def test_bleu_single_reference(self) -> None:
        candidates = ["the squirrel is eating the nut", "the cat is on the mat"]
        references = [
            "a squirrel is eating a nut",
            ["there is a cat on the mat", "a cat is on the mat"],
        ]
        n_gram = 3
        score = bleu_score(candidates, references, n_gram)
        self.assertAlmostEqual(score.item(), 0.60822022)

    def test_bleu_with_w(self) -> None:
        candidate = ["the squirrel is eating the nut"]
        reference = [
            ["a squirrel is eating a nut", "the squirrel is eating a tasty nut"]
        ]
        n_gram = 4
        weights = torch.tensor([0.1, 0.2, 0.3, 0.4])
        score = bleu_score(candidate, reference, n_gram, weights)
        self.assertAlmostEqual(score.item(), 0.46403915)

    def test_bleu_incorrect_w_specified(self) -> None:
        candidate = ["the squirrel is eating the nut"]
        reference = [
            ["a squirrel is eating a nut", "the squirrel is eating a tasty nut"]
        ]
        n_gram = 4
        weights = torch.tensor([0.1, 0.2, 0.3])
        with self.assertRaisesRegex(
            ValueError, "the length of weights should equal n_gram"
        ):
            bleu_score(candidate, reference, n_gram, weights)

    def test_bleu_input_too_short(self) -> None:
        candidate = ["small sentence"]
        reference = ["this is a really small sentence"]
        n_gram = 4
        with self.assertRaisesRegex(
            ValueError, "the input is too short to find all n-gram matches"
        ):
            bleu_score(candidate, reference, n_gram)
