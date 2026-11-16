# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest

import torch
from torcheval.metrics.functional import (
    binary_recall_at_fixed_precision,
    multilabel_recall_at_fixed_precision,
)


class TestBinaryRecallAtFixedPrecision(unittest.TestCase):
    def test_binary_recall_at_fixed_precision_base(self) -> None:
        input = torch.tensor([0.1, 0.4, 0.6, 0.6, 0.6, 0.35, 0.8])
        target = torch.tensor([0, 0, 1, 1, 1, 1, 1])
        my_compute_result = binary_recall_at_fixed_precision(
            input, target, min_precision=0.5
        )
        expected_result = (torch.tensor(1.0), torch.tensor(0.35))
        torch.testing.assert_close(
            my_compute_result,
            expected_result,
            equal_nan=True,
            atol=1e-8,
            rtol=1e-5,
        )

        input = torch.tensor([0.1, 0.4, 0.6, 0.6, 0.6, 0.35, 0.8])
        target = torch.tensor([0, 0, 0, 0, 0, 0, 0])
        my_compute_result = binary_recall_at_fixed_precision(
            input, target, min_precision=0.5
        )
        expected_result = (torch.tensor(0.0), torch.tensor(1.0))
        torch.testing.assert_close(
            my_compute_result, expected_result, equal_nan=True, atol=1e-8, rtol=1e-5
        )

    def test_binary_recall_at_fixed_precision_invalid_input(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "input should be a one-dimensional tensor, "
            r"got shape torch.Size\(\[3, 2\]\).",
        ):
            binary_recall_at_fixed_precision(
                torch.rand(3, 2), torch.rand(3), min_precision=0.5
            )

        with self.assertRaisesRegex(
            ValueError,
            "target should be a one-dimensional tensor, "
            r"got shape torch.Size\(\[3, 2\]\).",
        ):
            binary_recall_at_fixed_precision(
                torch.rand(3), torch.rand(3, 2), min_precision=0.5
            )

        with self.assertRaisesRegex(
            ValueError,
            "The `input` and `target` should have the same shape, "
            r"got shapes torch.Size\(\[4\]\) and torch.Size\(\[3\]\).",
        ):
            binary_recall_at_fixed_precision(
                torch.rand(4), torch.rand(3), min_precision=0.5
            )

        with self.assertRaisesRegex(
            ValueError,
            r"Expected min_precision to be a float in the \[0, 1\] range"
            r" but got 1.1.",
        ):
            binary_recall_at_fixed_precision(
                torch.rand(3), torch.rand(3), min_precision=1.1
            )


class TestMultilabelRecallAtFixedPrecision(unittest.TestCase):
    def test_multilabel_recall_at_fixed_precision_base(self) -> None:
        input = torch.tensor(
            [
                [0.75, 0.05, 0.35],
                [0.45, 0.75, 0.05],
                [0.05, 0.55, 0.75],
                [0.05, 0.65, 0.05],
            ]
        )
        target = torch.tensor([[1, 0, 1], [0, 0, 0], [0, 1, 1], [1, 1, 1]])
        my_compute_result = multilabel_recall_at_fixed_precision(
            input, target, num_labels=3, min_precision=0.5
        )
        expected_result = (
            [torch.tensor(1.0), torch.tensor(1.0), torch.tensor(1.0)],
            [torch.tensor(0.05), torch.tensor(0.55), torch.tensor(0.05)],
        )
        torch.testing.assert_close(
            my_compute_result, expected_result, equal_nan=True, atol=1e-8, rtol=1e-5
        )

    def test_multilabel_recall_at_fixed_precision_label_not_exist(self) -> None:
        input = torch.tensor(
            [
                [0.75, 0.05, 0.35],
                [0.45, 0.75, 0.05],
                [0.05, 0.55, 0.75],
                [0.05, 0.65, 0.05],
            ]
        )
        target = torch.tensor([[1, 0, 0], [0, 0, 0], [0, 1, 0], [1, 1, 0]])
        my_compute_result = multilabel_recall_at_fixed_precision(
            input, target, num_labels=3, min_precision=0.5
        )
        expected_result = (
            [torch.tensor(1.0), torch.tensor(1.0), torch.tensor(0.0)],
            [torch.tensor(0.05), torch.tensor(0.55), torch.tensor(1.0)],
        )
        torch.testing.assert_close(
            my_compute_result, expected_result, equal_nan=True, atol=1e-8, rtol=1e-5
        )

    def test_multilabel_recall_at_fixed_precision_invalid_input(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "Expected both input.shape and target.shape to have the same shape"
            r" but got torch.Size\(\[4, 2\]\) and torch.Size\(\[4, 3\]\).",
        ):
            multilabel_recall_at_fixed_precision(
                torch.rand(4, 2),
                torch.randint(high=2, size=(4, 3)),
                num_labels=3,
                min_precision=0.5,
            )

        with self.assertRaisesRegex(
            ValueError,
            r"input should have shape of \(num_sample, num_labels\), "
            r"got torch.Size\(\[4, 2\]\) and num_labels=3.",
        ):
            multilabel_recall_at_fixed_precision(
                torch.rand(4, 2),
                torch.randint(high=2, size=(4, 2)),
                num_labels=3,
                min_precision=0.5,
            )

        with self.assertRaisesRegex(
            ValueError,
            r"Expected min_precision to be a float in the \[0, 1\] range"
            r" but got 1.1.",
        ):
            multilabel_recall_at_fixed_precision(
                torch.rand(4, 2),
                torch.randint(high=2, size=(4, 2)),
                num_labels=2,
                min_precision=1.1,
            )
