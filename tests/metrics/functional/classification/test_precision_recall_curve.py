# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest

import torch
from torcheval.metrics.functional import (
    binary_precision_recall_curve,
    multiclass_precision_recall_curve,
    multilabel_precision_recall_curve,
)


class TestBinaryPrecisionRecallCurve(unittest.TestCase):
    def test_binary_precision_recall_curve_base(self) -> None:
        input = torch.tensor([0.1, 0.4, 0.6, 0.6, 0.6, 0.35, 0.8])
        target = torch.tensor([0, 0, 1, 1, 1, 1, 1])
        my_compute_result = binary_precision_recall_curve(input, target)
        expected_result = (
            torch.tensor([0.71428571, 0.83333333, 0.8, 1.0, 1.0, 1.0]),
            torch.tensor([1.0, 1.0, 0.8, 0.8, 0.2, 0.0]),
            torch.tensor([0.1, 0.35, 0.4, 0.6, 0.8]),
        )
        torch.testing.assert_close(
            my_compute_result,
            expected_result,
            equal_nan=True,
            atol=1e-8,
            rtol=1e-5,
        )

        input = torch.tensor([0.1, 0.4, 0.6, 0.6, 0.6, 0.35, 0.8])
        target = torch.tensor([0, 0, 0, 0, 0, 0, 0])
        my_compute_result = binary_precision_recall_curve(input, target)
        expected_result = (
            torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 0.0]),
            torch.tensor([0.1, 0.35, 0.4, 0.6, 0.8]),
        )
        torch.testing.assert_close(
            my_compute_result,
            expected_result,
            equal_nan=True,
            atol=1e-8,
            rtol=1e-5,
        )

    def test_binary_precision_recall_curve_invalid_input(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "input should be a one-dimensional tensor, "
            r"got shape torch.Size\(\[3, 2\]\).",
        ):
            binary_precision_recall_curve(torch.rand(3, 2), torch.rand(3))

        with self.assertRaisesRegex(
            ValueError,
            "target should be a one-dimensional tensor, "
            r"got shape torch.Size\(\[3, 2\]\).",
        ):
            binary_precision_recall_curve(torch.rand(3), torch.rand(3, 2))

        with self.assertRaisesRegex(
            ValueError,
            "The `input` and `target` should have the same shape, "
            r"got shapes torch.Size\(\[4\]\) and torch.Size\(\[3\]\).",
        ):
            binary_precision_recall_curve(torch.rand(4), torch.rand(3))


class TestMulticlassPrecisionRecallCurve(unittest.TestCase):
    def test_multiclass_precision_recall_curve_base(self) -> None:
        input = torch.tensor(
            [
                [0.1, 0.2, 0.1],
                [0.4, 0.2, 0.1],
                [0.6, 0.1, 0.2],
                [0.4, 0.2, 0.3],
                [0.6, 0.2, 0.4],
            ]
        )
        target = torch.tensor([0, 1, 2, 1, 0])
        my_compute_result = multiclass_precision_recall_curve(
            input, target, num_classes=3
        )
        expected_result = (
            [
                torch.tensor([0.4, 0.25, 0.5, 1.0]),
                torch.tensor([0.4, 0.5, 1.0]),
                torch.tensor([0.2, 0.33333333, 0.0, 0.0, 1.0]),
            ],
            [
                torch.tensor([1.0, 0.5, 0.5, 0.0]),
                torch.tensor([1.0, 1.0, 0.0]),
                torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0]),
            ],
            [
                torch.tensor([0.1, 0.4, 0.6]),
                torch.tensor([0.1, 0.2]),
                torch.tensor([0.1, 0.2, 0.3, 0.4]),
            ],
        )
        torch.testing.assert_close(
            my_compute_result,
            expected_result,
            equal_nan=True,
            atol=1e-8,
            rtol=1e-5,
        )

        input = torch.tensor(
            [
                [0.1, 0.2, 0.1, 0.5],
                [0.4, 0.3, 0.1, 0.9],
                [0.7, 0.1, 0.2, 0.1],
                [0.4, 0.2, 0.9, 0.2],
                [0.6, 0.8, 0.4, 0.6],
            ]
        )
        target = torch.tensor([3, 1, 2, 1, 0])
        my_compute_result = multiclass_precision_recall_curve(
            input, target, num_classes=4
        )
        expected_result = (
            [
                torch.tensor([0.2, 0.25, 0.5, 0.0, 1.0]),
                torch.tensor([0.4, 0.5, 0.5, 0.0, 1.0]),
                torch.tensor([0.2, 0.33333333, 0.0, 0.0, 1.0]),
                torch.tensor([0.2, 0.25, 0.33333333, 0.0, 0.0, 1.0]),
            ],
            [
                torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0]),
                torch.tensor([1.0, 1.0, 0.5, 0.0, 0.0]),
                torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0]),
                torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]),
            ],
            [
                torch.tensor([0.1, 0.4, 0.6, 0.7]),
                torch.tensor([0.1, 0.2, 0.3, 0.8]),
                torch.tensor([0.1, 0.2, 0.4, 0.9]),
                torch.tensor([0.1, 0.2, 0.5, 0.6, 0.9]),
            ],
        )
        torch.testing.assert_close(
            my_compute_result,
            expected_result,
            equal_nan=True,
            atol=1e-8,
            rtol=1e-5,
        )

    def test_multiclass_precision_recall_curve_label_not_exist(self) -> None:
        input = torch.tensor(
            [
                [0.1, 0.2, 0.1, 0.5],
                [0.4, 0.3, 0.1, 0.9],
                [0.7, 0.1, 0.2, 0.1],
                [0.4, 0.2, 0.9, 0.2],
                [0.6, 0.8, 0.4, 0.6],
            ]
        )
        target = torch.tensor([2, 1, 2, 1, 0])
        my_compute_result = multiclass_precision_recall_curve(
            input, target, num_classes=4
        )
        expected_result = (
            [
                torch.tensor([0.2, 0.25, 0.5, 0.0, 1.0]),
                torch.tensor([0.4, 0.5, 0.5, 0.0, 1.0]),
                torch.tensor([0.4, 0.33333333, 0.0, 0.0, 1.0]),
                torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            ],
            [
                torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0]),
                torch.tensor([1.0, 1.0, 0.5, 0.0, 0.0]),
                torch.tensor([1.0, 0.5, 0.0, 0.0, 0.0]),
                torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 0.0]),
            ],
            [
                torch.tensor([0.1, 0.4, 0.6, 0.7]),
                torch.tensor([0.1, 0.2, 0.3, 0.8]),
                torch.tensor([0.1, 0.2, 0.4, 0.9]),
                torch.tensor([0.1, 0.2, 0.5, 0.6, 0.9]),
            ],
        )
        torch.testing.assert_close(
            my_compute_result,
            expected_result,
            equal_nan=True,
            atol=1e-8,
            rtol=1e-5,
        )

    def test_multiclass_precision_recall_curve_invalid_input(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "The `input` and `target` should have the same first dimension, "
            r"got shapes torch.Size\(\[4, 2\]\) and torch.Size\(\[3\]\).",
        ):
            multiclass_precision_recall_curve(
                torch.rand(4, 2), torch.rand(3), num_classes=2
            )

        with self.assertRaisesRegex(
            ValueError,
            "target should be a one-dimensional tensor, "
            r"got shape torch.Size\(\[3, 2\]\).",
        ):
            multiclass_precision_recall_curve(
                torch.rand(3, 2), torch.rand(3, 2), num_classes=2
            )

        with self.assertRaisesRegex(
            ValueError,
            r"input should have shape of \(num_sample, num_classes\), "
            r"got torch.Size\(\[3, 4\]\) and num_classes=2.",
        ):
            multiclass_precision_recall_curve(
                torch.rand(3, 4), torch.rand(3), num_classes=2
            )


class TestMultilabelPrecisionRecallCurve(unittest.TestCase):
    def test_multilabel_precision_recall_curve_base(self) -> None:
        input = torch.tensor(
            [
                [0.75, 0.05, 0.35],
                [0.45, 0.75, 0.05],
                [0.05, 0.55, 0.75],
                [0.05, 0.65, 0.05],
            ]
        )
        target = torch.tensor([[1, 0, 1], [0, 0, 0], [0, 1, 1], [1, 1, 1]])
        my_compute_result = multilabel_precision_recall_curve(
            input, target, num_labels=3
        )
        expected_result = (
            [
                torch.tensor([0.5, 0.5, 1.0, 1.0]),
                torch.tensor([0.5, 0.66666667, 0.5, 0.0, 1.0]),
                torch.tensor([0.75, 1.0, 1.0, 1.0]),
            ],
            [
                torch.tensor([1.0, 0.5, 0.5, 0.0]),
                torch.tensor([1.0, 1.0, 0.5, 0.0, 0.0]),
                torch.tensor([1.0, 0.66666667, 0.33333333, 0.0]),
            ],
            [
                torch.tensor([0.05, 0.45, 0.75]),
                torch.tensor([0.05, 0.55, 0.65, 0.75]),
                torch.tensor([0.05, 0.35, 0.75]),
            ],
        )
        torch.testing.assert_close(
            my_compute_result, expected_result, equal_nan=True, atol=1e-8, rtol=1e-5
        )

    def test_multilabel_precision_recall_curve_label_not_exist(self) -> None:
        input = torch.tensor(
            [
                [0.75, 0.05, 0.35],
                [0.45, 0.75, 0.05],
                [0.05, 0.55, 0.75],
                [0.05, 0.65, 0.05],
            ]
        )
        target = torch.tensor([[1, 0, 0], [0, 0, 0], [0, 1, 0], [1, 1, 0]])
        my_compute_result = multilabel_precision_recall_curve(
            input, target, num_labels=3
        )
        expected_result = (
            [
                torch.tensor([0.5, 0.5, 1.0, 1.0]),
                torch.tensor([0.5, 0.66666667, 0.5, 0.0, 1.0]),
                torch.tensor([0.0, 0.0, 0.0, 1.0]),
            ],
            [
                torch.tensor([1.0, 0.5, 0.5, 0.0]),
                torch.tensor([1.0, 1.0, 0.5, 0.0, 0.0]),
                torch.tensor([1.0, 1.0, 1.0, 0.0]),
            ],
            [
                torch.tensor([0.05, 0.45, 0.75]),
                torch.tensor([0.05, 0.55, 0.65, 0.75]),
                torch.tensor([0.05, 0.35, 0.75]),
            ],
        )
        print(my_compute_result)
        torch.testing.assert_close(
            my_compute_result, expected_result, equal_nan=True, atol=1e-8, rtol=1e-5
        )

    def test_multilabel_precision_recall_curve_num_labels_None(self) -> None:
        input = torch.tensor(
            [
                [0.75, 0.05, 0.35],
                [0.45, 0.75, 0.05],
                [0.05, 0.55, 0.75],
                [0.05, 0.65, 0.05],
            ]
        )
        target = torch.tensor([[1, 0, 0], [0, 0, 0], [0, 1, 0], [1, 1, 0]])
        my_compute_result = multilabel_precision_recall_curve(input, target)
        expected_result = (
            [
                torch.tensor([0.5, 0.5, 1.0, 1.0]),
                torch.tensor([0.5, 0.66666667, 0.5, 0.0, 1.0]),
                torch.tensor([0.0, 0.0, 0.0, 1.0]),
            ],
            [
                torch.tensor([1.0, 0.5, 0.5, 0.0]),
                torch.tensor([1.0, 1.0, 0.5, 0.0, 0.0]),
                torch.tensor([1.0, 1.0, 1.0, 0.0]),
            ],
            [
                torch.tensor([0.05, 0.45, 0.75]),
                torch.tensor([0.05, 0.55, 0.65, 0.75]),
                torch.tensor([0.05, 0.35, 0.75]),
            ],
        )
        print(my_compute_result)
        torch.testing.assert_close(
            my_compute_result, expected_result, equal_nan=True, atol=1e-8, rtol=1e-5
        )

    def test_multilabel_precision_recall_curve_invalid_input(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            r"input should be a two-dimensional tensor, got shape torch.Size\(\[4\]\).",
        ):
            multilabel_precision_recall_curve(
                torch.rand(4), torch.randint(high=2, size=(4,)), num_labels=4
            )

        with self.assertRaisesRegex(
            ValueError,
            "Expected both input.shape and target.shape to have the same shape"
            r" but got torch.Size\(\[4, 2\]\) and torch.Size\(\[4, 3\]\).",
        ):
            multilabel_precision_recall_curve(
                torch.rand(4, 2), torch.randint(high=2, size=(4, 3)), num_labels=3
            )

        with self.assertRaisesRegex(
            ValueError,
            r"input should be a two-dimensional tensor, got shape torch.Size\(\[4, 3, 2\]\).",
        ):
            multilabel_precision_recall_curve(
                torch.rand(4, 3, 2), torch.randint(high=2, size=(4, 3, 2)), num_labels=4
            )

        with self.assertRaisesRegex(
            ValueError,
            r"input should have shape of \(num_sample, num_labels\), "
            r"got torch.Size\(\[4, 2\]\) and num_labels=3.",
        ):
            multilabel_precision_recall_curve(
                torch.rand(4, 2), torch.randint(high=2, size=(4, 2)), num_labels=3
            )
