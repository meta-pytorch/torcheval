# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import torch
from sklearn.metrics import confusion_matrix as skcm
from torcheval.metrics.functional import (
    binary_confusion_matrix as bcm,
    multiclass_confusion_matrix as mccm,
)
from torcheval.utils.test_utils.metric_class_tester import BATCH_SIZE


class TestBinaryConfusionMatrix(unittest.TestCase):
    def test_binary_confusion_matrix_float_target(self) -> None:
        input = torch.tensor([0, 0, 0, 1, 1, 0, 1, 1, 1, 1])
        target = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        for dtype in (torch.float32, torch.float64):
            for normalize in (None, "pred", "true", "all"):
                self._test_binary_confusion_matrix_with_input(
                    input, target.to(dtype), normalize=normalize
                )

    def _test_binary_confusion_matrix_with_input(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        normalize: str | None = None,
    ) -> None:
        sklearn_result = torch.tensor(
            skcm(target, input, labels=[0, 1], normalize=normalize)
        ).to(torch.float32)

        torch.testing.assert_close(
            bcm(input, target, normalize=normalize).to(torch.float32),
            sklearn_result,
            equal_nan=True,
            atol=1e-8,
            rtol=1e-5,
        )

    def test_binary_confusion_matrix_base(self) -> None:
        num_classes = 2
        input = torch.randint(high=num_classes, size=(BATCH_SIZE,))
        target = torch.randint(high=num_classes, size=(BATCH_SIZE,))

        self._test_binary_confusion_matrix_with_input(input, target)
        self._test_binary_confusion_matrix_with_input(input, target, normalize="all")
        self._test_binary_confusion_matrix_with_input(input, target, normalize="true")
        self._test_binary_confusion_matrix_with_input(input, target, normalize="pred")

    def test_binary_confusion_matrix_score_thresholding(self) -> None:
        num_classes = 2
        threshold = 0.7
        input = torch.tensor([0.7, 0.6, 0.5, 0.3, 0.9, 0.1, 1.0, 0.95, 0.2])
        input_thresholded = torch.tensor([1, 0, 0, 0, 1, 0, 1, 1, 0])
        target = torch.randint(high=num_classes, size=(9,))

        sklearn_result = torch.tensor(
            skcm(target, input_thresholded, labels=[0, 1])
        ).to(torch.float32)

        my_result = bcm(input, target, threshold=threshold).to(torch.float32)

        # test threshold larger than every prediction gives 0
        torch.testing.assert_close(
            my_result,
            sklearn_result,
            equal_nan=True,
            atol=1e-8,
            rtol=1e-5,
        )

    def test_binary_confusion_matrix_invalid_input(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "input should be a one-dimensional tensor for binary confusion matrix, "
            r"got shape torch.Size\(\[5, 10\]\).",
        ):
            input = torch.randint(high=2, size=(5, 10))
            target = torch.randint(high=2, size=(5, 10))
            bcm(input, target)

        with self.assertRaisesRegex(
            ValueError,
            "target should be a one-dimensional tensor for binary confusion matrix, "
            r"got shape torch.Size\(\[5, 10\]\).",
        ):
            input = torch.randint(high=2, size=(10,))
            target = torch.randint(high=2, size=(5, 10))
            bcm(input, target)

        with self.assertRaisesRegex(
            ValueError,
            "The `input` and `target` should have the same dimensions, "
            r"got shapes torch.Size\(\[11\]\) and torch.Size\(\[10\]\).",
        ):
            input = torch.randint(high=2, size=(11,))
            target = torch.randint(high=2, size=(10,))
            bcm(input, target)

        with self.assertRaisesRegex(
            ValueError, "normalize must be one of 'all', 'pred', 'true', or 'none'."
        ):
            input = torch.randint(high=2, size=(10,))
            target = torch.randint(high=2, size=(10,))
            bcm(input, target, normalize="this is not a valid option")


class TestMultiClassConfusionMatrix(unittest.TestCase):
    def test_multiclass_confusion_matrix_float_target(self) -> None:
        input = torch.tensor([0, 1, 2, 1, 2, 0])
        target = torch.tensor([0, 1, 1, 2, 0, 2])
        for dtype in (torch.float32, torch.float64):
            for normalize in (None, "pred", "true", "all"):
                self._test_multiclass_confusion_matrix_with_input(
                    input, target.to(dtype), num_classes=3, normalize=normalize
                )

    def _test_multiclass_confusion_matrix_with_input(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        num_classes: int,
        normalize: str | None = None,
    ) -> None:
        sklearn_result = torch.tensor(
            skcm(target, input, labels=list(range(num_classes)), normalize=normalize)
        ).to(torch.float32)

        torch.testing.assert_close(
            mccm(input, target, num_classes=num_classes, normalize=normalize).to(
                torch.float32
            ),
            sklearn_result,
            equal_nan=True,
            atol=1e-8,
            rtol=1e-5,
        )

    def test_multiclass_confusion_matrix_base(self) -> None:
        num_classes = 4
        input = torch.randint(high=num_classes, size=(BATCH_SIZE,))
        target = torch.randint(high=num_classes, size=(BATCH_SIZE,))
        self._test_multiclass_confusion_matrix_with_input(input, target, num_classes)

        # test normalizations
        self._test_multiclass_confusion_matrix_with_input(
            input, target, num_classes, normalize=None
        )
        self._test_multiclass_confusion_matrix_with_input(
            input, target, num_classes, normalize="pred"
        )
        self._test_multiclass_confusion_matrix_with_input(
            input, target, num_classes, normalize="true"
        )

    def test_multiclass_confusion_matrix_with_probabilities(self) -> None:
        # should reduce to [2, 0, 1, 2, 1]
        input = torch.tensor(
            [
                [0.2948, 0.3343, 0.3709],
                [0.4988, 0.4836, 0.0176],
                [0.3727, 0.5145, 0.1128],
                [0.3759, 0.2115, 0.4126],
                [0.3076, 0.4226, 0.2698],
            ]
        )

        target = torch.tensor([0, 1, 2, 0, 0])

        expected_output = torch.tensor([[0, 1, 2], [1, 0, 0], [0, 1, 0]]).to(
            torch.float32
        )

        output = mccm(input, target, num_classes=3, normalize=None).to(torch.float32)

        torch.testing.assert_close(
            output,
            expected_output,
            equal_nan=True,
            atol=1e-8,
            rtol=1e-5,
        )

    def test_multiclass_confusion_matrix_invalid_input(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "normalize must be one of 'all', 'pred', 'true', or 'none'."
        ):
            mccm(
                torch.randint(high=4, size=(BATCH_SIZE,)),
                torch.randint(high=4, size=(BATCH_SIZE,)),
                num_classes=4,
                normalize="this is not a valid option",
            )

        with self.assertRaisesRegex(
            ValueError, r"Must be at least two classes for confusion matrix"
        ):
            mccm(
                torch.randint(high=4, size=(BATCH_SIZE,)),
                torch.randint(high=4, size=(BATCH_SIZE,)),
                num_classes=1,
            )

        with self.assertRaisesRegex(
            ValueError,
            "The `input` and `target` should have the same first dimension, "
            r"got shapes torch.Size\(\[4, 2\]\) and torch.Size\(\[3\]\).",
        ):
            mccm(torch.rand(4, 2), torch.rand(3), num_classes=2)

        with self.assertRaisesRegex(
            ValueError,
            "target should be a one-dimensional tensor, "
            r"got shape torch.Size\(\[3, 2\]\).",
        ):
            mccm(torch.rand(3, 2), torch.rand(3, 2), num_classes=2)

        with self.assertRaisesRegex(
            ValueError,
            r"input should have shape of \(num_sample,\) or \(num_sample, num_classes\), "
            r"got torch.Size\(\[3, 2, 2\]\).",
        ):
            mccm(torch.rand(3, 2, 2), torch.rand(3), num_classes=2)

        with self.assertRaisesRegex(
            ValueError,
            "Got `input` prediction class which is too large for the number of classes, "
            "num_classes: 3 must be strictly greater than max class predicted: 3.",
        ):
            mccm(
                torch.tensor([1, 2, 3, 3, 2, 1]),
                torch.tensor([0, 2, 2, 2, 1, 1]),
                num_classes=3,
            )

        with self.assertRaisesRegex(
            ValueError,
            "Got `target` class which is larger than the number of classes, "
            "num_classes: 3 must be strictly greater than max target: 3.",
        ):
            mccm(
                torch.tensor([0, 2, 2, 2, 1, 1]),
                torch.tensor([1, 2, 3, 3, 2, 1]),
                num_classes=3,
            )
