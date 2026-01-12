# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import numpy as np
import torch
from sklearn.metrics import f1_score
from torcheval.metrics.functional import (
    binary_f1_score as my_binary_f1_score,
    multiclass_f1_score as my_f1_score,
)
from torcheval.utils.test_utils.metric_class_tester import BATCH_SIZE


class TestBinaryF1Score(unittest.TestCase):
    def _test_binary_f1_score_with_input(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        threshold: float = 0.5,
    ) -> None:
        input_np = input.numpy().round()
        target_np = target.squeeze().numpy()
        sklearn_result = torch.tensor(f1_score(target_np, input_np)).to(torch.float32)

        torch.testing.assert_close(
            my_binary_f1_score(input, target, threshold=threshold),
            sklearn_result,
            equal_nan=True,
            atol=1e-8,
            rtol=1e-5,
        )

    def test_binary_f1_score(self) -> None:
        num_classes = 2
        input = torch.randint(high=num_classes, size=(BATCH_SIZE,))
        target = torch.randint(high=num_classes, size=(BATCH_SIZE,))

        self._test_binary_f1_score_with_input(input, target)

    def test_binary_f1_score_with_0s(self) -> None:
        num_classes = 2

        # test input all 0s
        input = torch.zeros(size=(BATCH_SIZE,))
        target = torch.randint(high=num_classes, size=(BATCH_SIZE,))

        torch.testing.assert_close(
            my_binary_f1_score(input, target),
            torch.Tensor([0.0]).sum(),
            equal_nan=True,
            atol=1e-8,
            rtol=1e-5,
        )

        # test target all 0s
        input = torch.randint(high=num_classes, size=(BATCH_SIZE,))
        target = torch.zeros(size=(BATCH_SIZE,))

        torch.testing.assert_close(
            my_binary_f1_score(input, target),
            torch.Tensor([0.0]).sum(),
            equal_nan=True,
            atol=1e-8,
            rtol=1e-5,
        )

    def test_binary_f1_score_thresholding(self) -> None:
        num_classes = 2
        input = torch.randint(high=num_classes, size=(BATCH_SIZE,))
        target = torch.randint(high=num_classes, size=(BATCH_SIZE,))

        # test threshold larger than every prediction gives 0
        torch.testing.assert_close(
            my_binary_f1_score(input, target, threshold=2),
            torch.Tensor([0.0]).sum(),
            equal_nan=True,
            atol=1e-8,
            rtol=1e-5,
        )

    def test_binary_f1_score_invalid_input(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "input should be a one-dimensional tensor for binary f1 score, "
            r"got shape torch.Size\(\[4, 2\]\).",
        ):
            my_binary_f1_score(torch.rand(4, 2), torch.rand(3))

        with self.assertRaisesRegex(
            ValueError,
            "target should be a one-dimensional tensor for binary f1 score, "
            r"got shape torch.Size\(\[4, 2\]\).",
        ):
            my_binary_f1_score(torch.rand(3), torch.rand(4, 2))

        with self.assertRaisesRegex(
            ValueError,
            "The `input` and `target` should have the same dimensions, "
            r"got shapes torch.Size\(\[11\]\) and torch.Size\(\[10\]\).",
        ):
            my_binary_f1_score(torch.rand(11), torch.rand(10))


class TestMultiClassF1Score(unittest.TestCase):
    def _test_f1_score_with_input(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        num_classes: int | None = None,
        average: str | None = "micro",
    ) -> None:
        if num_classes is None:
            if average == "micro":
                my_compute_result = my_f1_score(input, target)
            else:
                my_compute_result = my_f1_score(input, target, average=average)
        else:
            if average == "micro":
                my_compute_result = my_f1_score(input, target, num_classes=num_classes)
            else:
                my_compute_result = my_f1_score(
                    input, target, num_classes=num_classes, average=average
                )
        target = target.squeeze().flatten()
        input_squeezed = input.squeeze()
        input_label_ids = (
            torch.argmax(input_squeezed, dim=1)
            if input_squeezed.ndim == 2
            else input_squeezed
        )
        input = input_label_ids.flatten()
        compute_result = torch.tensor(f1_score(target, input, average=average)).to(
            torch.float32
        )
        if my_compute_result.shape != compute_result.shape:
            compute_result = torch.from_numpy(
                np.append(compute_result.numpy(), 0.0)
            ).to(torch.float32)
        torch.testing.assert_close(
            my_compute_result,
            compute_result,
            equal_nan=True,
            atol=1e-8,
            rtol=1e-5,
        )

    def test_f1_score_base(self) -> None:
        num_classes = 4
        input = torch.randint(high=num_classes, size=(BATCH_SIZE,))
        target = torch.randint(high=num_classes, size=(BATCH_SIZE,))
        self._test_f1_score_with_input(input, target)

        input = torch.rand(BATCH_SIZE, num_classes)
        self._test_f1_score_with_input(input, target)

    def test_f1_score_average(self) -> None:
        num_classes = 4
        input = torch.randint(high=num_classes, size=(BATCH_SIZE,))
        target = torch.randint(high=num_classes, size=(BATCH_SIZE,))
        self._test_f1_score_with_input(
            input, target, num_classes=num_classes, average=None
        )
        input = torch.rand(BATCH_SIZE, num_classes)
        self._test_f1_score_with_input(
            input, target, num_classes=num_classes, average="macro"
        )
        input = torch.rand(BATCH_SIZE, num_classes)
        self._test_f1_score_with_input(
            input, target, num_classes=num_classes, average="weighted"
        )

    def test_f1_score_label_not_exist(self) -> None:
        num_classes = 4
        input = torch.randint(high=num_classes, size=(BATCH_SIZE,))
        target = torch.randint(high=num_classes - 1, size=(BATCH_SIZE,))
        self._test_f1_score_with_input(input, target)

        input = torch.randint(high=num_classes - 1, size=(BATCH_SIZE,))
        target = torch.randint(high=num_classes - 1, size=(BATCH_SIZE,))
        self._test_f1_score_with_input(
            input, target, num_classes=num_classes, average="macro"
        )

        input = torch.randint(high=num_classes - 1, size=(BATCH_SIZE,))
        target = torch.randint(high=num_classes, size=(BATCH_SIZE,))
        self._test_f1_score_with_input(
            input, target, num_classes=num_classes, average=None
        )

        input = torch.randint(high=num_classes - 1, size=(BATCH_SIZE,))
        target = torch.randint(high=num_classes - 1, size=(BATCH_SIZE,))
        self._test_f1_score_with_input(
            input, target, num_classes=num_classes, average=None
        )

    def test_f1_score_invalid_input(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "`average` was not in the allowed value of .*, got gaussian."
        ):
            my_f1_score(
                torch.randint(high=4, size=(BATCH_SIZE,)),
                torch.randint(high=4, size=(BATCH_SIZE,)),
                num_classes=4,
                average="gaussian",
            )

        with self.assertRaisesRegex(
            ValueError,
            r"num_classes should be a positive number when average=None, "
            r"got num_classes=None",
        ):
            my_f1_score(
                torch.randint(high=4, size=(BATCH_SIZE,)),
                torch.randint(high=4, size=(BATCH_SIZE,)),
                average=None,
            )

        with self.assertRaisesRegex(
            ValueError,
            "The `input` and `target` should have the same first dimension, "
            r"got shapes torch.Size\(\[4, 2\]\) and torch.Size\(\[3\]\).",
        ):
            my_f1_score(torch.rand(4, 2), torch.rand(3), num_classes=2)

        with self.assertRaisesRegex(
            ValueError,
            "target should be a one-dimensional tensor, "
            r"got shape torch.Size\(\[3, 2\]\).",
        ):
            my_f1_score(torch.rand(3, 2), torch.rand(3, 2), num_classes=2)

        with self.assertRaisesRegex(
            ValueError,
            r"input should have shape of \(num_sample,\) or \(num_sample, num_classes\), "
            r"got torch.Size\(\[3, 2, 2\]\).",
        ):
            my_f1_score(torch.rand(3, 2, 2), torch.rand(3), num_classes=2)
