# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import numpy as np
import torch
from sklearn.metrics import recall_score as ref_recall_score
from torcheval.metrics.functional import multiclass_recall as my_recall_score
from torcheval.metrics.functional.classification import (
    binary_recall as my_binary_recall_score,
)
from torcheval.utils.test_utils.metric_class_tester import BATCH_SIZE


class TestMulticlassRecall(unittest.TestCase):
    def _test_recall_with_input(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        num_classes: int | None = None,
        average: str | None = "micro",
    ) -> None:
        if num_classes is None:
            if average == "micro":
                my_compute_result = my_recall_score(input, target)
            else:
                my_compute_result = my_recall_score(input, target, average=average)
        else:
            if average == "micro":
                my_compute_result = my_recall_score(
                    input, target, num_classes=num_classes
                )
            else:
                my_compute_result = my_recall_score(
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
        compute_result = torch.tensor(
            ref_recall_score(target, input, average=average)
        ).to(torch.float32)
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

    def test_recall_base(self) -> None:
        num_classes = 4
        input = torch.randint(high=num_classes, size=(BATCH_SIZE,))
        target = torch.randint(high=num_classes, size=(BATCH_SIZE,))
        self._test_recall_with_input(input, target)

        input = torch.rand(BATCH_SIZE, num_classes)
        self._test_recall_with_input(input, target)

    def test_recall_average(self) -> None:
        num_classes = 4
        input = torch.randint(high=num_classes, size=(BATCH_SIZE,))
        target = torch.randint(high=num_classes, size=(BATCH_SIZE,))
        self._test_recall_with_input(
            input, target, num_classes=num_classes, average=None
        )

        input = torch.rand(BATCH_SIZE, num_classes)
        self._test_recall_with_input(
            input, target, num_classes=num_classes, average=None
        )

        input = torch.rand(BATCH_SIZE, num_classes)
        self._test_recall_with_input(
            input, target, num_classes=num_classes, average="macro"
        )

        input = torch.rand(BATCH_SIZE, num_classes)
        self._test_recall_with_input(
            input, target, num_classes=num_classes, average="weighted"
        )

    def test_recall_absent_labels(self) -> None:
        num_classes = 4
        input = torch.randint(high=num_classes, size=(BATCH_SIZE,))
        target = torch.randint(high=num_classes - 1, size=(BATCH_SIZE,))
        self._test_recall_with_input(input, target)

        input = torch.randint(high=num_classes - 1, size=(BATCH_SIZE,))
        target = torch.randint(high=num_classes, size=(BATCH_SIZE,))
        self._test_recall_with_input(
            input, target, num_classes=num_classes, average=None
        )

    def test_nan_handling(self) -> None:
        num_classes = 4
        with self.assertLogs(level="WARNING") as logger:
            input = torch.randint(high=num_classes, size=(BATCH_SIZE,))
            target = torch.randint(high=num_classes - 1, size=(BATCH_SIZE,))
            self._test_recall_with_input(
                input, target, num_classes=num_classes, average=None
            )
            self.assertRegex(
                logger.output[0],
                "One or more NaNs identified, as no ground-truth instances of "
                ".* have been seen. These have been converted to zero.",
            )
        with self.assertLogs(level="WARNING") as logger:
            input = torch.randint(high=num_classes - 1, size=(BATCH_SIZE,))
            target = torch.randint(high=num_classes - 1, size=(BATCH_SIZE,))
            self._test_recall_with_input(
                input, target, num_classes=num_classes, average=None
            )
            self.assertRegex(
                logger.output[0],
                "One or more NaNs identified, as no ground-truth instances of "
                ".* have been seen. These have been converted to zero.",
            )

    def test_recall_invalid_input(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "`average` was not in the allowed values of .*, got binary"
        ):
            my_recall_score(
                torch.randint(high=4, size=(BATCH_SIZE,)),
                torch.randint(high=4, size=(BATCH_SIZE,)),
                average="binary",
            )
        with self.assertRaisesRegex(
            ValueError,
            r"`num_classes` should be a positive number when average=None, "
            r"got num_classes=None",
        ):
            my_recall_score(
                torch.randint(high=4, size=(BATCH_SIZE,)),
                torch.randint(high=4, size=(BATCH_SIZE,)),
                average=None,
            )
        with self.assertRaisesRegex(
            ValueError,
            r"The `input` and `target` should have the same first dimension, "
            r"got shapes torch.Size\(\[3, 4\]\) and torch.Size\(\[2\]\).",
        ):
            my_recall_score(torch.rand(3, 4), torch.rand(2), num_classes=4)
        with self.assertRaisesRegex(
            ValueError,
            r"`target` should be a one-dimensional tensor, got shape torch.Size\(\[3, 4\]\).",
        ):
            my_recall_score(torch.rand(3, 4), torch.rand(3, 4), num_classes=4)
        with self.assertRaisesRegex(
            ValueError,
            r"`input` should have shape \(num_samples,\) or \(num_samples, num_classes\), "
            r"got torch.Size\(\[3, 4, 5\]\).",
        ):
            my_recall_score(torch.rand(3, 4, 5), torch.rand(3), num_classes=4)


class TestBinaryRecall(unittest.TestCase):
    def _test_binary_recall_with_input(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        threshold: float = 0.5,
    ) -> None:
        my_compute_result = my_binary_recall_score(input, target, threshold=threshold)

        input_np = np.where(input.numpy() < threshold, 0, 1)
        target_np = target.squeeze().numpy()

        sklearn_result = torch.tensor(
            ref_recall_score(target_np, input_np, average="binary")
        ).to(torch.float32)

        torch.testing.assert_close(
            my_compute_result,
            sklearn_result,
            equal_nan=True,
            atol=1e-8,
            rtol=1e-5,
        )

    def test_binary_recall_base(self) -> None:
        num_classes = 2
        input = torch.randint(high=num_classes, size=(BATCH_SIZE,))
        target = torch.randint(high=num_classes, size=(BATCH_SIZE,))
        self._test_binary_recall_with_input(input, target)

    def test_binary_recall_threshold(self) -> None:
        num_classes = 2
        input = torch.randint(high=num_classes, size=(BATCH_SIZE,))
        target = torch.randint(high=num_classes, size=(BATCH_SIZE,))
        self._test_binary_recall_with_input(input, target, threshold=0)

        self._test_binary_recall_with_input(input, target, threshold=0.2)

        self._test_binary_recall_with_input(input, target, threshold=0.8)

        self._test_binary_recall_with_input(input, target, threshold=1)

        self._test_binary_recall_with_input(input, target, threshold=2)

    def test_binary_recall_with_rounding(self) -> None:
        num_classes = 2
        input = torch.rand(size=(BATCH_SIZE,))
        target = torch.randint(high=num_classes, size=(BATCH_SIZE,))

        self._test_binary_recall_with_input(input, target)

    def test_binary_recall_invalid_input(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "The `input` and `target` should have the same dimensions, "
            r"got shapes torch.Size\(\[4, 2\]\) and torch.Size\(\[3\]\).",
        ):
            my_binary_recall_score(torch.rand(4, 2), torch.rand(3))
