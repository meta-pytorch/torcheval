# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import torch
from torcheval.metrics.functional.classification import (
    binary_binned_auroc,
    multiclass_binned_auroc,
)


class TestBinaryBinnedAUROC(unittest.TestCase):
    def _test_binary_binned_auroc_with_input(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        num_tasks: int,
        threshold: torch.Tensor | int,
        compute_result: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        my_compute_result = binary_binned_auroc(
            input,
            target,
            num_tasks=num_tasks,
            threshold=threshold,
        )
        torch.testing.assert_close(
            my_compute_result,
            compute_result,
            equal_nan=True,
            atol=1e-8,
            rtol=1e-5,
        )

    def test_binary_binned_auroc(self) -> None:
        input = torch.tensor([0.2, 0.3, 0.4, 0.5])
        target = torch.tensor([0, 0, 1, 1])
        num_tasks = 1
        threshold = torch.tensor([0.0000, 0.2500, 0.7500, 1.0000])

        compute_result = (
            torch.tensor([0.75], dtype=torch.float64),
            torch.tensor([0.0000, 0.2500, 0.7500, 1.0000]),
        )
        self._test_binary_binned_auroc_with_input(
            input, target, num_tasks, threshold, compute_result
        )

        if torch.cuda.is_available():
            self._test_binary_binned_auroc_with_input(
                input.to("cuda"),
                target.to("cuda"),
                num_tasks,
                threshold.to("cuda"),
                # pyrefly: ignore [bad-argument-type]
                tuple([c.to("cuda") for c in compute_result]),
            )

        input = torch.tensor([[0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.1, 0.5]])
        target = torch.tensor([[0, 0, 1, 1], [1, 0, 1, 1]])
        num_tasks = 2
        threshold = torch.tensor([0.0000, 0.2500, 0.7500, 1.0000])

        compute_result = (
            torch.tensor([0.75, 0.6666666666666666], dtype=torch.float64),
            torch.tensor([0.0000, 0.2500, 0.7500, 1.0000]),
        )
        self._test_binary_binned_auroc_with_input(
            input, target, num_tasks, threshold, compute_result
        )

        if torch.cuda.is_available():
            self._test_binary_binned_auroc_with_input(
                input.to("cuda"),
                target.to("cuda"),
                num_tasks,
                threshold.to("cuda"),
                # pyrefly: ignore [bad-argument-type]
                tuple([c.to("cuda") for c in compute_result]),
            )

        input = torch.tensor([0.2, 0.8, 0.5, 0.9])
        target = torch.tensor([0, 1, 0, 1])
        num_tasks = 1
        threshold = 5
        compute_result = (
            torch.tensor([1.0], dtype=torch.float64),
            torch.tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000]),
        )
        self._test_binary_binned_auroc_with_input(
            input, target, num_tasks, threshold, compute_result
        )

        if torch.cuda.is_available():
            self._test_binary_binned_auroc_with_input(
                input.to("cuda"),
                target.to("cuda"),
                num_tasks,
                threshold,
                # pyrefly: ignore [bad-argument-type]
                tuple([c.to("cuda") for c in compute_result]),
            )

    def test_binary_binned_auroc_invalid_input(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "`num_tasks` has to be at least 1.",
        ):
            binary_binned_auroc(torch.rand(3, 2), torch.rand(3, 2), num_tasks=-1)

        with self.assertRaisesRegex(
            ValueError,
            "The `input` and `target` should have the same shape, "
            r"got shapes torch.Size\(\[4\]\) and torch.Size\(\[3\]\).",
        ):
            binary_binned_auroc(torch.rand(4), torch.rand(3))

        with self.assertRaisesRegex(
            ValueError,
            "`num_tasks = 1`, `input` is expected to be one-dimensional tensor, "
            r"but got shape torch.Size\(\[3, 2\]\).",
        ):
            binary_binned_auroc(torch.rand(3, 2), torch.rand(3, 2))

        with self.assertRaisesRegex(
            ValueError,
            "`input` is expected to be two dimensions or less, but got 3D tensor.",
        ):
            binary_binned_auroc(torch.rand(3, 2, 2), torch.rand(3, 2, 2), num_tasks=2)

        with self.assertRaisesRegex(
            ValueError, "The `threshold` should be a sorted tensor."
        ):
            binary_binned_auroc(
                torch.rand(4),
                torch.rand(4),
                threshold=torch.tensor([0.1, 0.2, 0.5, 0.7, 0.6]),
            )

        with self.assertRaisesRegex(
            ValueError,
            r"The values in `threshold` should be in the range of \[0, 1\].",
        ):
            binary_binned_auroc(
                torch.rand(4),
                torch.rand(4),
                threshold=torch.tensor([-0.1, 0.2, 0.5, 0.7]),
            )


class TestMulticlassBinnedAUROC(unittest.TestCase):
    def _test_multiclass_binned_auroc_with_input(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        num_classes: int,
        threshold: torch.Tensor | int,
        compute_result: tuple[torch.Tensor, torch.Tensor],
        average: str | None = "macro",
    ) -> None:
        my_compute_result = multiclass_binned_auroc(
            input,
            target,
            num_classes=num_classes,
            threshold=threshold,
            average=average,
        )
        torch.testing.assert_close(
            my_compute_result,
            compute_result,
            equal_nan=True,
            atol=1e-8,
            rtol=1e-5,
        )

    def test_multiclass_binned_auroc(self) -> None:
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
        num_classes = 3
        threshold = 5

        compute_result = (
            torch.tensor(0.4000),
            torch.tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000]),
        )
        self._test_multiclass_binned_auroc_with_input(
            input,
            target,
            num_classes,
            threshold,
            compute_result,
        )

        if torch.cuda.is_available():
            self._test_multiclass_binned_auroc_with_input(
                input.to("cuda"),
                target.to("cuda"),
                num_classes,
                threshold,
                # pyrefly: ignore [bad-argument-type]
                tuple([c.to("cuda") for c in compute_result]),
            )

        compute_result = (
            torch.tensor([0.5000, 0.2500, 0.2500, 0.0000, 1.0000]),
            torch.tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000]),
        )
        self._test_multiclass_binned_auroc_with_input(
            input,
            target,
            num_classes,
            threshold,
            compute_result,
            average=None,
        )

        if torch.cuda.is_available():
            self._test_multiclass_binned_auroc_with_input(
                input.to("cuda"),
                target.to("cuda"),
                num_classes,
                threshold,
                # pyrefly: ignore [bad-argument-type]
                tuple([c.to("cuda") for c in compute_result]),
                average=None,
            )

    def test_multiclass_binned_auroc_invalid_input(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "`average` was not in the allowed value of .*, got micro."
        ):
            multiclass_binned_auroc(
                torch.randint(high=4, size=(4,)),
                torch.randint(high=4, size=(4,)),
                num_classes=4,
                average="micro",
            )

        with self.assertRaisesRegex(ValueError, "`num_classes` has to be at least 2."):
            multiclass_binned_auroc(torch.rand(4, 2), torch.rand(2), num_classes=1)

        with self.assertRaisesRegex(
            ValueError,
            "The `input` and `target` should have the same first dimension, "
            r"got shapes torch.Size\(\[4, 2\]\) and torch.Size\(\[3\]\).",
        ):
            multiclass_binned_auroc(torch.rand(4, 2), torch.rand(3), num_classes=2)

        with self.assertRaisesRegex(
            ValueError,
            "target should be a one-dimensional tensor, "
            r"got shape torch.Size\(\[3, 2\]\).",
        ):
            multiclass_binned_auroc(torch.rand(3, 2), torch.rand(3, 2), num_classes=2)

        with self.assertRaisesRegex(
            ValueError,
            r"input should have shape of \(num_sample, num_classes\), "
            r"got torch.Size\(\[3, 4\]\) and num_classes=2.",
        ):
            multiclass_binned_auroc(torch.rand(3, 4), torch.rand(3), num_classes=2)

        with self.assertRaisesRegex(
            ValueError, "The `threshold` should be a sorted tensor."
        ):
            multiclass_binned_auroc(
                torch.randint(high=4, size=(4,)),
                torch.randint(high=4, size=(4,)),
                num_classes=4,
                threshold=torch.tensor([0.1, 0.2, 0.5, 0.7, 0.6]),
            )

        with self.assertRaisesRegex(
            ValueError,
            r"The values in `threshold` should be in the range of \[0, 1\].",
        ):
            multiclass_binned_auroc(
                torch.randint(high=4, size=(4,)),
                torch.randint(high=4, size=(4,)),
                num_classes=4,
                threshold=torch.tensor([-0.1, 0.2, 0.5, 0.7]),
            )
