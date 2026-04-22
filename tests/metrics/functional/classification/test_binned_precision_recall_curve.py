# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import torch
from torcheval.metrics.functional import (
    binary_binned_precision_recall_curve,
    multiclass_binned_precision_recall_curve,
    multilabel_binned_precision_recall_curve,
)


class TestBinaryBinnedPrecisionRecallCurve(unittest.TestCase):
    def _test_binary_binned_precision_recall_curve_with_input(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        threshold: torch.Tensor | int,
        compute_result: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        my_compute_result = binary_binned_precision_recall_curve(
            input,
            target,
            threshold=threshold,
        )
        torch.testing.assert_close(
            my_compute_result,
            compute_result,
            equal_nan=True,
            atol=1e-8,
            rtol=1e-5,
        )

    def test_binary_binned_precision_recall_curve_base(self) -> None:
        input = torch.tensor([0.2, 0.3, 0.4, 0.5])
        target = torch.tensor([0, 0, 1, 1])
        threshold = torch.tensor([0.0000, 0.2500, 0.7500, 1.0000])

        compute_result = (
            torch.tensor([0.5000, 2 / 3, 1.0000, 1.0000, 1.0000]),
            torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0]),
            torch.tensor([0.0000, 0.2500, 0.7500, 1.0000]),
        )
        self._test_binary_binned_precision_recall_curve_with_input(
            input, target, threshold, compute_result
        )

        if torch.cuda.is_available():
            self._test_binary_binned_precision_recall_curve_with_input(
                input.to("cuda"),
                target.to("cuda"),
                threshold.to("cuda"),
                # pyrefly: ignore [bad-argument-type]
                tuple([c.to("cuda") for c in compute_result]),
            )

        input = torch.tensor([0.2, 0.8, 0.5, 0.9])
        target = torch.tensor([0, 1, 0, 1])
        threshold = 5
        compute_result = (
            torch.tensor(
                [
                    0.5000,
                    2 / 3,
                    2 / 3,
                    1.0000,
                    1.0000,
                    1.0000,
                ]
            ),
            torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0]),
            torch.tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000]),
        )
        self._test_binary_binned_precision_recall_curve_with_input(
            input, target, threshold, compute_result
        )

        if torch.cuda.is_available():
            self._test_binary_binned_precision_recall_curve_with_input(
                input.to("cuda"),
                target.to("cuda"),
                threshold,
                # pyrefly: ignore [bad-argument-type]
                tuple([c.to("cuda") for c in compute_result]),
            )

    def test_binary_binned_precision_recall_curve_invalid_input(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "input should be a one-dimensional tensor, "
            r"got shape torch.Size\(\[3, 2\]\).",
        ):
            binary_binned_precision_recall_curve(torch.rand(3, 2), torch.rand(3))

        with self.assertRaisesRegex(
            ValueError,
            "target should be a one-dimensional tensor, "
            r"got shape torch.Size\(\[3, 2\]\).",
        ):
            binary_binned_precision_recall_curve(torch.rand(3), torch.rand(3, 2))

        with self.assertRaisesRegex(
            ValueError,
            "The `input` and `target` should have the same shape, "
            r"got shapes torch.Size\(\[4\]\) and torch.Size\(\[3\]\).",
        ):
            binary_binned_precision_recall_curve(torch.rand(4), torch.rand(3))
        with self.assertRaisesRegex(
            ValueError, "The `threshold` should be a sorted tensor."
        ):
            binary_binned_precision_recall_curve(
                torch.rand(4),
                torch.rand(4),
                threshold=torch.tensor([0.1, 0.2, 0.5, 0.7, 0.6]),
            )
        with self.assertRaisesRegex(
            ValueError,
            r"The values in `threshold` should be in the range of \[0, 1\].",
        ):
            binary_binned_precision_recall_curve(
                torch.rand(4),
                torch.rand(4),
                threshold=torch.tensor([-0.1, 0.2, 0.5, 0.7]),
            )
        with self.assertRaisesRegex(
            ValueError,
            r"The values in `threshold` should be in the range of \[0, 1\].",
        ):
            binary_binned_precision_recall_curve(
                torch.rand(4),
                torch.rand(4),
                threshold=torch.tensor([0.1, 0.2, 0.5, 1.7]),
            )


class TestMulticlassBinnedPrecisionRecallCurve(unittest.TestCase):
    def test_multiclass_binned_precision_recall_curve_base(self) -> None:
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
        threshold = 10

        compute_result = (
            [
                torch.tensor(
                    [
                        0.4,
                        0.25,
                        0.25,
                        0.25,
                        0.5,
                        0.5,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                    ]
                ),
                torch.tensor(
                    [
                        0.4,
                        0.5,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                    ]
                ),
                torch.tensor(
                    [
                        0.2,
                        0.3333,
                        0.0,
                        0.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                    ]
                ),
            ],
            [
                torch.tensor(
                    [
                        1.0,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.5,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ]
                ),
                torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            ],
            torch.tensor(
                [
                    0.0,
                    0.1111,
                    0.2222,
                    0.3333,
                    0.4444,
                    0.5556,
                    0.6667,
                    0.7778,
                    0.8889,
                    1.0,
                ]
            ),
        )

        for optimization in ("vectorized", "memory"):
            my_compute_result = multiclass_binned_precision_recall_curve(
                input,
                target,
                num_classes=3,
                threshold=threshold,
                optimization=optimization,
            )
            torch.testing.assert_close(
                my_compute_result,
                compute_result,
                equal_nan=True,
                atol=1e-4,
                rtol=1e-4,
            )

    def test_multiclass_binned_precision_recall_curve_invalid_input(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "The `input` and `target` should have the same first dimension, "
            r"got shapes torch.Size\(\[4, 2\]\) and torch.Size\(\[3\]\).",
        ):
            multiclass_binned_precision_recall_curve(
                torch.rand(4, 2), torch.rand(3), num_classes=2
            )

        with self.assertRaisesRegex(
            ValueError,
            "target should be a one-dimensional tensor, "
            r"got shape torch.Size\(\[3, 2\]\).",
        ):
            multiclass_binned_precision_recall_curve(
                torch.rand(3, 2), torch.rand(3, 2), num_classes=2
            )

        with self.assertRaisesRegex(
            ValueError,
            r"input should have shape of \(num_sample, num_classes\), "
            r"got torch.Size\(\[3, 4\]\) and num_classes=2.",
        ):
            multiclass_binned_precision_recall_curve(
                torch.rand(3, 4), torch.rand(3), num_classes=2
            )
        with self.assertRaisesRegex(
            ValueError, "The `threshold` should be a sorted tensor."
        ):
            multiclass_binned_precision_recall_curve(
                torch.rand(4),
                torch.rand(4),
                threshold=torch.tensor([0.1, 0.2, 0.5, 0.7, 0.6]),
            )
        with self.assertRaisesRegex(
            ValueError,
            r"The values in `threshold` should be in the range of \[0, 1\].",
        ):
            multiclass_binned_precision_recall_curve(
                torch.rand(4),
                torch.rand(4),
                threshold=torch.tensor([-0.1, 0.2, 0.5, 0.7]),
            )
        with self.assertRaisesRegex(
            ValueError,
            r"The values in `threshold` should be in the range of \[0, 1\].",
        ):
            multiclass_binned_precision_recall_curve(
                torch.rand(4),
                torch.rand(4),
                threshold=torch.tensor([0.1, 0.2, 0.5, 1.7]),
            )
        with self.assertRaisesRegex(
            ValueError,
            r"Unknown memory approach: expected 'vectorized' or 'memory', but got foo.",
        ):
            multiclass_binned_precision_recall_curve(
                torch.rand(4),
                torch.rand(4),
                threshold=5,
                optimization="foo",
            )


class TestMultilabelBinnedPrecisionRecallCurve(unittest.TestCase):
    def test_multilabel_binned_precision_recall_curve_base(self) -> None:
        input = torch.tensor(
            [
                [0.75, 0.05, 0.35],
                [0.45, 0.75, 0.05],
                [0.05, 0.55, 0.75],
                [0.05, 0.65, 0.05],
            ]
        )
        target = torch.tensor([[1, 0, 1], [0, 0, 0], [0, 1, 1], [1, 1, 1]])

        num_labels = 3

        for optimization in ["vectorized", "memory"]:
            threshold = 5
            my_compute_result = multilabel_binned_precision_recall_curve(
                input, target, num_labels, threshold, optimization
            )
            expected_result = (
                [
                    torch.tensor([0.5000, 0.5000, 1.0000, 1.0000, 1.0000, 1.0000]),
                    torch.tensor([0.5000, 0.6667, 0.6667, 0.0000, 1.0000, 1.0000]),
                    torch.tensor([0.7500, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000]),
                ],
                [
                    torch.tensor([1.0000, 0.5000, 0.5000, 0.5000, 0.0000, 0.0000]),
                    torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]),
                    torch.tensor([1.0000, 0.6667, 0.3333, 0.3333, 0.0000, 0.0000]),
                ],
                torch.tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000]),
            )
            torch.testing.assert_close(
                my_compute_result,
                expected_result,
                equal_nan=True,
                atol=1e-4,
                rtol=1e-4,
            )

            threshold = torch.tensor([0.0, 0.2, 0.5, 0.8, 1.0])
            my_compute_result = multilabel_binned_precision_recall_curve(
                input, target, num_labels, threshold, optimization
            )
            expected_result = (
                [
                    torch.tensor([0.5000, 0.5000, 1.0000, 1.0000, 1.0000, 1.0000]),
                    torch.tensor([0.5000, 0.6667, 0.6667, 1.0000, 1.0000, 1.0000]),
                    torch.tensor([0.7500, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000]),
                ],
                [
                    torch.tensor([1.0000, 0.5000, 0.5000, 0.0000, 0.0000, 0.0000]),
                    torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]),
                    torch.tensor([1.0000, 0.6667, 0.3333, 0.0000, 0.0000, 0.0000]),
                ],
                torch.tensor([0.0000, 0.2000, 0.5000, 0.8000, 1.0000]),
            )

            torch.testing.assert_close(
                my_compute_result,
                expected_result,
                equal_nan=True,
                atol=1e-4,
                rtol=1e-4,
            )

            # Test with floating-point tensors now
            target_floating_point = target.float()
            my_compute_result = multilabel_binned_precision_recall_curve(
                input, target_floating_point, num_labels, threshold, optimization
            )
            torch.testing.assert_close(
                my_compute_result,
                expected_result,
                equal_nan=True,
                atol=1e-4,
                rtol=1e-4,
            )

    def test_multilabel_binned_precision_recall_curve_invalid_input(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "Expected both input.shape and target.shape to have the same shape"
            r" but got torch.Size\(\[4, 2\]\) and torch.Size\(\[3\]\).",
        ):
            multilabel_binned_precision_recall_curve(
                torch.rand(4, 2), torch.rand(3), num_labels=3
            )

        with self.assertRaisesRegex(
            ValueError,
            "input should be a two-dimensional tensor, got shape "
            r"torch.Size\(\[3\]\).",
        ):
            multilabel_binned_precision_recall_curve(
                torch.rand(3), torch.rand(3), num_labels=3
            )

        with self.assertRaisesRegex(
            ValueError,
            "input should have shape of "
            r"\(num_sample, num_labels\), got torch.Size\(\[4, 2\]\) and num_labels=3.",
        ):
            multilabel_binned_precision_recall_curve(
                torch.rand(4, 2), torch.rand(4, 2), num_labels=3
            )
        with self.assertRaisesRegex(
            ValueError, "The `threshold` should be a sorted tensor."
        ):
            multilabel_binned_precision_recall_curve(
                torch.rand(4),
                torch.rand(4),
                threshold=torch.tensor([0.1, 0.2, 0.5, 0.7, 0.6]),
            )
        with self.assertRaisesRegex(
            ValueError,
            r"The values in `threshold` should be in the range of \[0, 1\].",
        ):
            multilabel_binned_precision_recall_curve(
                torch.rand(4),
                torch.rand(4),
                threshold=torch.tensor([-0.1, 0.2, 0.5, 0.7]),
            )
        with self.assertRaisesRegex(
            ValueError,
            r"The values in `threshold` should be in the range of \[0, 1\].",
        ):
            multilabel_binned_precision_recall_curve(
                torch.rand(4),
                torch.rand(4),
                threshold=torch.tensor([0.1, 0.2, 0.5, 1.7]),
            )
        with self.assertRaisesRegex(
            ValueError,
            r"Unknown memory approach: expected 'vectorized' or 'memory', but got cpu.",
        ):
            multilabel_binned_precision_recall_curve(
                torch.rand(4, 3),
                torch.randint(high=3, size=(4, 3)),
                threshold=5,
                optimization="cpu",
            )
