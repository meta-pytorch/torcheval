# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import numpy as np

import torch
from sklearn.metrics import accuracy_score
from torcheval.metrics.functional import (
    binary_accuracy,
    multiclass_accuracy,
    multilabel_accuracy,
    topk_multilabel_accuracy,
)
from torcheval.utils.test_utils.metric_class_tester import BATCH_SIZE


class TestBinaryAccuracy(unittest.TestCase):
    def _test_binary_accuracy_with_input(
        self, input: torch.Tensor, target: torch.Tensor
    ) -> None:
        input_np = input.numpy().round()
        target_np = target.squeeze().numpy()
        sklearn_result = torch.tensor(accuracy_score(target_np, input_np)).to(
            torch.float32
        )

        torch.testing.assert_close(
            binary_accuracy(input, target),
            sklearn_result,
            equal_nan=True,
            atol=1e-8,
            rtol=1e-5,
        )

    def test_binary_accuracy(self) -> None:
        num_classes = 2
        input = torch.randint(high=num_classes, size=(BATCH_SIZE,))
        target = torch.randint(high=num_classes, size=(BATCH_SIZE,))

        self._test_binary_accuracy_with_input(input, target)

    def test_binary_accuracy_with_bool_target(self) -> None:
        num_classes = 2
        input = torch.randint(high=num_classes, size=(BATCH_SIZE,))
        target = torch.randint(high=num_classes, size=(BATCH_SIZE,))

        output = binary_accuracy(input, target)
        output_bool = binary_accuracy(input, target.bool())

        torch.testing.assert_close(output, output_bool)

    def test_binary_accuracy_with_rounding(self) -> None:
        num_classes = 2
        input = torch.rand(size=(BATCH_SIZE,))
        target = torch.randint(high=num_classes, size=(BATCH_SIZE,))

        self._test_binary_accuracy_with_input(input, target)

    def test_binary_accuracy_invalid_input(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "The `input` and `target` should have the same dimensions, "
            r"got shapes torch.Size\(\[4, 2\]\) and torch.Size\(\[3\]\).",
        ):
            binary_accuracy(torch.rand(4, 2), torch.rand(3))


class TestMultiClassAccuracy(unittest.TestCase):
    def _test_accuracy_with_input(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        num_classes: int,
    ) -> None:
        target_np = target.squeeze().flatten().numpy()
        input_squeezed = input.squeeze()
        input_label_ids = (
            torch.argmax(input_squeezed, dim=1)
            if input_squeezed.ndim == 2
            else input_squeezed
        )
        input_np = input_label_ids.flatten().numpy()
        compute_result = torch.tensor(accuracy_score(target_np, input_np)).to(
            torch.float32
        )
        torch.testing.assert_close(
            multiclass_accuracy(input, target),
            compute_result,
            equal_nan=True,
            atol=1e-8,
            rtol=1e-5,
        )

    def test_accuracy_base(self) -> None:
        num_classes = 4
        input = torch.randint(high=num_classes, size=(BATCH_SIZE,))
        target = torch.randint(high=num_classes, size=(BATCH_SIZE,))
        self._test_accuracy_with_input(input, target, num_classes)

        y_score = torch.rand(BATCH_SIZE, num_classes)
        self._test_accuracy_with_input(y_score, target, num_classes)

    def test_accuracy_class_average(self) -> None:
        num_classes = 4
        # high=num_classes-1 gives us NaN value for the last class
        input = torch.randint(high=num_classes, size=(BATCH_SIZE,))
        target = torch.randint(high=num_classes - 1, size=(BATCH_SIZE,))

        input_flattened = input.flatten()
        target_flattened = target.flatten()
        accuracy_per_class = np.empty(num_classes)
        for i in range(num_classes):
            mask = target_flattened == i
            if mask.sum().item() == 0:
                accuracy_per_class[i] = np.nan
            else:
                accuracy_per_class[i] = accuracy_score(
                    target_flattened[mask].numpy(),
                    input_flattened[mask].numpy(),
                )


        torch.testing.assert_close(
            multiclass_accuracy(
                input, target, average="macro", num_classes=num_classes
            ),
            torch.tensor(np.mean(accuracy_per_class[~np.isnan(accuracy_per_class)])).to(
                torch.float32
            ),
            atol=1e-8,
            rtol=1e-5,
        )

        torch.testing.assert_close(
            multiclass_accuracy(input, target, average=None, num_classes=num_classes),
            torch.tensor(accuracy_per_class).view(-1).to(torch.float32),
            equal_nan=True,
            atol=1e-8,
            rtol=1e-5,
        )

    def test_accuracy_invalid_input(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "`average` was not in the allowed value of .*, got weighted."
        ):
            multiclass_accuracy(
                torch.randint(high=4, size=(BATCH_SIZE,)),
                torch.randint(high=4, size=(BATCH_SIZE,)),
                num_classes=4,
                average="weighted",
            )

        with self.assertRaisesRegex(
            ValueError,
            "num_classes should be a positive number when average=None. Got num_classes=None",
        ):
            multiclass_accuracy(
                torch.randint(high=4, size=(BATCH_SIZE,)),
                torch.randint(high=4, size=(BATCH_SIZE,)),
                average=None,
            )

        with self.assertRaisesRegex(
            ValueError,
            "The `input` and `target` should have the same first dimension, "
            r"got shapes torch.Size\(\[4, 2\]\) and torch.Size\(\[3\]\).",
        ):
            multiclass_accuracy(torch.rand(4, 2), torch.rand(3))

        with self.assertRaisesRegex(
            ValueError, "target should be a one-dimensional tensor, got shape ."
        ):
            multiclass_accuracy(torch.rand(BATCH_SIZE, 1), torch.rand(BATCH_SIZE, 1))

        with self.assertRaisesRegex(ValueError, "input should have shape"):
            multiclass_accuracy(torch.rand(BATCH_SIZE, 2, 1), torch.rand(BATCH_SIZE))


class TestMultilabelAccuracy(unittest.TestCase):
    def _test_exact_match_with_input(
        self, input: torch.Tensor, target: torch.Tensor
    ) -> None:
        input_np = input.numpy().round()
        target_np = target.squeeze().numpy()
        sklearn_result = torch.tensor(accuracy_score(target_np, input_np)).to(
            torch.float32
        )

        torch.testing.assert_close(
            multilabel_accuracy(input, target),
            sklearn_result,
            equal_nan=True,
            atol=1e-8,
            rtol=1e-5,
        )

    def _test_hamming_with_input(
        self, input: torch.Tensor, target: torch.Tensor
    ) -> None:
        input_np = input.numpy().round()
        target_np = target.numpy()
        sklearn_result = torch.tensor(
            accuracy_score(target_np.flatten(), input_np.flatten())
        ).to(torch.float32)

        torch.testing.assert_close(
            multilabel_accuracy(input, target, criteria="hamming"),
            sklearn_result,
            equal_nan=True,
            atol=1e-8,
            rtol=1e-5,
        )

    def test_multilabel_accuracy_exact_match(self) -> None:
        num_classes = 2
        input = torch.randint(0, 2, size=(BATCH_SIZE, num_classes))
        target = torch.randint(0, 2, size=(BATCH_SIZE, num_classes))

        self._test_exact_match_with_input(input, target)

    def test_multilabel_accuracy_exact_match_with_rounding(self) -> None:
        num_classes = 2
        input = torch.rand(size=(BATCH_SIZE, num_classes))
        target = torch.randint(0, 2, size=(BATCH_SIZE, num_classes))

        self._test_exact_match_with_input(input, target)

    def test_multilabel_accuracy_hamming(self) -> None:
        num_classes = 2
        input = torch.randint(0, 2, size=(BATCH_SIZE, num_classes))
        target = torch.randint(0, 2, size=(BATCH_SIZE, num_classes))

        self._test_hamming_with_input(input, target)

    def test_multilabel_accuracy_criteria(self) -> None:
        input = torch.tensor([[0, 1, 1], [1, 1, 0], [1, 1, 1], [0, 1, 1]])
        target = torch.tensor([[1, 1, 0], [0, 1, 0], [1, 1, 1], [0, 1, 0]])
        # test overlap criteria with positive inputs
        overlap_accuracy = multilabel_accuracy(input, target, criteria="overlap")
        torch.testing.assert_close(
            overlap_accuracy,
            torch.tensor(1.0),
            equal_nan=True,
            atol=1e-8,
            rtol=1e-5,
        )
        # test overlap criteria with no positive inputs: input[2]
        input_overlap_no_pos = torch.tensor(
            [[0, 0, 1], [0, 0, 0], [0, 0, 0], [1, 0, 0]]
        )
        target_overlap_no_pos = torch.tensor(
            [[0, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 1]]
        )
        overlap_accuracy = multilabel_accuracy(
            input_overlap_no_pos, target_overlap_no_pos, criteria="overlap"
        )
        torch.testing.assert_close(
            overlap_accuracy,
            torch.tensor(1 / 4),
            equal_nan=True,
            atol=1e-8,
            rtol=1e-5,
        )
        # test contain criteria: input[1], input[2], input[3]
        contain_accuracy = multilabel_accuracy(input, target, criteria="contain")
        torch.testing.assert_close(
            contain_accuracy,
            torch.tensor(3 / 4),
            equal_nan=True,
            atol=1e-8,
            rtol=1e-5,
        )
        # test belong criteria: input[2]
        belong_accuracy = multilabel_accuracy(input, target, criteria="belong")
        torch.testing.assert_close(
            belong_accuracy,
            torch.tensor(1 / 4),
            equal_nan=True,
            atol=1e-8,
            rtol=1e-5,
        )

    def test_multilabel_accuracy_hamming_with_rounding(self) -> None:
        num_classes = 2
        input = torch.rand(size=(BATCH_SIZE, num_classes))
        target = torch.randint(0, 2, size=(BATCH_SIZE, num_classes))

        self._test_hamming_with_input(input, target)

    def test_multilabel_accuracy_invalid_input(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "`criteria` was not in the allowed value of .*, got weighted."
        ):
            multilabel_accuracy(
                torch.randint(0, 2, size=(BATCH_SIZE, 4)),
                torch.randint(0, 2, size=(BATCH_SIZE, 4)),
                criteria="weighted",
            )

        with self.assertRaisesRegex(
            ValueError,
            "The `input` and `target` should have the same dimensions, "
            r"got shapes torch.Size\(\[4, 2\]\) and torch.Size\(\[3\]\).",
        ):
            multilabel_accuracy(torch.rand(4, 2), torch.rand(3))


class TestTopKAccuracy(unittest.TestCase):
    def test_topk_accuracy_with_input(self) -> None:
        input = torch.tensor(
            [
                [0.9, 0.1, 0, 0],
                [0.1, 0.2, 0.4, 0.3],
                [0.4, 0.4, 0.1, 0.1],
                [0, 0, 0.8, 0.2],
            ]
        )
        target = torch.tensor([0, 1, 2, 3])

        compute_result = torch.tensor(0.5)
        torch.testing.assert_close(
            multiclass_accuracy(input, target, k=2),
            compute_result,
            equal_nan=True,
            atol=1e-8,
            rtol=1e-5,
        )

        compute_result = torch.tensor(1.0)
        torch.testing.assert_close(
            multiclass_accuracy(input, target, k=4),
            compute_result,
            equal_nan=True,
            atol=1e-8,
            rtol=1e-5,
        )

    def test_topk_accuracy_with_input_macro(self) -> None:
        input = torch.tensor(
            [
                [0.9, 0.1, 0, 0],
                [0.9, 0.1, 0, 0],
                [0.3, 0.2, 0.4, 0.1],
                [0.3, 0.2, 0.4, 0.1],
                [0.4, 0.4, 0.1, 0.1],
                [0.3, 0.5, 0.1, 0.1],
            ]
        )
        target = torch.tensor([0, 0, 0, 0, 2, 2])

        compute_result = torch.tensor(0.5)
        torch.testing.assert_close(
            multiclass_accuracy(input, target, average="macro", num_classes=4, k=2),
            compute_result,
            equal_nan=True,
            atol=1e-8,
            rtol=1e-5,
        )

        compute_result = torch.tensor(1.0)
        torch.testing.assert_close(
            multiclass_accuracy(input, target, average="macro", num_classes=4, k=4),
            compute_result,
            equal_nan=True,
            atol=1e-8,
            rtol=1e-5,
        )

    def test_topk_accuracy_with_input_no_average(self) -> None:
        input = torch.tensor(
            [
                [0.9, 0.1, 0, 0],
                [0.9, 0.1, 0, 0],
                [0.3, 0.2, 0.4, 0.1],
                [0.3, 0.2, 0.4, 0.1],
                [0.4, 0.4, 0.1, 0.1],
                [0.3, 0.5, 0.1, 0.1],
            ]
        )
        target = torch.tensor([0, 0, 0, 0, 2, 2])

        compute_result = torch.tensor([1.0, np.NAN, 0, np.NAN])
        torch.testing.assert_close(
            multiclass_accuracy(input, target, average=None, num_classes=4, k=2),
            compute_result,
            equal_nan=True,
            atol=1e-8,
            rtol=1e-5,
        )

        compute_result = torch.tensor([1.0, np.NAN, 1.0, np.NAN])
        torch.testing.assert_close(
            multiclass_accuracy(input, target, average=None, num_classes=4, k=4),
            compute_result,
            equal_nan=True,
            atol=1e-8,
            rtol=1e-5,
        )

    def test_topk_accuracy_invalid_params(self) -> None:
        k = -2
        with self.assertRaisesRegex(
            ValueError,
            f"Expected `k` to be an integer greater than 0, but {k} was provided.",
        ):
            multiclass_accuracy(torch.rand(4, 2), torch.rand(4), k=k)

        input = torch.rand(2)
        with self.assertRaisesRegex(
            ValueError,
            r"input should have shape \(num_sample, num_classes\) for k > 1, "
            r"got shape torch.Size\(\[2\]\).",
        ):
            multiclass_accuracy(input, torch.rand(2), k=2)


class TestTopKMultilabelAccuracy(unittest.TestCase):
    def test_topk_accuracy_base(self) -> None:
        input = torch.tensor(
            [[0.1, 0.5, 0.2], [0.3, 0.2, 0.1], [0.2, 0.4, 0.5], [0, 0.1, 0.9]]
        )
        target = torch.tensor([[1, 1, 0], [0, 1, 0], [1, 1, 1], [0, 1, 0]])

        compute_result = torch.tensor(0.0)
        torch.testing.assert_close(
            topk_multilabel_accuracy(input, target, k=2),
            compute_result,
            equal_nan=True,
            atol=1e-8,
            rtol=1e-5,
        )

    def test_topk_accuracy_hamming(self) -> None:
        input = torch.tensor(
            [[0.1, 0.5, 0.2], [0.3, 0.2, 0.1], [0.2, 0.4, 0.5], [0, 0.1, 0.9]]
        )
        target = torch.tensor([[1, 1, 0], [0, 1, 0], [1, 1, 1], [0, 1, 0]])

        compute_result = torch.tensor(7 / 12)
        torch.testing.assert_close(
            topk_multilabel_accuracy(input, target, criteria="hamming", k=2),
            compute_result,
            equal_nan=True,
            atol=1e-8,
            rtol=1e-5,
        )

    def test_topk_accuracy_overlap(self) -> None:
        input = torch.tensor(
            [[0.1, 0.5, 0.2], [0.3, 0.2, 0.1], [0.2, 0.4, 0.5], [0, 0.1, 0.9]]
        )
        target = torch.tensor([[1, 1, 0], [0, 1, 0], [1, 1, 1], [0, 1, 0]])

        compute_result = torch.tensor(4 / 4)
        torch.testing.assert_close(
            topk_multilabel_accuracy(input, target, criteria="overlap", k=2),
            compute_result,
            equal_nan=True,
            atol=1e-8,
            rtol=1e-5,
        )

    def test_topk_accuracy_contain(self) -> None:
        input = torch.tensor(
            [[0.1, 0.5, 0.2], [0.3, 0.2, 0.1], [0.2, 0.4, 0.5], [0, 0.1, 0.9]]
        )
        target = torch.tensor([[1, 1, 0], [0, 1, 0], [1, 1, 1], [0, 1, 0]])

        compute_result = torch.tensor(2 / 4)
        torch.testing.assert_close(
            topk_multilabel_accuracy(input, target, criteria="contain", k=2),
            compute_result,
            equal_nan=True,
            atol=1e-8,
            rtol=1e-5,
        )

    def test_topk_accuracy_belong(self) -> None:
        input = torch.tensor(
            [[0.1, 0.5, 0.2], [0.3, 0.2, 0.1], [0.2, 0.4, 0.5], [0, 0.1, 0.9]]
        )
        target = torch.tensor([[1, 1, 0], [0, 1, 0], [1, 1, 1], [0, 1, 0]])

        compute_result = torch.tensor(1 / 4)
        torch.testing.assert_close(
            topk_multilabel_accuracy(input, target, criteria="belong", k=2),
            compute_result,
            equal_nan=True,
            atol=1e-8,
            rtol=1e-5,
        )

    def test_topk_accuracy_invalid_params(self) -> None:
        k = -2
        with self.assertRaisesRegex(
            ValueError,
            f"Expected `k` to be an integer greater than 1, but {k} was provided.",
        ):
            topk_multilabel_accuracy(torch.rand(4, 2), torch.rand(4), k=k)

        input = torch.rand(2)
        with self.assertRaisesRegex(
            ValueError,
            r"input should have shape \(num_sample, num_classes\) for k > 1, "
            r"got shape torch.Size\(\[2\]\).",
        ):
            topk_multilabel_accuracy(input, torch.rand(2), k=2)
