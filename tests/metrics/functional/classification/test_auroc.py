# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# pyre-ignore-all-errors[56]: Pyre was not able to infer the type of argument

import unittest

import torch
from sklearn.metrics import roc_auc_score
from torcheval.metrics.functional import binary_auroc, multiclass_auroc
from torcheval.utils.test_utils.metric_class_tester import BATCH_SIZE


class TestBinaryAUROC(unittest.TestCase):
    def _test_auroc_with_input(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        num_tasks: int = 1,
        weight: torch.Tensor | None = None,
        compute_result: torch.Tensor | None = None,
        use_fbgemm: bool | None = False,
    ) -> None:
        if compute_result is None:
            compute_result = (
                torch.tensor(roc_auc_score(target, input))
                if weight is None
                else torch.tensor(roc_auc_score(target, input, sample_weight=weight))
            )
        if torch.cuda.is_available():
            my_compute_result = binary_auroc(
                input.to(device="cuda"),
                target.to(device="cuda"),
                weight=weight if weight is None else weight.to(device="cuda"),
                num_tasks=num_tasks,
                use_fbgemm=use_fbgemm,
            )
            compute_result = compute_result.to(device="cuda")
        else:
            my_compute_result = binary_auroc(input, target, num_tasks=num_tasks)

        torch.testing.assert_close(
            my_compute_result,
            compute_result,
            equal_nan=True,
            atol=1e-8,
            rtol=1e-5,
        )

    def _test_auroc_set(self, use_fbgemm: bool | None = False) -> None:
        input = torch.tensor([1, 1, 0, 0])
        target = torch.tensor([1, 0, 1, 0])
        weight = torch.tensor([0.2, 0.2, 1.0, 1.0], dtype=torch.float64)
        self._test_auroc_with_input(input, target, use_fbgemm=use_fbgemm)
        if use_fbgemm is False:
            # TODO: use_fbgemm = True will fail the situation with weight input
            self._test_auroc_with_input(
                input, target, weight=weight, use_fbgemm=use_fbgemm
            )

        input = torch.rand(BATCH_SIZE)
        target = torch.randint(high=2, size=(BATCH_SIZE,))
        self._test_auroc_with_input(input, target, use_fbgemm=use_fbgemm)

        input = torch.tensor([[1, 1, 1, 0], [0.1, 0.5, 0.7, 0.8]])
        target = torch.tensor([[1, 0, 1, 0], [1, 0, 1, 0]])
        self._test_auroc_with_input(
            input,
            target,
            num_tasks=2,
            compute_result=torch.tensor([0.7500, 0.2500], dtype=torch.float64),
            use_fbgemm=use_fbgemm,
        )

    @unittest.skipUnless(
        condition=torch.cuda.is_available(), reason="This test needs a GPU host to run."
    )
    def test_auroc_fbgemm(self) -> None:
        self._test_auroc_set(use_fbgemm=True)

    def test_auroc_base(self) -> None:
        self._test_auroc_set(use_fbgemm=False)

    def test_binary_auroc_invalid_input(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "The `input` and `target` should have the same shape, "
            r"got shapes torch.Size\(\[4\]\) and torch.Size\(\[3\]\).",
        ):
            binary_auroc(torch.rand(4), torch.rand(3))

        with self.assertRaisesRegex(
            ValueError,
            "The `weight` and `target` should have the same shape, "
            r"got shapes torch.Size\(\[3\]\) and torch.Size\(\[4\]\).",
        ):
            binary_auroc(torch.rand(4), torch.rand(4), weight=torch.rand(3))

        with self.assertRaisesRegex(
            ValueError,
            "`num_tasks = 1`, `input` is expected to be one-dimensional tensor,",
        ):
            binary_auroc(torch.rand(4, 5), torch.rand(4, 5))

        with self.assertRaisesRegex(
            ValueError,
            "`num_tasks = 2`, `input`'s shape is expected to be",
        ):
            binary_auroc(torch.rand(4, 5), torch.rand(4, 5), num_tasks=2)


class TestMulticlassAUROC(unittest.TestCase):
    def test_auroc_base(self) -> None:
        # Seed so the randomly drawn target deterministically contains every
        # class. roc_auc_score(multi_class="ovr") raises when y_true is missing a
        # class, which made this test fail intermittently on unlucky batches.
        torch.manual_seed(0)
        num_classes = 4
        input = 10 * torch.randn(BATCH_SIZE, num_classes)
        input_prob = input.abs() / input.abs().sum(dim=-1, keepdim=True)
        target = torch.randint(high=num_classes, size=(BATCH_SIZE,))
        compute_result = torch.tensor(
            roc_auc_score(target, input_prob, average="macro", multi_class="ovr"),
            dtype=torch.float32,
        )
        my_compute_result = multiclass_auroc(
            input_prob, target, num_classes=num_classes
        )
        torch.testing.assert_close(
            my_compute_result,
            compute_result,
            equal_nan=True,
            atol=1e-8,
            rtol=1e-5,
        )

    def test_auroc_average_options(self) -> None:
        input = torch.tensor(
            [
                [0.16, 0.04, 0.8],
                [0.1, 0.7, 0.2],
                [0.16, 0.8, 0.04],
                [0.16, 0.04, 0.8],
            ]
        )
        target = torch.tensor([0, 0, 1, 2])
        # average = macro
        compute_result = torch.tensor(
            roc_auc_score(target, input, average="macro", multi_class="ovr"),
            dtype=torch.float32,
        )
        my_compute_result = multiclass_auroc(
            input, target, num_classes=3, average="macro"
        )
        torch.testing.assert_close(
            my_compute_result,
            compute_result,
            equal_nan=True,
            atol=1e-8,
            rtol=1e-5,
        )

        # average = None
        # sklearn.metrics.roc_auc_score does not support average=None
        expected_compute_result = torch.tensor([0.2500, 1.0000, 5 / 6])
        my_compute_result = multiclass_auroc(input, target, num_classes=3, average=None)
        torch.testing.assert_close(my_compute_result, expected_compute_result)

    def test_multiclass_auroc_invalid_input(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "`average` was not in the allowed value of .*, got micro."
        ):
            multiclass_auroc(
                torch.randint(high=4, size=(BATCH_SIZE,)),
                torch.randint(high=4, size=(BATCH_SIZE,)),
                num_classes=4,
                average="micro",
            )

        with self.assertRaisesRegex(ValueError, "`num_classes` has to be at least 2."):
            multiclass_auroc(torch.rand(4, 2), torch.rand(2), num_classes=1)

        with self.assertRaisesRegex(
            ValueError,
            "The `input` and `target` should have the same first dimension, "
            r"got shapes torch.Size\(\[4, 2\]\) and torch.Size\(\[3\]\).",
        ):
            multiclass_auroc(torch.rand(4, 2), torch.rand(3), num_classes=2)

        with self.assertRaisesRegex(
            ValueError,
            "target should be a one-dimensional tensor, "
            r"got shape torch.Size\(\[3, 2\]\).",
        ):
            multiclass_auroc(torch.rand(3, 2), torch.rand(3, 2), num_classes=2)

        with self.assertRaisesRegex(
            ValueError,
            r"input should have shape of \(num_sample, num_classes\), "
            r"got torch.Size\(\[3, 4\]\) and num_classes=2.",
        ):
            multiclass_auroc(torch.rand(3, 4), torch.rand(3), num_classes=2)
