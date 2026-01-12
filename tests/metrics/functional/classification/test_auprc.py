# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import random
import unittest

import numpy as np
import torch
from sklearn.metrics import average_precision_score as sk_ap
from torcheval.metrics.functional import binary_auprc, multiclass_auprc
from torcheval.metrics.functional.classification.auprc import multilabel_auprc
from torcheval.utils.test_utils.metric_class_tester import BATCH_SIZE


class TestBinaryAUPRC(unittest.TestCase):
    def _get_sklearn_equivalent(
        self, input: torch.Tensor, target: torch.Tensor, device: str = "cpu"
    ) -> torch.Tensor:
        # Convert input/target to sklearn style inputs
        # run each task once at a time since no multi-task/multiclass
        # available for sklearn
        skinputs = input.numpy()
        sktargets = target.numpy()
        sk_auprcs = []
        for i in range(skinputs.shape[0]):
            skinput = skinputs[i, :]
            sktarget = sktargets[i, :]
            sk_auprcs.append(np.nan_to_num(sk_ap(sktarget, skinput)))
        return torch.tensor(sk_auprcs, device=device).to(torch.float32)

    def _test_binary_auprc_with_input(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        num_tasks: int = 1,
        compute_result: torch.Tensor | None = None,
    ) -> None:
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"

        if compute_result is None:
            compute_result = self._get_sklearn_equivalent(input, target, device)

        # Get torcheval compute result
        te_compute_result = binary_auprc(
            input.to(device=device),
            target.to(device=device),
            num_tasks=num_tasks,
        )

        torch.testing.assert_close(
            te_compute_result,
            compute_result,
            equal_nan=True,
            atol=1e-8,
            rtol=1e-5,
        )

    def _get_rand_inputs_binary(
        self, num_tasks: int, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input = torch.rand(size=[num_tasks, batch_size])
        targets = torch.randint(low=0, high=2, size=[num_tasks, batch_size])
        return input, targets

    def test_binary_auprc_docstring_examples(self) -> None:
        """This test also tests whether the shape is preserved for 1D and 2D examples with ntasks=1"""
        input = torch.tensor([0.1, 0.5, 0.7, 0.8])
        target = torch.tensor([1, 0, 1, 1])
        compute_result = torch.tensor(0.9167)

        torch.testing.assert_close(
            binary_auprc(input, target),
            compute_result,
            atol=1e-4,
            rtol=1e-4,
        )

        input = torch.tensor([[1, 1, 1, 0]])
        target = torch.tensor([[1, 0, 1, 0]])
        compute_result = torch.tensor([0.6667])

        torch.testing.assert_close(
            binary_auprc(input, target),
            compute_result,
            atol=1e-4,
            rtol=1e-4,
        )

        input = torch.tensor([[0.1, 0.5, 0.7, 0.8], [1, 1, 1, 0]])
        target = torch.tensor([[1, 0, 1, 1], [1, 0, 1, 0]])
        compute_result = torch.tensor([0.9167, 0.6667])

        torch.testing.assert_close(
            binary_auprc(input, target, num_tasks=2),
            compute_result,
            atol=1e-4,
            rtol=1e-4,
        )

    def test_binary_auprc_with_good_input(self) -> None:
        # one task
        input, target = self._get_rand_inputs_binary(1, BATCH_SIZE)
        self._test_binary_auprc_with_input(input, target, num_tasks=1)

        # two tasks
        input, target = self._get_rand_inputs_binary(2, BATCH_SIZE)
        self._test_binary_auprc_with_input(input, target, num_tasks=2)

        # random number of tasks between 3 and 10
        num_tasks = random.randint(3, 10)
        input, target = self._get_rand_inputs_binary(num_tasks, BATCH_SIZE)
        self._test_binary_auprc_with_input(input, target, num_tasks=num_tasks)

    def test_binary_auprc_invalid_input(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "The `input` and `target` should have the same shape, "
            r"got shapes torch.Size\(\[4\]\) and torch.Size\(\[3\]\).",
        ):
            binary_auprc(torch.rand(4), torch.rand(3))

        with self.assertRaisesRegex(
            ValueError,
            "`num_tasks = 1`, `input` and `target` are expected to be one-dimensional tensors or 1xN tensors, but got shape input: "
            r"torch.Size\(\[4, 5\]\), target: torch.Size\(\[4, 5\]\).",
        ):
            binary_auprc(torch.rand(4, 5), torch.rand(4, 5))

        with self.assertRaisesRegex(
            ValueError,
            "`num_tasks = 1`, `input` and `target` are expected to be one-dimensional tensors or 1xN tensors, but got shape input: "
            r"torch.Size\(\[4, 5, 5\]\), target: torch.Size\(\[4, 5, 5\]\).",
        ):
            binary_auprc(torch.rand(4, 5, 5), torch.rand(4, 5, 5))

        with self.assertRaisesRegex(
            ValueError,
            r"`num_tasks = 2`, `input` and `target` shape is expected to be \(2, num_samples\), but got shape input: "
            r"torch.Size\(\[4, 5\]\), target: torch.Size\(\[4, 5\]\).",
        ):
            binary_auprc(torch.rand(4, 5), torch.rand(4, 5), num_tasks=2)


class TestMulticlassAUPRC(unittest.TestCase):
    def _get_sklearn_equivalent(
        self, input: torch.Tensor, target: torch.Tensor, device: str = "cpu"
    ) -> torch.Tensor:
        # Convert input/target to sklearn style inputs
        # run each task once at a time since no multi-task/multiclass
        # available for sklearn
        skinputs = input.numpy()
        sktargets = target.numpy()
        auprcs = []
        for i in range(input.shape[1]):
            skinput = skinputs[:, i]
            sktarget = np.where(sktargets == i, 1, 0)
            auprcs.append(np.nan_to_num(sk_ap(sktarget, skinput)))
        return torch.tensor(auprcs, device=device).to(torch.float32)

    def _test_multiclass_auprc_with_input(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        num_classes: int,
        compute_result: torch.Tensor | None = None,
    ) -> None:
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"

        # get sklearn compute result if none given
        if compute_result is None:
            compute_result = self._get_sklearn_equivalent(input, target, device)

        # Get torcheval compute result
        te_compute_result = multiclass_auprc(
            input.to(device=device),
            target.to(device=device),
            num_classes=num_classes,
            average=None,
        )

        # test no average
        torch.testing.assert_close(
            te_compute_result,
            compute_result,
            equal_nan=True,
            atol=1e-8,
            rtol=1e-5,
        )

    def _get_rand_inputs_multiclass(
        self, num_classes: int, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input = torch.rand(size=[batch_size, num_classes])
        targets = torch.randint(low=0, high=num_classes, size=[batch_size])
        return input, targets

    def test_multiclass_auprc_with_good_input(self) -> None:
        num_classes = 2
        input, target = self._get_rand_inputs_multiclass(num_classes, BATCH_SIZE)
        self._test_multiclass_auprc_with_input(input, target, num_classes)

        num_classes = 4
        input, target = self._get_rand_inputs_multiclass(num_classes, BATCH_SIZE)
        self._test_multiclass_auprc_with_input(input, target, num_classes)

        num_classes = 5
        input, target = self._get_rand_inputs_multiclass(num_classes, BATCH_SIZE)
        self._test_multiclass_auprc_with_input(input, target, num_classes)

        num_classes = 8
        input, target = self._get_rand_inputs_multiclass(num_classes, BATCH_SIZE)
        self._test_multiclass_auprc_with_input(input, target, num_classes)

    def test_multiclass_auprc_options(self) -> None:
        # average = macro, num_classes not given
        input, target = self._get_rand_inputs_multiclass(5, BATCH_SIZE)
        compute_result = torch.mean(self._get_sklearn_equivalent(input, target))
        te_compute_result = multiclass_auprc(input, target, average="macro")
        torch.testing.assert_close(
            te_compute_result,
            compute_result,
            equal_nan=True,
            atol=1e-8,
            rtol=1e-5,
        )

        # average = macro (not given), num_classes given
        input, target = self._get_rand_inputs_multiclass(5, BATCH_SIZE)
        compute_result = torch.mean(self._get_sklearn_equivalent(input, target))
        te_compute_result = multiclass_auprc(input, target)
        torch.testing.assert_close(
            te_compute_result,
            compute_result,
            equal_nan=True,
            atol=1e-8,
            rtol=1e-5,
        )

        # average = none
        input, target = self._get_rand_inputs_multiclass(5, BATCH_SIZE)
        compute_result = self._get_sklearn_equivalent(input, target)
        te_compute_result = multiclass_auprc(input, target, average="none")
        torch.testing.assert_close(
            te_compute_result,
            compute_result,
            equal_nan=True,
            atol=1e-8,
            rtol=1e-5,
        )
        te_compute_result = multiclass_auprc(input, target, average=None)
        torch.testing.assert_close(
            te_compute_result,
            compute_result,
            equal_nan=True,
            atol=1e-8,
            rtol=1e-5,
        )

    def test_multiclass_auprc_docstring_examples(self) -> None:
        input = torch.tensor([[0.5647, 0.2726], [0.9143, 0.1895], [0.7782, 0.3082]])
        target = torch.tensor([0, 1, 0])
        output = torch.tensor([0.5833, 0.3333])
        result = multiclass_auprc(input, target, average=None)
        torch.testing.assert_close(
            result,
            output,
            equal_nan=True,
            atol=1e-4,
            rtol=1e-3,
        )

        avg_result = multiclass_auprc(input, target)
        avg_output = torch.tensor(0.4583)
        torch.testing.assert_close(
            avg_result,
            avg_output,
            equal_nan=True,
            atol=1e-4,
            rtol=1e-3,
        )

        input = torch.tensor([[0.1, 1], [0.5, 1], [0.7, 1], [0.8, 0]])
        target = torch.tensor([1, 0, 0, 1])
        result = multiclass_auprc(input, target, 2, average=None)
        output = torch.tensor([0.5833, 0.4167])
        torch.testing.assert_close(
            avg_result,
            avg_output,
            equal_nan=True,
            atol=1e-4,
            rtol=1e-3,
        )

    def test_multiclass_auprc_invalid_input(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "`average` was not in the allowed value of .*, got micro."
        ):
            num_classes = 4
            input, target = self._get_rand_inputs_multiclass(num_classes, BATCH_SIZE)
            multiclass_auprc(
                input,
                target,
                num_classes=num_classes,
                average="micro",
            )

        with self.assertRaisesRegex(ValueError, "`num_classes` has to be at least 2."):
            multiclass_auprc(torch.rand(4, 2), torch.rand(2), num_classes=1)

        with self.assertRaisesRegex(
            ValueError,
            "The `input` and `target` should have the same first dimension, "
            r"got shapes torch.Size\(\[4, 2\]\) and torch.Size\(\[3\]\).",
        ):
            multiclass_auprc(torch.rand(4, 2), torch.rand(3), num_classes=2)

        with self.assertRaisesRegex(
            ValueError,
            "target should be a one-dimensional tensor, "
            r"got shape torch.Size\(\[3, 2\]\).",
        ):
            multiclass_auprc(torch.rand(3, 2), torch.rand(3, 2), num_classes=2)

        with self.assertRaisesRegex(
            ValueError,
            r"input should have shape of \(num_sample, num_classes\), "
            r"got torch.Size\(\[3, 4\]\) and num_classes=2.",
        ):
            multiclass_auprc(torch.rand(3, 4), torch.rand(3), num_classes=2)


class TestMultilabelAUPRC(unittest.TestCase):
    def _get_sklearn_equivalent(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        average: str | None = "macro",
        device: str = "cpu",
    ) -> torch.Tensor:
        # Convert input/target to sklearn style inputs
        # run each task once at a time since no multi-task/multilabel
        # available for sklearn
        skinputs = input.numpy()
        sktargets = target.numpy()
        auprcs = []
        for i in range(skinputs.shape[1]):
            auprcs.append(np.nan_to_num(sk_ap(sktargets[:, i], skinputs[:, i])))
        result = torch.tensor(auprcs, device=device)
        if average == "macro":
            result = torch.mean(result)
        return result.to(torch.float32)

    def _test_multilabel_auprc_with_input(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        num_labels: int,
        average: str | None = "macro",
        compute_result: torch.Tensor | None = None,
    ) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # get sklearn compute result if none given
        if compute_result is None:
            compute_result = self._get_sklearn_equivalent(
                input, target, average, device
            )

        # Get torcheval compute result
        te_compute_result = multilabel_auprc(
            input.to(device=device),
            target.to(device=device),
            num_labels=num_labels,
            average=average,
        )

        # test no average
        torch.testing.assert_close(
            te_compute_result,
            compute_result,
            equal_nan=True,
            atol=1e-8,
            rtol=1e-5,
        )

    def _test_multilabel_auprc_unspecified_average(self, num_labels: int) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        input, target = self._get_rand_inputs_multilabel(num_labels, BATCH_SIZE)

        compute_result = self._get_sklearn_equivalent(input, target, device=device)

        te_compute_result = multilabel_auprc(
            input.to(device=device),
            target.to(device=device),
            num_labels=num_labels,
        )
        torch.testing.assert_close(
            te_compute_result,
            compute_result,
            equal_nan=True,
            atol=1e-8,
            rtol=1e-5,
        )

    def _get_rand_inputs_multilabel(
        self, num_labels: int, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input = torch.rand(size=[batch_size, num_labels])
        targets = torch.randint(low=0, high=2, size=[batch_size, num_labels])
        return input, targets

    def test_multilabel_auprc_none_average_with_good_input(self) -> None:
        num_labels = 2
        input, target = self._get_rand_inputs_multilabel(num_labels, BATCH_SIZE)
        self._test_multilabel_auprc_with_input(input, target, num_labels, average=None)

        num_labels = 4
        input, target = self._get_rand_inputs_multilabel(num_labels, BATCH_SIZE)
        self._test_multilabel_auprc_with_input(input, target, num_labels, average=None)

        num_labels = 5
        input, target = self._get_rand_inputs_multilabel(num_labels, BATCH_SIZE)
        self._test_multilabel_auprc_with_input(input, target, num_labels, average=None)

        num_labels = 8
        input, target = self._get_rand_inputs_multilabel(num_labels, BATCH_SIZE)
        self._test_multilabel_auprc_with_input(input, target, num_labels, average=None)

    def test_multilabel_auprc_macro_average_with_good_input(self) -> None:
        num_labels = 2
        input, target = self._get_rand_inputs_multilabel(num_labels, BATCH_SIZE)
        self._test_multilabel_auprc_with_input(
            input, target, num_labels, average="macro"
        )

        num_labels = 4
        input, target = self._get_rand_inputs_multilabel(num_labels, BATCH_SIZE)
        self._test_multilabel_auprc_with_input(
            input, target, num_labels, average="macro"
        )

        num_labels = 5
        input, target = self._get_rand_inputs_multilabel(num_labels, BATCH_SIZE)
        self._test_multilabel_auprc_with_input(
            input, target, num_labels, average="macro"
        )

        num_labels = 8
        input, target = self._get_rand_inputs_multilabel(num_labels, BATCH_SIZE)
        self._test_multilabel_auprc_with_input(
            input, target, num_labels, average="macro"
        )

    def test_multilabel_auprc_unspecified_average_with_good_input(self) -> None:
        num_labels = 2
        self._test_multilabel_auprc_unspecified_average(num_labels)

        num_labels = 4
        self._test_multilabel_auprc_unspecified_average(num_labels)

        num_labels = 5
        self._test_multilabel_auprc_unspecified_average(num_labels)

        num_labels = 8
        self._test_multilabel_auprc_unspecified_average(num_labels)

    def test_multilabel_auprc_options(self) -> None:
        # average = macro, num_labels not given
        input, target = self._get_rand_inputs_multilabel(5, BATCH_SIZE)
        compute_result = self._get_sklearn_equivalent(input, target)
        te_compute_result = multilabel_auprc(input, target, average="macro")
        torch.testing.assert_close(
            te_compute_result,
            compute_result,
            equal_nan=True,
            atol=1e-8,
            rtol=1e-5,
        )

        # average = macro (not given), num_labels given
        input, target = self._get_rand_inputs_multilabel(5, BATCH_SIZE)
        compute_result = self._get_sklearn_equivalent(input, target)
        te_compute_result = multilabel_auprc(input, target, num_labels=5)
        torch.testing.assert_close(
            te_compute_result,
            compute_result,
            equal_nan=True,
            atol=1e-8,
            rtol=1e-5,
        )

        # average = macro (not given), num_labels (not given)
        input, target = self._get_rand_inputs_multilabel(5, BATCH_SIZE)
        compute_result = self._get_sklearn_equivalent(input, target)
        te_compute_result = multilabel_auprc(input, target)
        torch.testing.assert_close(
            te_compute_result,
            compute_result,
            equal_nan=True,
            atol=1e-8,
            rtol=1e-5,
        )

        # average = none, num_labels not given
        input, target = self._get_rand_inputs_multilabel(5, BATCH_SIZE)
        compute_result = self._get_sklearn_equivalent(input, target, average="none")
        te_compute_result = multilabel_auprc(input, target, average="none")
        torch.testing.assert_close(
            te_compute_result,
            compute_result,
            equal_nan=True,
            atol=1e-8,
            rtol=1e-5,
        )
        te_compute_result = multilabel_auprc(input, target, average=None)
        torch.testing.assert_close(
            te_compute_result,
            compute_result,
            equal_nan=True,
            atol=1e-8,
            rtol=1e-5,
        )

    def test_multilabel_auprc_docstring_example(self) -> None:
        input = torch.tensor(
            [
                [0.75, 0.05, 0.35],
                [0.45, 0.75, 0.05],
                [0.05, 0.55, 0.75],
                [0.05, 0.65, 0.05],
            ]
        )
        target = torch.tensor([[1, 0, 1], [0, 0, 0], [0, 1, 1], [1, 1, 1]])

        compute_result_average_none = torch.tensor([0.7500, 0.5833, 0.9167])
        torch.testing.assert_close(
            multilabel_auprc(input, target, num_labels=3, average=None),
            compute_result_average_none,
            atol=1e-4,
            rtol=1e-4,
        )
        torch.testing.assert_close(
            multilabel_auprc(input, target, average=None),
            compute_result_average_none,
            atol=1e-4,
            rtol=1e-4,
        )

        compute_result_average_macro = torch.tensor(0.7500)
        torch.testing.assert_close(
            multilabel_auprc(input, target, num_labels=3, average="macro"),
            compute_result_average_macro,
            atol=1e-4,
            rtol=1e-4,
        )
        torch.testing.assert_close(
            multilabel_auprc(input, target, num_labels=3),
            compute_result_average_macro,
            atol=1e-4,
            rtol=1e-4,
        )
        torch.testing.assert_close(
            multilabel_auprc(input, target, average="macro"),
            compute_result_average_macro,
            atol=1e-4,
            rtol=1e-4,
        )
        torch.testing.assert_close(
            multilabel_auprc(input, target),
            compute_result_average_macro,
            atol=1e-4,
            rtol=1e-4,
        )

        input = torch.tensor([[0.1, 0, 0], [0, 1, 0], [0.1, 0.2, 0.7], [0, 0, 1]])
        target = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]])
        compute_result_average_none = torch.tensor([0.5000, 1.0000, 1.0000])
        torch.testing.assert_close(
            multilabel_auprc(input, target, average=None),
            compute_result_average_none,
            atol=1e-4,
            rtol=1e-4,
        )

    def test_multilabel_auprc_invalid_input(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            r"input should be a two-dimensional tensor, got shape torch.Size\(\[4\]\).",
        ):
            num_labels = 4
            input, target = self._get_rand_inputs_multilabel(num_labels, BATCH_SIZE)
            multilabel_auprc(
                torch.rand(4),
                torch.randint(high=2, size=(4,)),
                num_labels=num_labels,
                average="micro",
            )

        with self.assertRaisesRegex(
            ValueError, "`average` was not in the allowed value of .*, got micro."
        ):
            num_labels = 4
            input, target = self._get_rand_inputs_multilabel(num_labels, BATCH_SIZE)
            multilabel_auprc(
                input,
                target,
                num_labels=num_labels,
                average="micro",
            )

        with self.assertRaisesRegex(
            ValueError,
            "Expected both input.shape and target.shape to have the same shape"
            r" but got torch.Size\(\[5, 3\]\) and torch.Size\(\[4, 3\]\).",
        ):
            multilabel_auprc(
                torch.rand(5, 3), torch.randint(high=2, size=(4, 3)), num_labels=3
            )

        with self.assertRaisesRegex(
            ValueError,
            r"input should be a two-dimensional tensor, got shape torch.Size\(\[4, 3, 2\]\).",
        ):
            num_labels = 4
            input, target = self._get_rand_inputs_multilabel(num_labels, BATCH_SIZE)
            multilabel_auprc(
                torch.rand(4, 3, 2),
                torch.randint(high=2, size=(4, 3, 2)),
                num_labels=num_labels,
                average="micro",
            )

        with self.assertRaisesRegex(
            ValueError,
            r"input should have shape of \(num_sample, num_labels\), "
            r"got torch.Size\(\[4, 2\]\) and num_labels=3.",
        ):
            multilabel_auprc(
                torch.rand(4, 2), torch.randint(high=2, size=(4, 2)), num_labels=3
            )
