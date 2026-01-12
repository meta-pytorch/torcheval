# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import torch
from sklearn.metrics import confusion_matrix as skcm
from torcheval.metrics import BinaryConfusionMatrix, MulticlassConfusionMatrix
from torcheval.utils.test_utils.metric_class_tester import (
    BATCH_SIZE,
    MetricClassTester,
    NUM_TOTAL_UPDATES,
)


class TestBinaryConfusionMatrix(MetricClassTester):
    def _test_binary_confusion_matrix_with_input(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        normalize: str | None = None,
    ) -> None:
        input_np = input.flatten().numpy()
        target_np = target.flatten().numpy()

        compute_result = torch.tensor(
            skcm(target_np, input_np, labels=[0, 1], normalize=normalize),
            dtype=torch.float32,
        )

        self.run_class_implementation_tests(
            metric=BinaryConfusionMatrix(normalize=normalize),
            state_names={"confusion_matrix"},
            update_kwargs={"input": input, "target": target},
            compute_result=compute_result,
        )

    def test_binary_confusion_matrix_base(self) -> None:
        num_classes = 2
        input = torch.randint(high=num_classes, size=(NUM_TOTAL_UPDATES, BATCH_SIZE))
        target = torch.randint(high=num_classes, size=(NUM_TOTAL_UPDATES, BATCH_SIZE))

        self._test_binary_confusion_matrix_with_input(input, target)

    def test_binary_confusion_matrix_normalization(self) -> None:
        num_classes = 2
        input = torch.randint(high=num_classes, size=(NUM_TOTAL_UPDATES, BATCH_SIZE))
        target = torch.randint(high=num_classes, size=(NUM_TOTAL_UPDATES, BATCH_SIZE))

        self._test_binary_confusion_matrix_with_input(input, target, normalize="all")
        self._test_binary_confusion_matrix_with_input(input, target, normalize="true")
        self._test_binary_confusion_matrix_with_input(input, target, normalize="pred")

        # ========= test normalization with normalized() =============
        input = torch.randint(high=num_classes, size=(BATCH_SIZE,))
        target = torch.randint(high=num_classes, size=(BATCH_SIZE,))
        metric = BinaryConfusionMatrix()
        metric.update(input, target)
        metric.compute()

        # all
        compute_result_all = torch.tensor(
            skcm(target, input, labels=[0, 1], normalize="all"),
            dtype=torch.float32,
        )
        torch.testing.assert_close(
            metric.normalized("all").to(torch.float32),
            compute_result_all,
            equal_nan=True,
            atol=1e-8,
            rtol=1e-5,
        )

        # pred
        compute_result_pred = torch.tensor(
            skcm(target, input, labels=[0, 1], normalize="pred"),
            dtype=torch.float32,
        )
        torch.testing.assert_close(
            metric.normalized("pred").to(torch.float32),
            compute_result_pred,
            equal_nan=True,
            atol=1e-8,
            rtol=1e-5,
        )

        # true
        compute_result_true = torch.tensor(
            skcm(target, input, labels=[0, 1], normalize="true"),
            dtype=torch.float32,
        )
        torch.testing.assert_close(
            metric.normalized("true").to(torch.float32),
            compute_result_true,
            equal_nan=True,
            atol=1e-8,
            rtol=1e-5,
        )

    def test_binary_confusion_matrix_score_thresholding(self) -> None:
        num_classes = 2
        threshold = 0.7

        input = [
            torch.tensor([0.7, 0.6, 0.5, 0.3, 0.9, 0.1, 1.0, 0.95, 0.2]),
            torch.tensor([0.7, 0.8, 0.3, 0.3, 0.3, 1.0, 0.1, 0.65, 0.2]),
        ]

        input_thresholded = [
            torch.tensor([1, 0, 0, 0, 1, 0, 1, 1, 0]),
            torch.tensor([1, 1, 0, 0, 0, 1, 0, 0, 0]),
        ]

        target = [
            torch.randint(high=num_classes, size=(9,)),
            torch.randint(high=num_classes, size=(9,)),
        ]

        compute_result = (
            torch.tensor(
                skcm(
                    torch.cat(target, dim=0),
                    torch.cat(input_thresholded, dim=0),
                    labels=[0, 1],
                )
            )
            .to(torch.float32)
            .squeeze()
        )

        self.run_class_implementation_tests(
            metric=BinaryConfusionMatrix(threshold=threshold),
            state_names={"confusion_matrix"},
            update_kwargs={"input": input, "target": target},
            compute_result=compute_result,
            num_total_updates=2,
            num_processes=2,
        )

    def test_binary_confusion_matrix_invalid_input(self) -> None:
        metric = BinaryConfusionMatrix()

        with self.assertRaisesRegex(
            ValueError,
            "input should be a one-dimensional tensor for binary confusion matrix, "
            r"got shape torch.Size\(\[5, 10\]\).",
        ):
            input = torch.randint(high=2, size=(5, 10))
            target = torch.randint(high=2, size=(5, 10))
            metric.update(input, target)

        with self.assertRaisesRegex(
            ValueError,
            "target should be a one-dimensional tensor for binary confusion matrix, "
            r"got shape torch.Size\(\[5, 10\]\).",
        ):
            input = torch.randint(high=2, size=(10,))
            target = torch.randint(high=2, size=(5, 10))
            metric.update(input, target)

        with self.assertRaisesRegex(
            ValueError,
            "The `input` and `target` should have the same dimensions, "
            r"got shapes torch.Size\(\[11\]\) and torch.Size\(\[10\]\).",
        ):
            input = torch.randint(high=2, size=(11,))
            target = torch.randint(high=2, size=(10,))
            metric.update(input, target)

        with self.assertRaisesRegex(
            ValueError, "normalize must be one of 'all', 'pred', 'true', or 'none'."
        ):
            metric = BinaryConfusionMatrix(normalize="this is not a valid option")


class TestMulticlassConfusionMatrix(MetricClassTester):
    def _test_multiclass_confusion_matrix_with_input(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        num_classes: int,
        normalize: str | None = None,
    ) -> None:
        if input.ndim == 3:
            input_np = input.argmax(dim=2).flatten().numpy()
        else:
            input_np = input.flatten().numpy()
        target_np = target.flatten().numpy()

        compute_result = torch.tensor(
            skcm(
                target_np,
                input_np,
                labels=list(range(num_classes)),
                normalize=normalize,
            ),
            dtype=torch.float32,
        )

        self.run_class_implementation_tests(
            metric=MulticlassConfusionMatrix(num_classes, normalize=normalize),
            state_names={"confusion_matrix"},
            update_kwargs={"input": input, "target": target},
            compute_result=compute_result,
        )

    def test_multiclass_confusion_matrix_base(self) -> None:
        num_classes = 6
        input = torch.randint(high=num_classes, size=(NUM_TOTAL_UPDATES, BATCH_SIZE))
        target = torch.randint(high=num_classes, size=(NUM_TOTAL_UPDATES, BATCH_SIZE))

        self._test_multiclass_confusion_matrix_with_input(input, target, num_classes)

    def test_multiclass_confusion_matrix_normalization(self) -> None:
        num_classes = 6
        input = torch.randint(high=num_classes, size=(NUM_TOTAL_UPDATES, BATCH_SIZE))
        target = torch.randint(high=num_classes, size=(NUM_TOTAL_UPDATES, BATCH_SIZE))
        self._test_multiclass_confusion_matrix_with_input(
            input, target, num_classes, normalize="all"
        )
        self._test_multiclass_confusion_matrix_with_input(
            input, target, num_classes, normalize="true"
        )
        self._test_multiclass_confusion_matrix_with_input(
            input, target, num_classes, normalize="pred"
        )

        # ========= test normalization with normalized() =============
        input = torch.randint(high=num_classes, size=(BATCH_SIZE,))
        target = torch.randint(high=num_classes, size=(BATCH_SIZE,))
        metric = MulticlassConfusionMatrix(num_classes)
        metric.update(input, target)
        metric.compute()

        # all
        compute_result_all = torch.tensor(
            skcm(target, input, labels=list(range(num_classes)), normalize="all"),
            dtype=torch.float32,
        )
        torch.testing.assert_close(
            metric.normalized("all").to(torch.float32),
            compute_result_all,
            equal_nan=True,
            atol=1e-8,
            rtol=1e-5,
        )

        # pred
        compute_result_pred = torch.tensor(
            skcm(target, input, labels=list(range(num_classes)), normalize="pred"),
            dtype=torch.float32,
        )
        torch.testing.assert_close(
            metric.normalized("pred").to(torch.float32),
            compute_result_pred,
            equal_nan=True,
            atol=1e-8,
            rtol=1e-5,
        )

        # true
        compute_result_true = torch.tensor(
            skcm(target, input, labels=list(range(num_classes)), normalize="true"),
            dtype=torch.float32,
        )
        torch.testing.assert_close(
            metric.normalized("true").to(torch.float32),
            compute_result_true,
            equal_nan=True,
            atol=1e-8,
            rtol=1e-5,
        )

    def test_multiclass_confusion_matrix_with_probabilities(self) -> None:
        num_classes = 3
        input = [
            torch.tensor(
                [
                    [0.2948, 0.3343, 0.3709],
                    [0.4988, 0.4836, 0.0176],
                    [0.3727, 0.5145, 0.1128],
                    [0.3759, 0.2115, 0.4126],
                    [0.3076, 0.4226, 0.2698],
                ]
            ),
            torch.tensor(
                [
                    [0, 1, 0],
                    [1, 0, 0],
                    [0, 0, 1],
                    [0, 1, 0],
                    [1, 0, 0],
                ]
            ),
        ]

        input_sklearn = [torch.tensor([2, 0, 1, 2, 1]), torch.tensor([1, 0, 2, 1, 0])]

        target = [
            torch.randint(high=num_classes, size=(5,)),
            torch.randint(high=num_classes, size=(5,)),
        ]

        compute_result = (
            torch.tensor(
                skcm(
                    torch.cat(target, dim=0),
                    torch.cat(input_sklearn, dim=0),
                    labels=[0, 1, 2],
                )
            )
            .to(torch.float32)
            .squeeze()
        )
        self.run_class_implementation_tests(
            metric=MulticlassConfusionMatrix(num_classes),
            state_names={"confusion_matrix"},
            update_kwargs={"input": input, "target": target},
            compute_result=compute_result,
            num_total_updates=2,
            num_processes=2,
        )

        # test also with random inputs:
        input = torch.randint(
            high=num_classes, size=(NUM_TOTAL_UPDATES, BATCH_SIZE, num_classes)
        )
        target = torch.randint(high=num_classes, size=(NUM_TOTAL_UPDATES, BATCH_SIZE))
        self._test_multiclass_confusion_matrix_with_input(input, target, num_classes)

    def test_multiclass_confusion_matrix_invalid_input(self) -> None:
        num_classes = 4

        with self.assertRaisesRegex(
            ValueError, "normalize must be one of 'all', 'pred', 'true', or 'none'."
        ):
            metric = MulticlassConfusionMatrix(
                num_classes, normalize="this is not a valid option"
            )

        with self.assertRaisesRegex(
            ValueError, r"Must be at least two classes for confusion matrix"
        ):
            metric = MulticlassConfusionMatrix(1)

        metric = MulticlassConfusionMatrix(num_classes)

        with self.assertRaisesRegex(
            ValueError,
            "The `input` and `target` should have the same first dimension, "
            r"got shapes torch.Size\(\[4, 2\]\) and torch.Size\(\[3\]\).",
        ):
            metric.update(torch.rand(4, 2), torch.rand(3))

        with self.assertRaisesRegex(
            ValueError,
            "target should be a one-dimensional tensor, "
            r"got shape torch.Size\(\[3, 2\]\).",
        ):
            metric.update(torch.rand(3, 2), torch.rand(3, 2))

        with self.assertRaisesRegex(
            ValueError,
            r"input should have shape of \(num_sample,\) or \(num_sample, num_classes\), "
            r"got torch.Size\(\[3, 2, 2\]\).",
        ):
            metric.update(torch.rand(3, 2, 2), torch.rand(3))

        with self.assertRaisesRegex(
            ValueError,
            "Got `input` prediction class which is too large for the number of classes, "
            "num_classes: 4 must be strictly greater than max class predicted: 4.",
        ):
            metric.update(
                torch.tensor([1, 2, 4, 3, 2, 1]),
                torch.tensor([0, 2, 2, 2, 1, 1]),
            )

        with self.assertRaisesRegex(
            ValueError,
            "Got `target` class which is larger than the number of classes, "
            "num_classes: 4 must be strictly greater than max target: 4.",
        ):
            metric.update(
                torch.tensor([0, 2, 2, 2, 1, 1]),
                torch.tensor([1, 2, 4, 3, 2, 1]),
            )
