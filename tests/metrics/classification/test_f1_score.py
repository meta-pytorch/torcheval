# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import numpy as np
import torch
from sklearn.metrics import f1_score
from torcheval.metrics import BinaryF1Score, MulticlassF1Score
from torcheval.utils.test_utils.metric_class_tester import (
    BATCH_SIZE,
    MetricClassTester,
    NUM_TOTAL_UPDATES,
)


class TestBinaryF1Score(MetricClassTester):
    def _test_binary_f1_score_with_input(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        threshold: float = 0.5,
    ) -> None:
        input_tensors = [torch.argmax(t, dim=1) if t.ndim == 2 else t for t in input]
        target_tensors = list(target)
        target_np = torch.stack(target_tensors).flatten().numpy()
        input_np = torch.stack(input_tensors).flatten().numpy()
        compute_result = torch.tensor(
            f1_score(target_np, input_np, average="binary"), dtype=torch.float32
        )

        self.run_class_implementation_tests(
            metric=BinaryF1Score(threshold=threshold),
            state_names={"num_tp", "num_label", "num_prediction"},
            update_kwargs={"input": input, "target": target},
            compute_result=compute_result,
        )

    def test_binary_f1_score_base(self) -> None:
        num_classes = 2
        input = torch.randint(high=num_classes, size=(NUM_TOTAL_UPDATES, BATCH_SIZE))
        target = torch.randint(high=num_classes, size=(NUM_TOTAL_UPDATES, BATCH_SIZE))

        self._test_binary_f1_score_with_input(input, target)

    def test_binary_f1_score_with_0s(self) -> None:
        num_classes = 2
        expected_result = torch.Tensor([0.0]).sum()

        # test input all 0s
        input = torch.zeros(size=(NUM_TOTAL_UPDATES, BATCH_SIZE))
        target = torch.randint(high=num_classes, size=(NUM_TOTAL_UPDATES, BATCH_SIZE))

        self.run_class_implementation_tests(
            metric=BinaryF1Score(),
            state_names={"num_tp", "num_label", "num_prediction"},
            update_kwargs={"input": input, "target": target},
            compute_result=expected_result,
        )

        # test target all 0s
        input = torch.randint(high=num_classes, size=(NUM_TOTAL_UPDATES, BATCH_SIZE))
        target = torch.zeros(size=(NUM_TOTAL_UPDATES, BATCH_SIZE))

        self.run_class_implementation_tests(
            metric=BinaryF1Score(),
            state_names={"num_tp", "num_label", "num_prediction"},
            update_kwargs={"input": input, "target": target},
            compute_result=expected_result,
        )

    def test_binary_f1_score_thresholding(self) -> None:
        num_classes = 2
        threshold = 2

        # test threshold larger than every prediction gives 0
        input = torch.randint(high=num_classes, size=(NUM_TOTAL_UPDATES, BATCH_SIZE))
        target = torch.randint(high=num_classes, size=(NUM_TOTAL_UPDATES, BATCH_SIZE))
        expected_result = torch.Tensor([0.0]).sum()
        self.run_class_implementation_tests(
            metric=BinaryF1Score(threshold=threshold),
            state_names={"num_tp", "num_label", "num_prediction"},
            update_kwargs={"input": input, "target": target},
            compute_result=expected_result,
        )

    def test_binary_f1_score_class_update_input_shape_different(self) -> None:
        num_classes = 2
        update_input = [
            torch.randint(high=num_classes, size=(5,)),
            torch.randint(high=num_classes, size=(8,)),
            torch.randint(high=num_classes, size=(2,)),
            torch.randint(high=num_classes, size=(5,)),
        ]

        update_target = [
            torch.randint(high=num_classes, size=(5,)),
            torch.randint(high=num_classes, size=(8,)),
            torch.randint(high=num_classes, size=(2,)),
            torch.randint(high=num_classes, size=(5,)),
        ]

        compute_result = (
            torch.tensor(
                f1_score(
                    torch.cat(update_target, dim=0),
                    torch.cat(update_input, dim=0),
                    average="binary",
                )
            )
            .to(torch.float32)
            .squeeze()
        )
        print(f"Expected result {compute_result}")

        self.run_class_implementation_tests(
            metric=BinaryF1Score(),
            state_names={"num_tp", "num_label", "num_prediction"},
            update_kwargs={
                "input": update_input,
                "target": update_target,
            },
            compute_result=compute_result,
            num_total_updates=4,
            num_processes=2,
        )

    def test_binary_f1_score_invalid_input(self) -> None:
        metric = BinaryF1Score()

        with self.assertRaisesRegex(
            ValueError,
            "input should be a one-dimensional tensor for binary f1 score, "
            r"got shape torch.Size\(\[4, 2\]\).",
        ):
            metric.update(torch.rand(4, 2), torch.rand(3))

        with self.assertRaisesRegex(
            ValueError,
            "target should be a one-dimensional tensor for binary f1 score, "
            r"got shape torch.Size\(\[4, 2\]\).",
        ):
            metric.update(torch.rand(3), torch.rand(4, 2))

        with self.assertRaisesRegex(
            ValueError,
            "The `input` and `target` should have the same dimensions, "
            r"got shapes torch.Size\(\[11\]\) and torch.Size\(\[10\]\).",
        ):
            metric.update(torch.rand(11), torch.rand(10))


class TestMulticlassF1Score(MetricClassTester):
    def _test_f1_score_class_with_input(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        num_classes: int | None = None,
        average: str | None = "micro",
    ) -> None:
        input_tensors = [torch.argmax(t, dim=1) if t.ndim == 2 else t for t in input]
        target_tensors = list(target)
        target_np = torch.stack(target_tensors).flatten().numpy()
        input_np = torch.stack(input_tensors).flatten().numpy()
        compute_result = torch.tensor(
            f1_score(target_np, input_np, average=average), dtype=torch.float32
        )
        if num_classes is None:
            if average == "micro":
                self.run_class_implementation_tests(
                    metric=MulticlassF1Score(),
                    state_names={"num_tp", "num_label", "num_prediction"},
                    update_kwargs={
                        "input": input,
                        "target": target,
                    },
                    compute_result=compute_result,
                )
            else:
                if average is None:
                    if compute_result.shape[0] != num_classes:
                        compute_result = torch.from_numpy(
                            np.append(compute_result.numpy(), 0.0)
                        ).to(torch.float32)
                self.run_class_implementation_tests(
                    metric=MulticlassF1Score(average=average),
                    state_names={"num_tp", "num_label", "num_prediction"},
                    update_kwargs={
                        "input": input,
                        "target": target,
                    },
                    compute_result=compute_result,
                )
        else:
            if average == "micro":
                self.run_class_implementation_tests(
                    metric=MulticlassF1Score(num_classes=num_classes),
                    state_names={"num_tp", "num_label", "num_prediction"},
                    update_kwargs={
                        "input": input,
                        "target": target,
                    },
                    compute_result=compute_result,
                )
            else:
                if average is None:
                    if compute_result.shape[0] != num_classes:
                        compute_result = torch.from_numpy(
                            np.append(compute_result.numpy(), 0.0)
                        ).to(torch.float32)
                self.run_class_implementation_tests(
                    metric=MulticlassF1Score(num_classes=num_classes, average=average),
                    state_names={"num_tp", "num_label", "num_prediction"},
                    update_kwargs={
                        "input": input,
                        "target": target,
                    },
                    compute_result=compute_result,
                )

    def test_f1_score_class_base(self) -> None:
        num_classes = 4
        input = torch.randint(high=num_classes, size=(NUM_TOTAL_UPDATES, BATCH_SIZE))
        target = torch.randint(high=num_classes, size=(NUM_TOTAL_UPDATES, BATCH_SIZE))
        self._test_f1_score_class_with_input(input, target)

        input = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE, num_classes)
        self._test_f1_score_class_with_input(input, target)

    def test_f1_score_class_average(self) -> None:
        num_classes = 4
        input = torch.randint(high=num_classes, size=(NUM_TOTAL_UPDATES, BATCH_SIZE))
        target = torch.randint(high=num_classes, size=(NUM_TOTAL_UPDATES, BATCH_SIZE))
        self._test_f1_score_class_with_input(input, target, num_classes, average=None)

        input = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE, num_classes)
        self._test_f1_score_class_with_input(input, target, num_classes, average=None)

        input = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE, num_classes)
        self._test_f1_score_class_with_input(
            input, target, num_classes, average="macro"
        )

        input = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE, num_classes)
        self._test_f1_score_class_with_input(
            input, target, num_classes, average="weighted"
        )

    def test_f1_score_class_label_not_exist(self) -> None:
        num_classes = 4
        input = torch.randint(high=num_classes, size=(NUM_TOTAL_UPDATES, BATCH_SIZE))
        target = torch.randint(
            high=num_classes - 1, size=(NUM_TOTAL_UPDATES, BATCH_SIZE)
        )
        self._test_f1_score_class_with_input(input, target)

        input = torch.randint(
            high=num_classes - 1, size=(NUM_TOTAL_UPDATES, BATCH_SIZE)
        )
        target = torch.randint(
            high=num_classes - 1, size=(NUM_TOTAL_UPDATES, BATCH_SIZE)
        )
        self._test_f1_score_class_with_input(
            input,
            target,
            num_classes=num_classes,
            average="macro",
        )

        input = torch.randint(
            high=num_classes - 1, size=(NUM_TOTAL_UPDATES, BATCH_SIZE)
        )
        target = torch.randint(high=num_classes, size=(NUM_TOTAL_UPDATES, BATCH_SIZE))
        self._test_f1_score_class_with_input(
            input, target, num_classes=num_classes, average=None
        )

        input = torch.randint(
            high=num_classes - 1, size=(NUM_TOTAL_UPDATES, BATCH_SIZE)
        )
        target = torch.randint(
            high=num_classes - 1, size=(NUM_TOTAL_UPDATES, BATCH_SIZE)
        )
        self._test_f1_score_class_with_input(
            input, target, num_classes=num_classes, average=None
        )

    def test_f1_score_class_update_input_shape_different(self) -> None:
        num_classes = 4
        update_input = [
            torch.randint(high=num_classes, size=(5,)),
            torch.randint(high=num_classes, size=(8,)),
            torch.randint(high=num_classes, size=(2,)),
            torch.randint(high=num_classes, size=(5,)),
        ]

        update_target = [
            torch.randint(high=num_classes, size=(5,)),
            torch.randint(high=num_classes, size=(8,)),
            torch.randint(high=num_classes, size=(2,)),
            torch.randint(high=num_classes, size=(5,)),
        ]
        self.run_class_implementation_tests(
            metric=MulticlassF1Score(),
            state_names={"num_tp", "num_label", "num_prediction"},
            update_kwargs={
                "input": update_input,
                "target": update_target,
            },
            compute_result=(
                torch.tensor(
                    f1_score(
                        torch.cat(update_target, dim=0),
                        torch.cat(update_input, dim=0),
                        average="micro",
                    )
                )
                .to(torch.float32)
                .squeeze()
            ),
            num_total_updates=4,
            num_processes=2,
        )

    def test_f1_score_class_invalid_input(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "`average` was not in the allowed value of .*, got gaussian."
        ):
            MulticlassF1Score(num_classes=2, average="gaussian")

        with self.assertRaisesRegex(
            ValueError,
            r"num_classes should be a positive number when average=None, "
            r"got num_classes=None",
        ):
            MulticlassF1Score(average=None)

        metric = MulticlassF1Score(num_classes=4)
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
