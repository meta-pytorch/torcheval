# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import numpy as np
import torch
from sklearn.metrics import recall_score
from torcheval.metrics.classification import BinaryRecall, MulticlassRecall
from torcheval.utils.test_utils.metric_class_tester import (
    BATCH_SIZE,
    MetricClassTester,
    NUM_TOTAL_UPDATES,
)


class TestMulticlassRecall(MetricClassTester):
    def _test_recall_class_with_input(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        num_classes: int | None = None,
        average: str | None = "micro",
    ) -> None:
        input_tensors = [torch.argmax(t, dim=1) if t.ndim == 2 else t for t in input]
        target_tensors = list(target)
        input_np = torch.stack(input_tensors).flatten().numpy()
        target_np = torch.stack(target_tensors).flatten().numpy()
        compute_result = torch.tensor(
            recall_score(target_np, input_np, average=average), dtype=torch.float32
        )

        if num_classes is None:
            if average == "micro":
                self.run_class_implementation_tests(
                    metric=MulticlassRecall(),
                    state_names={"num_tp", "num_labels", "num_predictions"},
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
                    metric=MulticlassRecall(average=average),
                    state_names={"num_tp", "num_labels", "num_predictions"},
                    update_kwargs={
                        "input": input,
                        "target": target,
                    },
                    compute_result=compute_result,
                )
        else:
            if average == "micro":
                self.run_class_implementation_tests(
                    metric=MulticlassRecall(num_classes=num_classes),
                    state_names={"num_tp", "num_labels", "num_predictions"},
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
                    metric=MulticlassRecall(num_classes=num_classes, average=average),
                    state_names={"num_tp", "num_labels", "num_predictions"},
                    update_kwargs={
                        "input": input,
                        "target": target,
                    },
                    compute_result=compute_result,
                )

    def test_recall_class_base(self) -> None:
        num_classes = 4
        input = torch.randint(high=num_classes, size=(NUM_TOTAL_UPDATES, BATCH_SIZE))
        target = torch.randint(high=num_classes, size=(NUM_TOTAL_UPDATES, BATCH_SIZE))
        self._test_recall_class_with_input(input, target)

        input = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE, num_classes)
        self._test_recall_class_with_input(input, target)

    def test_recall_class_average(self) -> None:
        num_classes = 4
        input = torch.randint(high=num_classes, size=(NUM_TOTAL_UPDATES, BATCH_SIZE))
        target = torch.randint(high=num_classes, size=(NUM_TOTAL_UPDATES, BATCH_SIZE))
        self._test_recall_class_with_input(input, target, num_classes, average=None)

        input = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE, num_classes)
        self._test_recall_class_with_input(input, target, num_classes, average=None)

        input = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE, num_classes)
        self._test_recall_class_with_input(input, target, num_classes, average="macro")

        input = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE, num_classes)
        self._test_recall_class_with_input(
            input, target, num_classes, average="weighted"
        )

    def test_recall_class_absent_labels(self) -> None:
        num_classes = 4
        input = torch.randint(high=num_classes, size=(NUM_TOTAL_UPDATES, BATCH_SIZE))
        target = torch.randint(
            high=num_classes - 1, size=(NUM_TOTAL_UPDATES, BATCH_SIZE)
        )
        self._test_recall_class_with_input(input, target)

        input = torch.randint(
            high=num_classes - 1, size=(NUM_TOTAL_UPDATES, BATCH_SIZE)
        )
        target = torch.randint(high=num_classes, size=(NUM_TOTAL_UPDATES, BATCH_SIZE))
        self._test_recall_class_with_input(input, target)

    def test_recall_class_nan_handling(self) -> None:
        num_classes = 4
        with self.assertLogs(level="WARNING") as logger:
            input = torch.randint(
                high=num_classes, size=(NUM_TOTAL_UPDATES, BATCH_SIZE)
            )
            target = torch.randint(
                high=num_classes - 1, size=(NUM_TOTAL_UPDATES, BATCH_SIZE)
            )
            self._test_recall_class_with_input(
                input, target, num_classes=num_classes, average=None
            )
            self.assertRegex(
                logger.output[0],
                "One or more NaNs identified, as no ground-truth instances of "
                ".* have been seen. These have been converted to zero.",
            )
        with self.assertLogs(level="WARNING") as logger:
            input = torch.randint(
                high=num_classes - 1, size=(NUM_TOTAL_UPDATES, BATCH_SIZE)
            )
            target = torch.randint(
                high=num_classes - 1, size=(NUM_TOTAL_UPDATES, BATCH_SIZE)
            )
            self._test_recall_class_with_input(
                input, target, num_classes=num_classes, average=None
            )
            self.assertRegex(
                logger.output[0],
                "One or more NaNs identified, as no ground-truth instances of "
                ".* have been seen. These have been converted to zero.",
            )

    def test_recall_class_update_input_different_shape(self) -> None:
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
            metric=MulticlassRecall(),
            state_names={"num_tp", "num_labels", "num_predictions"},
            update_kwargs={
                "input": update_input,
                "target": update_target,
            },
            compute_result=(
                torch.tensor(
                    recall_score(
                        torch.cat(update_target, dim=0),
                        torch.cat(update_input, dim=0),
                        average="micro",
                    )
                )
                .to(torch.float32)
                .squeeze()
            ),
            num_total_updates=4,
            num_processes=4,
        )

    def test_recall_class_invalid_input(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "`average` was not in the allowed values of .*, got binary"
        ):
            MulticlassRecall(num_classes=2, average="binary")
        with self.assertRaisesRegex(
            ValueError,
            r"`num_classes` should be a positive number when average=None, "
            r"got num_classes=None",
        ):
            MulticlassRecall(average=None)

        metric = MulticlassRecall(num_classes=4)
        with self.assertRaisesRegex(
            ValueError,
            r"The `input` and `target` should have the same first dimension, "
            r"got shapes torch.Size\(\[3, 4\]\) and torch.Size\(\[2\]\).",
        ):
            metric.update(torch.rand(3, 4), torch.rand(2))
        with self.assertRaisesRegex(
            ValueError,
            r"`target` should be a one-dimensional tensor, got shape torch.Size\(\[3, 4\]\).",
        ):
            metric.update(torch.rand(3, 4), torch.rand(3, 4))
        with self.assertRaisesRegex(
            ValueError,
            r"`input` should have shape \(num_samples,\) or \(num_samples, num_classes\), "
            r"got torch.Size\(\[3, 4, 5\]\).",
        ):
            metric.update(torch.rand(3, 4, 5), torch.rand(3))


class TestBinaryRecall(MetricClassTester):
    def _test_binary_recall_class_with_input(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        threshold: float = 0.5,
    ) -> None:
        input_np = np.where(input.numpy() < threshold, 0, 1)
        target_np = target.squeeze().numpy()

        sklearn_result = torch.tensor(
            recall_score(target_np.flatten(), input_np.flatten(), average="binary")
        ).to(torch.float32)

        self.run_class_implementation_tests(
            metric=BinaryRecall(),
            state_names={"num_tp", "num_true_labels"},
            update_kwargs={"input": input, "target": target},
            compute_result=sklearn_result,
        )

    def test_binary_recall_class_base(self) -> None:
        num_classes = 2
        input = torch.randint(high=num_classes, size=(NUM_TOTAL_UPDATES, BATCH_SIZE))
        target = torch.randint(high=num_classes, size=(NUM_TOTAL_UPDATES, BATCH_SIZE))
        self._test_binary_recall_class_with_input(input, target)

    def test_binary_recall_class_with_rounding(self) -> None:
        num_classes = 2
        input = torch.rand(size=(NUM_TOTAL_UPDATES, BATCH_SIZE))
        target = torch.randint(high=num_classes, size=(NUM_TOTAL_UPDATES, BATCH_SIZE))

        self._test_binary_recall_class_with_input(input, target)

    def test_binary_recall_class_invalid_input(self) -> None:
        metric = BinaryRecall()
        with self.assertRaisesRegex(
            ValueError,
            "The `input` and `target` should have the same dimensions, "
            r"got shapes torch.Size\(\[4, 2\]\) and torch.Size\(\[3\]\).",
        ):
            metric.update(torch.rand(4, 2), torch.rand(3))
