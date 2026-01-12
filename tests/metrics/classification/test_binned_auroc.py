# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import torch
from torcheval.metrics import BinaryBinnedAUROC, MulticlassBinnedAUROC
from torcheval.utils.test_utils.metric_class_tester import (
    BATCH_SIZE,
    MetricClassTester,
    NUM_TOTAL_UPDATES,
)


class TestBinaryBinnedAUROC(MetricClassTester):
    def _test_auroc_class_with_input(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        num_tasks: int,
        threshold: int | list[float] | torch.Tensor,
        compute_result: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        self.run_class_implementation_tests(
            metric=BinaryBinnedAUROC(num_tasks=num_tasks, threshold=threshold),
            state_names={"inputs", "targets"},
            update_kwargs={
                "input": input,
                "target": target,
            },
            compute_result=compute_result,
        )

    def test_auroc_class_valid_input(self) -> None:
        torch.manual_seed(123)
        # test case with num_task=1
        input = torch.randint(high=2, size=(NUM_TOTAL_UPDATES, BATCH_SIZE))
        target = torch.randint(high=2, size=(NUM_TOTAL_UPDATES, BATCH_SIZE))
        threshold = 5
        self._test_auroc_class_with_input(
            input,
            target,
            num_tasks=1,
            threshold=threshold,
            compute_result=(
                torch.tensor([0.5078144078144078], dtype=torch.float64),
                torch.tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000]),
            ),
        )

        # test case with num_task=2
        torch.manual_seed(123)
        num_tasks = 2
        input = torch.rand(NUM_TOTAL_UPDATES, num_tasks, BATCH_SIZE)
        target = torch.randint(high=2, size=(NUM_TOTAL_UPDATES, num_tasks, BATCH_SIZE))
        self._test_auroc_class_with_input(
            input,
            target,
            num_tasks=2,
            threshold=threshold,
            compute_result=(
                torch.tensor(
                    [0.5406473931307141, 0.49572336265884653], dtype=torch.float64
                ),
                torch.tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000]),
            ),
        )

        # test case with different update shape
        num_classes = 2
        threshold = 5
        torch.manual_seed(123)
        update_input = [
            torch.rand(5),
            torch.rand(8),
            torch.rand(2),
            torch.rand(5),
        ]

        update_target = [
            torch.randint(high=num_classes, size=(5,)),
            torch.randint(high=num_classes, size=(8,)),
            torch.randint(high=num_classes, size=(2,)),
            torch.randint(high=num_classes, size=(5,)),
        ]
        compute_result = (
            torch.tensor([0.3383838383838384], dtype=torch.float64),
            torch.tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000]),
        )

        self.run_class_implementation_tests(
            metric=BinaryBinnedAUROC(threshold=threshold),
            state_names={"inputs", "targets"},
            update_kwargs={
                "input": update_input,
                "target": update_target,
            },
            compute_result=compute_result,
            num_total_updates=4,
            num_processes=2,
        )

    def test_auroc_class_invalid_input(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "`num_tasks` has to be at least 1.",
        ):
            metric = BinaryBinnedAUROC(num_tasks=-1)

        with self.assertRaisesRegex(
            ValueError,
            "The `input` and `target` should have the same shape, "
            r"got shapes torch.Size\(\[4\]\) and torch.Size\(\[3\]\).",
        ):
            metric = BinaryBinnedAUROC()
            metric.update(torch.rand(4), torch.rand(3))

        with self.assertRaisesRegex(
            ValueError,
            "`num_tasks = 1`, `input` is expected to be one-dimensional tensor, "
            r"but got shape torch.Size\(\[3, 2\]\).",
        ):
            metric = BinaryBinnedAUROC()
            metric.update(torch.rand(3, 2), torch.rand(3, 2))

        with self.assertRaisesRegex(
            ValueError, "The `threshold` should be a sorted tensor."
        ):
            metric = BinaryBinnedAUROC(
                threshold=torch.tensor([0.1, 0.2, 0.5, 0.7, 0.6])
            )

        with self.assertRaisesRegex(
            ValueError,
            r"The values in `threshold` should be in the range of \[0, 1\].",
        ):
            metric = BinaryBinnedAUROC(threshold=torch.tensor([-0.1, 0.2, 0.5, 0.7]))


class TestMulticlassBinnedAUROC(MetricClassTester):
    def test_auroc_class_base(self) -> None:
        num_classes = 4
        threshold = 5
        torch.manual_seed(123)
        input = 10 * torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE, num_classes)
        input = input.abs() / input.abs().sum(dim=-1, keepdim=True)
        target = torch.randint(high=num_classes, size=(NUM_TOTAL_UPDATES, BATCH_SIZE))

        compute_result = (
            torch.tensor(0.5013020634651184),
            torch.tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000]),
        )

        self.run_class_implementation_tests(
            metric=MulticlassBinnedAUROC(num_classes=num_classes, threshold=threshold),
            state_names={"inputs", "targets"},
            update_kwargs={
                "input": input,
                "target": target,
            },
            compute_result=compute_result,
        )

    def test_auroc_average_options(self) -> None:
        input = torch.tensor(
            [
                [[0.16, 0.04, 0.8]],
                [[0.1, 0.7, 0.2]],
                [[0.16, 0.8, 0.04]],
                [[0.16, 0.04, 0.8]],
            ]
        )
        target = torch.tensor([[0], [0], [1], [2]])

        compute_result = (
            torch.tensor(0.625),
            torch.tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000]),
        )

        self.run_class_implementation_tests(
            metric=MulticlassBinnedAUROC(num_classes=3, threshold=5, average="macro"),
            state_names={"inputs", "targets"},
            update_kwargs={
                "input": input,
                "target": target,
            },
            num_total_updates=4,
            num_processes=2,
            compute_result=compute_result,
        )

        self.run_class_implementation_tests(
            metric=MulticlassBinnedAUROC(num_classes=3, threshold=5, average=None),
            state_names={"inputs", "targets"},
            update_kwargs={
                "input": input,
                "target": target,
            },
            num_total_updates=4,
            num_processes=2,
            compute_result=(
                torch.tensor([0.25, 0.25, 1.0, 1.0]),
                torch.tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000]),
            ),
        )

    def test_auroc_class_update_input_shape_different(self) -> None:
        torch.manual_seed(123)
        num_classes = 3
        update_input = [
            torch.rand(5, num_classes),
            torch.rand(8, num_classes),
            torch.rand(2, num_classes),
            torch.rand(5, num_classes),
        ]
        update_input = [
            input.abs() / input.abs().sum(dim=-1, keepdim=True)
            for input in update_input
        ]
        update_target = [
            torch.randint(high=num_classes, size=(5,)),
            torch.randint(high=num_classes, size=(8,)),
            torch.randint(high=num_classes, size=(2,)),
            torch.randint(high=num_classes, size=(5,)),
        ]
        compute_result = (
            torch.tensor(0.5),
            torch.tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000]),
        )

        self.run_class_implementation_tests(
            metric=MulticlassBinnedAUROC(num_classes=num_classes, threshold=5),
            state_names={"inputs", "targets"},
            update_kwargs={
                "input": update_input,
                "target": update_target,
            },
            compute_result=compute_result,
            num_total_updates=4,
            num_processes=2,
        )

    def test_auroc_class_invalid_input(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "`average` was not in the allowed value of .*, got micro."
        ):
            metric = MulticlassBinnedAUROC(num_classes=4, average="micro")

        with self.assertRaisesRegex(ValueError, "`num_classes` has to be at least 2."):
            metric = MulticlassBinnedAUROC(num_classes=1)

        with self.assertRaisesRegex(
            ValueError,
            "The `input` and `target` should have the same first dimension, "
            r"got shapes torch.Size\(\[4, 2\]\) and torch.Size\(\[3\]\).",
        ):
            metric = MulticlassBinnedAUROC(num_classes=3)
            metric.update(torch.rand(4, 2), torch.rand(3))

        with self.assertRaisesRegex(
            ValueError,
            "target should be a one-dimensional tensor, "
            r"got shape torch.Size\(\[3, 2\]\).",
        ):
            metric = MulticlassBinnedAUROC(num_classes=2)
            metric.update(torch.rand(3, 2), torch.rand(3, 2))

        with self.assertRaisesRegex(
            ValueError,
            r"input should have shape of \(num_sample, num_classes\), "
            r"got torch.Size\(\[3, 4\]\) and num_classes=2.",
        ):
            metric = MulticlassBinnedAUROC(num_classes=2)
            metric.update(torch.rand(3, 4), torch.rand(3))

        with self.assertRaisesRegex(
            ValueError, "The `threshold` should be a sorted tensor."
        ):
            metric = MulticlassBinnedAUROC(
                num_classes=4, threshold=torch.tensor([0.1, 0.2, 0.5, 0.7, 0.6])
            )

        with self.assertRaisesRegex(
            ValueError,
            r"The values in `threshold` should be in the range of \[0, 1\].",
        ):
            metric = MulticlassBinnedAUROC(
                num_classes=4, threshold=torch.tensor([-0.1, 0.2, 0.5, 0.7])
            )
