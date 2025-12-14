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

from torcheval.metrics import BinaryAUROC, MulticlassAUROC
from torcheval.utils.test_utils.metric_class_tester import (
    BATCH_SIZE,
    MetricClassTester,
    NUM_TOTAL_UPDATES,
)


class TestBinaryAUROC(MetricClassTester):
    def _test_auroc_class_with_input(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        num_tasks: int = 1,
        weight: torch.Tensor | None = None,
        compute_result: torch.Tensor | None = None,
        use_fbgemm: bool | None = False,
    ) -> None:
        input_tensors = input.reshape(-1, 1)
        target_tensors = target.reshape(-1, 1)
        weight_tensors = weight.reshape(-1, 1) if weight is not None else None

        if compute_result is None:
            score = (
                roc_auc_score(target_tensors, input_tensors, sample_weight=weight_tensors)
                if weight_tensors is not None
                else roc_auc_score(target_tensors, input_tensors)
            )
            compute_result = torch.tensor(score, dtype=torch.float64)




        if weight is not None:
            self.run_class_implementation_tests(
                metric=BinaryAUROC(num_tasks=num_tasks, use_fbgemm=use_fbgemm),
                state_names={"inputs", "targets", "weights"},
                update_kwargs={
                    "input": input,
                    "target": target,
                    "weight": weight,
                },
                compute_result=compute_result,
                test_devices=["cuda"] if use_fbgemm else None,
            )
        else:
            self.run_class_implementation_tests(
                metric=BinaryAUROC(num_tasks=num_tasks, use_fbgemm=use_fbgemm),
                state_names={"inputs", "targets", "weights"},
                update_kwargs={
                    "input": input,
                    "target": target,
                },
                compute_result=compute_result,
                test_devices=["cuda"] if use_fbgemm else None,
            )

    def _test_auroc_class_set(self, use_fbgemm: bool | None = False) -> None:
        input = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE)
        target = torch.randint(high=2, size=(NUM_TOTAL_UPDATES, BATCH_SIZE))
        # fbgemm version does not support weight in AUROC calculation
        weight = (
            torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE) if use_fbgemm is False else None
        )
        self._test_auroc_class_with_input(
            input, target, num_tasks=1, weight=weight, use_fbgemm=use_fbgemm
        )

        if not use_fbgemm:
            # Skip this test for use_fbgemm because FBGEMM AUC is an
            # approximation of AUC. It can give a significantly different
            # result if input data is highly redundant
            input = torch.randint(high=2, size=(NUM_TOTAL_UPDATES, BATCH_SIZE))
            target = torch.randint(high=2, size=(NUM_TOTAL_UPDATES, BATCH_SIZE))
            weight = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE)
            self._test_auroc_class_with_input(
                input,
                target,
                num_tasks=1,
                weight=weight,
                use_fbgemm=use_fbgemm,
            )

        num_tasks = 2
        torch.manual_seed(123)
        input = torch.rand(NUM_TOTAL_UPDATES, num_tasks, BATCH_SIZE)
        target = torch.randint(high=2, size=(NUM_TOTAL_UPDATES, num_tasks, BATCH_SIZE))
        self._test_auroc_class_with_input(
            input,
            target,
            num_tasks=2,
            compute_result=torch.tensor(
                [0.549048678033111, 0.512218963831867], dtype=torch.float64
            ),
            use_fbgemm=use_fbgemm,
        )

    @unittest.skipUnless(
        condition=torch.cuda.is_available(), reason="This test needs a GPU host to run."
    )
    def test_auroc_class_fbgemm(self) -> None:
        self._test_auroc_class_set(use_fbgemm=True)

    def test_auroc_class_base(self) -> None:
        self._test_auroc_class_set(use_fbgemm=False)

    def test_auroc_class_update_input_shape_different(self) -> None:
        num_classes = 2
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

        update_weight = [
            torch.rand(5),
            torch.rand(8),
            torch.rand(2),
            torch.rand(5),
        ]

        compute_result = torch.tensor(
            roc_auc_score(
                torch.cat(update_target, dim=0),
                torch.cat(update_input, dim=0),
                sample_weight=torch.cat(update_weight, dim=0),

            ),
                dtype=torch.float64,

        )

        self.run_class_implementation_tests(
            metric=BinaryAUROC(),
            state_names={"inputs", "targets", "weights"},
            update_kwargs={
                "input": update_input,
                "target": update_target,
                "weight": update_weight,
            },
            compute_result=compute_result,
            num_total_updates=4,
            num_processes=2,
        )

    def test_binary_auroc_class_invalid_input(self) -> None:
        metric = BinaryAUROC()
        with self.assertRaisesRegex(
            ValueError,
            "The `input` and `target` should have the same shape, "
            r"got shapes torch.Size\(\[4\]\) and torch.Size\(\[3\]\).",
        ):
            metric.update(torch.rand(4), torch.rand(3))

        with self.assertRaisesRegex(
            ValueError,
            "The `weight` and `target` should have the same shape, "
            r"got shapes torch.Size\(\[3\]\) and torch.Size\(\[4\]\).",
        ):
            metric.update(torch.rand(4), torch.rand(4), weight=torch.rand(3))

        with self.assertRaisesRegex(
            ValueError,
            "`num_tasks = 1`, `input` is expected to be one-dimensional tensor,",
        ):
            metric.update(torch.rand(4, 5), torch.rand(4, 5))

        with self.assertRaisesRegex(
            ValueError,
            "`num_tasks = 2`, `input`'s shape is expected to be",
        ):
            metric = BinaryAUROC(num_tasks=2)
            metric.update(torch.rand(4, 5), torch.rand(4, 5))

        with self.assertRaisesRegex(ValueError, "`num_tasks` value should be greater"):
            metric = BinaryAUROC(num_tasks=0)


class TestMulticlassAUROC(MetricClassTester):
    def test_auroc_class_base(self) -> None:
        num_classes = 4
        input = 10 * torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE, num_classes)
        input = input.abs() / input.abs().sum(dim=-1, keepdim=True)
        target = torch.randint(high=num_classes, size=(NUM_TOTAL_UPDATES, BATCH_SIZE))

        input_tensors = input.reshape(-1, num_classes)
        target_tensors = target.reshape(-1)
        compute_result = torch.tensor(
            roc_auc_score(
                target_tensors, input_tensors, average="macro", multi_class="ovr"
            ),
            dtype=torch.float32,
        )

        self.run_class_implementation_tests(
            metric=MulticlassAUROC(num_classes=num_classes),
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

        input_tensors = input.reshape(-1, 3)
        target_tensors = target.reshape(-1)
        compute_result = torch.tensor(
            roc_auc_score(
                target_tensors, input_tensors, average="macro", multi_class="ovr"
            ),
            dtype=torch.float32,
        )

        self.run_class_implementation_tests(
            metric=MulticlassAUROC(num_classes=3, average="macro"),
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
            metric=MulticlassAUROC(num_classes=3, average=None),
            state_names={"inputs", "targets"},
            update_kwargs={
                "input": input,
                "target": target,
            },
            num_total_updates=4,
            num_processes=2,
            compute_result=torch.tensor([0.2500, 1.0000, 5 / 6]),
        )

    def test_auroc_class_update_input_shape_different(self) -> None:
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
        compute_result = torch.tensor(
            roc_auc_score(
                torch.cat(update_target, dim=0),
                torch.cat(update_input, dim=0),
                average="macro",
                multi_class="ovr",
            ),
            dtype=torch.float32,
        )

        self.run_class_implementation_tests(
            metric=MulticlassAUROC(num_classes=num_classes),
            state_names={"inputs", "targets"},
            update_kwargs={
                "input": update_input,
                "target": update_target,
            },
            compute_result=compute_result,
            num_total_updates=4,
            num_processes=2,
        )

    def test_multiclass_auroc_class_invalid_input(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "`average` was not in the allowed value of .*, got micro."
        ):
            MulticlassAUROC(num_classes=4, average="micro")

        with self.assertRaisesRegex(ValueError, "`num_classes` has to be at least 2."):
            MulticlassAUROC(num_classes=1)

        metric = MulticlassAUROC(num_classes=2)
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
            r"input should have shape of \(num_sample, num_classes\), "
            r"got torch.Size\(\[3, 4\]\) and num_classes=2.",
        ):
            metric.update(torch.rand(3, 4), torch.rand(3))
