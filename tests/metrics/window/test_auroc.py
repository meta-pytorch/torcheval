# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import torch
from sklearn.metrics import roc_auc_score
from torcheval.metrics import WindowedBinaryAUROC
from torcheval.metrics.functional import binary_auroc
from torcheval.utils.test_utils.metric_class_tester import (
    BATCH_SIZE,
    MetricClassTester,
    NUM_TOTAL_UPDATES,
)


class TestWindowedBinaryAUROC(MetricClassTester):
    def _test_auroc_class_with_input(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor | None = None,
        max_num_samples: int = 100,
    ) -> None:
        input_tensors = input.reshape(-1)[-max_num_samples:]
        target_tensors = target.reshape(-1)[-max_num_samples:]
        weight_tensors = (
            weight.reshape(-1)[-max_num_samples:] if weight is not None else None
        )
        compute_result = torch.tensor(
            roc_auc_score(target_tensors, input_tensors, sample_weight=weight_tensors)
            if weight_tensors is not None
            else roc_auc_score(target_tensors, input_tensors)
        )

        input_tensors = input.reshape(-1)
        target_tensors = target.reshape(-1)
        weight_tensors = weight.reshape(-1) if weight is not None else None

        input_tensors = torch.cat(
            [
                input_tensors[22:32],
                input_tensors[54:64],
                input_tensors[86:96],
                input_tensors[118:],
            ]
        )
        target_tensors = torch.cat(
            [
                target_tensors[22:32],
                target_tensors[54:64],
                target_tensors[86:96],
                target_tensors[118:],
            ]
        )
        weight_tensors = (
            torch.cat(
                [
                    weight_tensors[22:32],
                    weight_tensors[54:64],
                    weight_tensors[86:96],
                    weight_tensors[118:],
                ]
            )
            if weight_tensors is not None
            else None
        )
        merge_compute_result = torch.tensor(
            roc_auc_score(target_tensors, input_tensors, sample_weight=weight_tensors)
            if weight is not None
            else roc_auc_score(target_tensors, input_tensors)
        )
        self.run_class_implementation_tests(
            metric=WindowedBinaryAUROC(max_num_samples=max_num_samples),
            state_names={
                "max_num_samples",
                "total_samples",
                "inputs",
                "targets",
                "weights",
            },
            update_kwargs={
                "input": input,
                "target": target,
                "weight": weight,
            },
            compute_result=compute_result,
            merge_and_compute_result=merge_compute_result,
            # merge will change window size, so combining merge and update
            # will have different result compared to update only.
            test_merge_with_one_update=False,
        )

    def test_auroc_class_base(self) -> None:
        input = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE)
        target = torch.randint(high=2, size=(NUM_TOTAL_UPDATES, BATCH_SIZE))
        weight = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE)
        self._test_auroc_class_with_input(input, target, weight, 10)

        input = torch.randint(high=2, size=(NUM_TOTAL_UPDATES, BATCH_SIZE))
        target = torch.randint(high=2, size=(NUM_TOTAL_UPDATES, BATCH_SIZE))
        weight = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE)
        self._test_auroc_class_with_input(input, target, weight, 10)

    def test_auroc_class_multiple_tasks(self) -> None:
        num_tasks = 2
        max_num_samples = 10
        input = torch.rand(NUM_TOTAL_UPDATES, num_tasks, BATCH_SIZE)
        target = torch.randint(high=2, size=(NUM_TOTAL_UPDATES, num_tasks, BATCH_SIZE))
        weight = torch.rand(NUM_TOTAL_UPDATES, num_tasks, BATCH_SIZE)

        input_tensors = input.permute(1, 0, 2).reshape(num_tasks, -1)[
            :, -max_num_samples:
        ]
        target_tensors = target.permute(1, 0, 2).reshape(num_tasks, -1)[
            :, -max_num_samples:
        ]
        weight_tensors = weight.permute(1, 0, 2).reshape(num_tasks, -1)[
            :, -max_num_samples:
        ]
        compute_result = binary_auroc(
            input_tensors, target_tensors, num_tasks=2, weight=weight_tensors
        )

        input_tensors = input.permute(1, 0, 2).reshape(num_tasks, -1)
        target_tensors = target.permute(1, 0, 2).reshape(num_tasks, -1)
        weight_tensors = weight.permute(1, 0, 2).reshape(num_tasks, -1)

        input_tensors = torch.cat(
            [
                input_tensors[:, 22:32],
                input_tensors[:, 54:64],
                input_tensors[:, 86:96],
                input_tensors[:, 118:],
            ],
            dim=1,
        )
        target_tensors = torch.cat(
            [
                target_tensors[:, 22:32],
                target_tensors[:, 54:64],
                target_tensors[:, 86:96],
                target_tensors[:, 118:],
            ],
            dim=1,
        )
        weight_tensors = torch.cat(
            [
                weight_tensors[:, 22:32],
                weight_tensors[:, 54:64],
                weight_tensors[:, 86:96],
                weight_tensors[:, 118:],
            ],
            dim=1,
        )
        merge_compute_result = binary_auroc(
            input_tensors, target_tensors, num_tasks=2, weight=weight_tensors
        )

        self.run_class_implementation_tests(
            metric=WindowedBinaryAUROC(
                num_tasks=num_tasks, max_num_samples=max_num_samples
            ),
            state_names={
                "max_num_samples",
                "total_samples",
                "inputs",
                "targets",
                "weights",
            },
            update_kwargs={
                "input": input,
                "target": target,
                "weight": weight,
            },
            compute_result=compute_result,
            merge_and_compute_result=merge_compute_result,
            # merge will change window size, so combining merge and update
            # will have different result compared to update only.
            test_merge_with_one_update=False,
        )

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

        compute_result = binary_auroc(
            torch.cat(update_input, dim=0)[-6:],
            torch.cat(update_target, dim=0)[-6:],
            weight=torch.cat(update_weight, dim=0)[-6:],
        )
        update_target_tensors = torch.cat(update_target, dim=0)
        update_input_tensors = torch.cat(update_input, dim=0)
        update_weight_tensors = torch.cat(update_weight, dim=0)
        merge_compute_result = binary_auroc(
            torch.cat([update_input_tensors[7:13], update_input_tensors[14:]], dim=0),
            torch.cat([update_target_tensors[7:13], update_target_tensors[14:]], dim=0),
            weight=torch.cat(
                [update_weight_tensors[7:13], update_weight_tensors[14:]], dim=0
            ),
        )

        self.run_class_implementation_tests(
            metric=WindowedBinaryAUROC(max_num_samples=6),
            state_names={
                "max_num_samples",
                "total_samples",
                "inputs",
                "targets",
                "weights",
            },
            update_kwargs={
                "input": update_input,
                "target": update_target,
                "weight": update_weight,
            },
            compute_result=compute_result,
            merge_and_compute_result=merge_compute_result,
            num_total_updates=4,
            num_processes=2,
            # merge will change window size, so combining merge and update
            # will have different result compared to update only.
            test_merge_with_one_update=False,
        )

    def test_binary_auroc_class_invalid_input(self) -> None:
        metric = WindowedBinaryAUROC()
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
            metric = WindowedBinaryAUROC(num_tasks=2)
            metric.update(torch.rand(4, 5), torch.rand(4, 5))

        with self.assertRaisesRegex(ValueError, "`num_tasks` value should be greater"):
            metric = WindowedBinaryAUROC(num_tasks=0)

        with self.assertRaisesRegex(
            ValueError, "`max_num_samples` value should be greater"
        ):
            metric = WindowedBinaryAUROC(max_num_samples=-1)
