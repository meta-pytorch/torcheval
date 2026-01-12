# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# pyre-ignore-all-errors[56]: Pyre was not able to infer the type of argument

import torch
from sklearn.metrics import auc as sklearn_auc
from torcheval.metrics.aggregation.auc import AUC
from torcheval.utils.test_utils.metric_class_tester import (
    BATCH_SIZE,
    MetricClassTester,
    NUM_TOTAL_UPDATES,
)


class TestAUC(MetricClassTester):
    def test_auc_class_base(self) -> None:
        NUM_PROCESSES = 4
        NUM_TASKS = 1

        x = [torch.rand(NUM_TASKS, BATCH_SIZE) for i in range(NUM_TOTAL_UPDATES)]
        y = [
            torch.randint(high=2, size=(NUM_TASKS, BATCH_SIZE))
            for i in range(NUM_TOTAL_UPDATES)
        ]

        # Build sklearn metrics
        sk_x, sk_y = torch.cat(x, dim=1), torch.cat(y, dim=1)
        sk_x, sk_x_idx = torch.sort(sk_x, dim=1, stable=True)
        sk_y = sk_y.gather(1, sk_x_idx)
        compute_result = torch.tensor(
            [sklearn_auc(temp_x, temp_y) for temp_x, temp_y in zip(sk_x, sk_y)],
            dtype=torch.float32,
        )

        self.run_class_implementation_tests(
            metric=AUC(n_tasks=NUM_TASKS),
            state_names={"x", "y"},
            update_kwargs={
                "x": x,
                "y": y,
            },
            compute_result=compute_result,
            num_total_updates=NUM_TOTAL_UPDATES,
            num_processes=NUM_PROCESSES,
        )

    def test_auc_class_n_tasks_3(self) -> None:
        NUM_PROCESSES = 4
        NUM_TASKS = 3

        x = [torch.rand(NUM_TASKS, BATCH_SIZE) for i in range(NUM_TOTAL_UPDATES)]
        y = [
            torch.randint(high=2, size=(NUM_TASKS, BATCH_SIZE))
            for i in range(NUM_TOTAL_UPDATES)
        ]

        # Build sklearn metrics
        sk_x, sk_y = torch.cat(x, dim=1), torch.cat(y, dim=1)
        sk_x, sk_x_idx = torch.sort(sk_x, dim=1, stable=True)
        sk_y = sk_y.gather(1, sk_x_idx)
        compute_result = torch.tensor(
            [sklearn_auc(temp_x, temp_y) for temp_x, temp_y in zip(sk_x, sk_y)],
            dtype=torch.float32,
        )

        self.run_class_implementation_tests(
            metric=AUC(n_tasks=NUM_TASKS),
            state_names={"x", "y"},
            update_kwargs={
                "x": x,
                "y": y,
            },
            compute_result=compute_result,
            num_total_updates=NUM_TOTAL_UPDATES,
            num_processes=NUM_PROCESSES,
        )

    def test_auc_class_num_no_reorder(self) -> None:
        num_total_updates = 6
        num_processes = 3
        x = [
            torch.tensor([0.02, 0.05, 0.08]).unsqueeze(0),
            torch.tensor([0.16, 0.17, 0.18]).unsqueeze(0),
            torch.tensor([0.23, 0.25, 0.28]).unsqueeze(0),
            torch.tensor([0.33, 0.35, 0.39]).unsqueeze(0),
            torch.tensor([0.43, 0.44, 0.49]).unsqueeze(0),
            torch.tensor([0.52, 0.57, 0.58]).unsqueeze(0),
        ]
        y = [
            torch.tensor([2, 0, 1]).unsqueeze(0),
            torch.tensor([0, 1, 0]).unsqueeze(0),
            torch.tensor([1, 1, 1]).unsqueeze(0),
            torch.tensor([2, 2, 2]).unsqueeze(0),
            torch.tensor([1, 1, 1]).unsqueeze(0),
            torch.tensor([2, 0, 1]).unsqueeze(0),
        ]

        sk_x, sk_y = torch.cat(x, dim=1), torch.cat(y, dim=1)
        compute_result = torch.tensor(
            [sklearn_auc(sk_x[0], sk_y[0])], dtype=torch.float32
        )

        self.run_class_implementation_tests(
            metric=AUC(reorder=False),
            state_names={"x", "y"},
            update_kwargs={
                "x": x,
                "y": y,
            },
            compute_result=compute_result,
            num_total_updates=num_total_updates,
            num_processes=num_processes,
        )

    def test_auc_class_no_unsqueeze(self) -> None:
        num_total_updates = 6
        num_processes = 3

        x = [
            torch.tensor([0.02, 0.05, 0.08]),
            torch.tensor([0.16, 0.17, 0.18]),
            torch.tensor([0.23, 0.25, 0.28]),
            torch.tensor([0.33, 0.35, 0.39]),
            torch.tensor([0.43, 0.44, 0.49]),
            torch.tensor([0.52, 0.57, 0.58]),
        ]
        y = [
            torch.tensor([2, 0, 1]),
            torch.tensor([0, 1, 0]),
            torch.tensor([1, 1, 1]),
            torch.tensor([2, 2, 2]),
            torch.tensor([1, 1, 1]),
            torch.tensor([2, 0, 1]),
        ]

        # Build sklearn metrics
        sk_x, sk_y = (
            torch.cat([tx.unsqueeze(0) for tx in x], dim=1),
            torch.cat([ty.unsqueeze(0) for ty in y], dim=1),
        )
        compute_result = torch.tensor(
            [sklearn_auc(temp_x, temp_y) for temp_x, temp_y in zip(sk_x, sk_y)],
            dtype=torch.float32,
        )

        self.run_class_implementation_tests(
            metric=AUC(),
            state_names={"x", "y"},
            update_kwargs={
                "x": x,
                "y": y,
            },
            compute_result=compute_result,
            num_total_updates=num_total_updates,
            num_processes=num_processes,
        )

    def test_auc_class_invalid_input(self) -> None:
        metric = AUC()
        with self.assertRaisesRegex(
            ValueError,
            r"Expected the same shape in `x` and `y` tensor but got shapes torch.Size\(\[4, 2\]\) and torch.Size\(\[3\]\).",
        ):
            metric.update(torch.rand(4, 2), torch.rand(3))

        with self.assertRaisesRegex(
            ValueError,
            r"Expected the same shape in `x` and `y` tensor but got shapes torch.Size\(\[3, 2\]\) and torch.Size\(\[3, 3\]\).",
        ):
            metric.update(torch.rand(3, 2), torch.rand(3, 3))

        with self.assertRaisesRegex(
            ValueError,
            "The `x` and `y` should have atleast 1 element, "
            r"got shapes torch.Size\(\[0\]\) and torch.Size\(\[0\]\).",
        ):
            metric.update(torch.tensor([]), torch.tensor([]))

        with self.assertRaisesRegex(
            ValueError,
            r"Expected `x` dim_1=2 and `y` dim_1=2 have first dimension equals to n_tasks=1.",
        ):
            metric.update(torch.ones(3), torch.ones(3))
            metric.update(torch.ones(2, 4), torch.ones(2, 4))

        with self.assertRaisesRegex(
            ValueError,
            r"Expected `x` dim_1=4 and `y` dim_1=4 have first dimension equals to n_tasks=3.",
        ):
            metric1 = AUC(n_tasks=3)
            metric1.update(torch.ones(4, 2), torch.ones(4, 2))
