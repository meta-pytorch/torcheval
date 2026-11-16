# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch
from torcheval.metrics import Cat
from torcheval.utils.test_utils.metric_class_tester import (
    BATCH_SIZE,
    MetricClassTester,
    NUM_TOTAL_UPDATES,
)


class TestCat(MetricClassTester):
    def _test_cat_class_with_input(
        self, input_val_tensors: list[torch.Tensor], dim: int = 0
    ) -> None:
        self.run_class_implementation_tests(
            metric=Cat(),
            state_names={"dim", "inputs"},
            update_kwargs={"input": input_val_tensors},
            compute_result=torch.cat(input_val_tensors, dim=dim),
        )

    def test_cat_class_base(self) -> None:
        input_val_tensor = [torch.rand(BATCH_SIZE) for _ in range(NUM_TOTAL_UPDATES)]
        self._test_cat_class_with_input(input_val_tensor)
        input_val_tensor = [torch.rand(BATCH_SIZE, 4) for _ in range(NUM_TOTAL_UPDATES)]
        self._test_cat_class_with_input(input_val_tensor)
        input_val_tensor = [
            torch.rand(BATCH_SIZE, 3, 4) for _ in range(NUM_TOTAL_UPDATES)
        ]
        self._test_cat_class_with_input(input_val_tensor)

    def test_cat_class_update_input_dim_0(self) -> None:
        update_inputs = [
            torch.tensor([[1, 2], [3, 4]], dtype=torch.float),
            torch.tensor([[5, 6], [7, 8]], dtype=torch.float),
            torch.tensor([[9, 10], [11, 12]], dtype=torch.float),
            torch.tensor([[13, 14], [15, 16]], dtype=torch.float),
        ]

        self.run_class_implementation_tests(
            metric=Cat(),
            state_names={"dim", "inputs"},
            update_kwargs={"input": update_inputs},
            compute_result=torch.cat(update_inputs, dim=0),
            num_total_updates=4,
            num_processes=2,
        )

    def test_cat_class_update_input_dim_1(self) -> None:
        update_inputs = [
            torch.tensor([[1, 2], [3, 4]], dtype=torch.float),
            torch.tensor([[5, 6], [7, 8]], dtype=torch.float),
            torch.tensor([[9, 10], [11, 12]], dtype=torch.float),
            torch.tensor([[13, 14], [15, 16]], dtype=torch.float),
        ]

        self.run_class_implementation_tests(
            metric=Cat(dim=1),
            state_names={"dim", "inputs"},
            update_kwargs={"input": update_inputs},
            compute_result=torch.cat(update_inputs, dim=1),
            num_total_updates=4,
            num_processes=2,
        )

    def test_cat_class_update_input_dim_minus_1(self) -> None:
        update_inputs = [
            torch.tensor([[1, 2], [3, 4]], dtype=torch.float),
            torch.tensor([[5, 6], [7, 8]], dtype=torch.float),
            torch.tensor([[9, 10], [11, 12]], dtype=torch.float),
            torch.tensor([[13, 14], [15, 16]], dtype=torch.float),
        ]

        self.run_class_implementation_tests(
            metric=Cat(dim=-1),
            state_names={"dim", "inputs"},
            update_kwargs={"input": update_inputs},
            compute_result=torch.cat(update_inputs, dim=-1),
            num_total_updates=4,
            num_processes=2,
        )

    def test_cat_class_update_input_shape_different(self) -> None:
        update_inputs = [
            torch.tensor([[1, 2], [3, 4]], dtype=torch.float),
            torch.tensor([[5, 6], [7, 8], [9, 10]], dtype=torch.float),
            torch.tensor([[11, 12]], dtype=torch.float),
            torch.tensor([[13, 14], [15, 16]], dtype=torch.float),
        ]

        self.run_class_implementation_tests(
            metric=Cat(),
            state_names={"dim", "inputs"},
            update_kwargs={"input": update_inputs},
            compute_result=torch.cat(update_inputs, dim=0),
            num_total_updates=4,
            num_processes=2,
        )

    def test_cat_class_update_input_shape_incompatible(self) -> None:
        metric = Cat(dim=1)
        metric.update(torch.tensor([[1, 2], [3, 4]], dtype=torch.float))
        with self.assertRaises(RuntimeError):
            metric.update(
                torch.tensor([[5, 6], [7, 8], [9, 10]], dtype=torch.float)
            ).compute()

    def test_cat_class_compute_without_update(self) -> None:
        metric = Cat()
        compute_result = metric.compute()
        expected_result = torch.empty(0)
        self.assertTrue(torch.equal(compute_result, expected_result))

    def test_cat_class_update_input_dtpe_int32(self) -> None:
        update_inputs = [
            torch.tensor([[1, 2], [3, 4]], dtype=torch.int32),
            torch.tensor([[5, 6], [7, 8], [9, 10]], dtype=torch.int32),
            torch.tensor([[11, 12]], dtype=torch.int32),
            torch.tensor([[13, 14], [15, 16]], dtype=torch.int32),
        ]

        self.run_class_implementation_tests(
            metric=Cat(),
            state_names={"dim", "inputs"},
            update_kwargs={"input": update_inputs},
            compute_result=torch.cat(update_inputs, dim=0),
            num_total_updates=4,
            num_processes=2,
        )
