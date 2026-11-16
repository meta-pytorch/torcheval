# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch
from torcheval.metrics import Max
from torcheval.utils.test_utils.metric_class_tester import (
    BATCH_SIZE,
    MetricClassTester,
    NUM_TOTAL_UPDATES,
)


class TestMax(MetricClassTester):
    def _test_max_class_with_input(self, input_val_tensor: torch.Tensor) -> None:
        self.run_class_implementation_tests(
            metric=Max(),
            state_names={"max"},
            update_kwargs={"input": input_val_tensor},
            compute_result=torch.max(input_val_tensor),
        )

    def test_max_class_base(self) -> None:
        input_val_tensor = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE)
        self._test_max_class_with_input(input_val_tensor)
        input_val_tensor = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE, 4)
        self._test_max_class_with_input(input_val_tensor)
        input_val_tensor = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE, 3, 4)
        self._test_max_class_with_input(input_val_tensor)

    def test_max_class_update_input_dimension_different(self) -> None:
        self.run_class_implementation_tests(
            metric=Max(),
            state_names={"max"},
            update_kwargs={
                "input": [
                    torch.tensor(1.0),
                    torch.tensor([2.0, 3.0, 5.0]),
                    torch.tensor([-1.0, 2.0]),
                    torch.tensor([[1.0, 6.0], [2.0, -4.0]]),
                ]
            },
            compute_result=torch.tensor(6.0),
            num_total_updates=4,
            num_processes=2,
        )
