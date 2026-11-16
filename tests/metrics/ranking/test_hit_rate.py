# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch
from torcheval.metrics.ranking import HitRate
from torcheval.utils.test_utils.metric_class_tester import MetricClassTester


class TestHitRate(MetricClassTester):
    def test_hitrate_with_valid_input(self) -> None:
        input = torch.tensor(
            [
                [
                    [0.4826, 0.9517, 0.8967, 0.8995, 0.1584, 0.9445, 0.9700],
                ],
                [
                    [0.4938, 0.7517, 0.8039, 0.7167, 0.9488, 0.9607, 0.7091],
                ],
                [
                    [0.5127, 0.4732, 0.5461, 0.5617, 0.9198, 0.0847, 0.2337],
                ],
                [
                    [0.4175, 0.9452, 0.9852, 0.2131, 0.5016, 0.7305, 0.0516],
                ],
            ]
        )
        target = torch.tensor([[3], [5], [2], [1]])

        self.run_class_implementation_tests(
            metric=HitRate(),
            state_names={"scores"},
            update_kwargs={"input": input, "target": target},
            compute_result=torch.tensor([1.0000, 1.0000, 1.0000, 1.0000]),
            num_total_updates=4,
            num_processes=2,
        )

        self.run_class_implementation_tests(
            metric=HitRate(k=3),
            state_names={"scores"},
            update_kwargs={"input": input, "target": target},
            compute_result=torch.tensor([0.0000, 1.0000, 1.0000, 1.0000]),
            num_total_updates=4,
            num_processes=2,
        )

    def test_hitrate_with_invalid_input(self) -> None:
        metric = HitRate()
        with self.assertRaisesRegex(
            ValueError, "target should be a one-dimensional tensor"
        ):
            metric.update(torch.rand(3, 2), torch.rand(3, 2))

        with self.assertRaisesRegex(
            ValueError, "input should be a two-dimensional tensor"
        ):
            metric.update(torch.rand(3, 2, 2), torch.rand(3))
        with self.assertRaisesRegex(
            ValueError, "`input` and `target` should have the same minibatch dimension"
        ):
            metric.update(torch.rand(4, 2), torch.rand(3))
