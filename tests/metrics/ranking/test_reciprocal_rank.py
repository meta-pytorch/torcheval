# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch
from torcheval.metrics.ranking import ReciprocalRank
from torcheval.utils.test_utils.metric_class_tester import MetricClassTester


class TestReciprocalRank(MetricClassTester):
    def test_mrr_with_valid_input(self) -> None:
        input = torch.tensor(
            [
                [
                    [0.9005, 0.0998, 0.2470, 0.6188, 0.9497, 0.6083, 0.7258],
                    [0.9505, 0.3270, 0.4734, 0.5854, 0.5202, 0.6546, 0.7869],
                ],
                [
                    [0.5546, 0.6027, 0.2650, 0.6624, 0.8755, 0.7838, 0.7529],
                    [0.4121, 0.6082, 0.7813, 0.5947, 0.9582, 0.8736, 0.7389],
                ],
                [
                    [0.1306, 0.7939, 0.5192, 0.0494, 0.7987, 0.3898, 0.0108],
                    [0.2399, 0.2969, 0.6738, 0.8633, 0.7939, 0.1052, 0.7702],
                ],
                [
                    [0.9097, 0.7436, 0.0051, 0.6264, 0.6616, 0.7328, 0.7413],
                    [0.5286, 0.2956, 0.0578, 0.1913, 0.8118, 0.1047, 0.7966],
                ],
            ]
        )
        target = torch.tensor([[1, 3], [3, 0], [2, 6], [4, 5]])

        self.run_class_implementation_tests(
            metric=ReciprocalRank(),
            state_names={"scores"},
            update_kwargs={"input": input, "target": target},
            compute_result=torch.tensor(
                [1.0 / 7, 0.25, 0.25, 1.0 / 7, 1.0 / 3, 1.0 / 3, 0.20, 1.0 / 6]
            ),
            num_total_updates=4,
            num_processes=2,
        )

        self.run_class_implementation_tests(
            metric=ReciprocalRank(k=5),
            state_names={"scores"},
            update_kwargs={"input": input, "target": target},
            compute_result=torch.tensor(
                [0.0, 0.25, 0.25, 0.0, 1.0 / 3, 1.0 / 3, 0.2, 0.0]
            ),
            num_total_updates=4,
            num_processes=2,
        )

    def test_mrr_with_invalid_input(self) -> None:
        metric = ReciprocalRank()
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
