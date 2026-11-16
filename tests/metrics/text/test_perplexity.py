# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch
from torcheval.metrics.text import Perplexity
from torcheval.utils.test_utils.metric_class_tester import MetricClassTester


class TestPerplexity(MetricClassTester):
    def test_perplexity(self) -> None:
        input = torch.tensor(
            [
                [[[0.3659, 0.7025, 0.3104]], [[0.5555, 0.5435, 0.7654]]],
                [[[0.0097, 0.6577, 0.1947]], [[0.5342, 0.6234, 0.8764]]],
                [[[0.4343, 0.0001, 0.9231]], [[0.6544, 0.0343, 0.0432]]],
                [[[0.2222, 0.0432, 0.3543]], [[0.9433, 0.8687, 0.5324]]],
            ]
        )
        target = torch.tensor(
            [
                [[2], [1]],
                [[1], [0]],
                [[2], [0]],
                [[1], [1]],
            ]
        )

        self.run_class_implementation_tests(
            metric=Perplexity(),
            state_names={"sum_log_probs", "num_total"},
            update_kwargs={"input": input, "target": target},
            compute_result=torch.tensor(2.784602403641, dtype=torch.float64),
            num_total_updates=4,
            num_processes=2,
        )

    def test_perplexity_with_ignore_index(self) -> None:
        input = torch.tensor(
            [
                [[[0.3659, 0.7025, 0.3104]], [[0.5555, 0.5435, 0.7654]]],
                [[[0.0097, 0.6577, 0.1947]], [[0.5342, 0.6234, 0.8764]]],
                [[[0.4343, 0.0001, 0.9231]], [[0.6544, 0.0343, 0.0432]]],
                [[[0.2222, 0.0432, 0.3543]], [[0.9433, 0.8687, 0.5324]]],
            ]
        )
        target = torch.tensor(
            [
                [[2], [1]],
                [[1], [0]],
                [[2], [0]],
                [[1], [1]],
            ]
        )

        self.run_class_implementation_tests(
            metric=Perplexity(ignore_index=2),
            state_names={"sum_log_probs", "num_total"},
            update_kwargs={"input": input, "target": target},
            compute_result=torch.tensor(2.824995994568, dtype=torch.float64),
            num_total_updates=4,
            num_processes=2,
        )

    def test_perplexity_with_invalid_input(self) -> None:
        metric = Perplexity()
        with self.assertRaisesRegex(
            ValueError, "target should be a two-dimensional tensor"
        ):
            metric.update(torch.rand(4, 2, 3), torch.randint(3, (4, 2, 2)))

        with self.assertRaisesRegex(
            ValueError, "input should be a three-dimensional tensor"
        ):
            metric.update(torch.rand(3, 2), torch.randint(3, (4, 2)))

        with self.assertRaisesRegex(
            ValueError, "The `input` and `target` should have the same second dimension"
        ):
            metric.update(torch.rand(3, 2, 2), torch.randint(2, (3, 9)))

        with self.assertRaisesRegex(
            ValueError, "The `input` and `target` should have the same first dimension"
        ):
            metric.update(torch.rand(3, 2, 3), torch.randint(3, (2, 2)))

        with self.assertRaisesRegex(
            ValueError,
            "Class labels in `target` tensor cannot be larger than vocab_size minus one",
        ):
            metric.update(
                torch.rand(3, 2, 3), target=torch.tensor([[4, 2], [1, 0], [0, 0]])
            )

        metric = Perplexity(ignore_index=1)
        with self.assertRaisesRegex(
            ValueError,
            "Class labels in `target` tensor cannot be larger than vocab_size minus one",
        ):
            metric.update(torch.rand(3, 2, 3), torch.tensor([[4, 2], [1, 0], [0, 0]]))

        metric = Perplexity(ignore_index=100)
        with self.assertRaisesRegex(
            ValueError,
            "Class labels in `target` tensor cannot be larger than vocab_size minus one",
        ):
            metric.update(torch.rand(3, 2, 3), torch.tensor([[4, 2], [1, 0], [100, 0]]))
