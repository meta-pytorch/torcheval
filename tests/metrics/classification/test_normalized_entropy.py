# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import torch
from torcheval.metrics.classification import BinaryNormalizedEntropy
from torcheval.utils.test_utils.metric_class_tester import MetricClassTester


class TestBinaryNormalizedEntropy(MetricClassTester):
    def test_ne_with_valid_input(self) -> None:
        input = torch.tensor([[0.2, 0.3], [0.5, 0.1], [0.3, 0.5], [0.2, 0.4]])
        input_logit = torch.logit(input)
        target = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
        weight = torch.tensor([[5.0, 1.0], [2.0, 3.0], [4.0, 7.0], [1.0, 1.0]])

        # without weight and input are probability value
        self.run_class_implementation_tests(
            metric=BinaryNormalizedEntropy(),
            state_names={"total_entropy", "num_examples", "num_positive"},
            update_kwargs={"input": input, "target": target},
            compute_result=torch.tensor([1.046165732800875], dtype=torch.float64),
            num_total_updates=4,
            num_processes=2,
        )

        # with weight and input are probability value
        self.run_class_implementation_tests(
            metric=BinaryNormalizedEntropy(),
            state_names={"total_entropy", "num_examples", "num_positive"},
            update_kwargs={"input": input, "target": target, "weight": weight},
            compute_result=torch.tensor([1.0060274419349144], dtype=torch.float64),
            num_total_updates=4,
            num_processes=2,
        )

        # without weight and input are logit value
        self.run_class_implementation_tests(
            metric=BinaryNormalizedEntropy(from_logits=True),
            state_names={"total_entropy", "num_examples", "num_positive"},
            update_kwargs={"input": input_logit, "target": target},
            compute_result=torch.tensor([1.046165732800875], dtype=torch.float64),
            num_total_updates=4,
            num_processes=2,
        )

        # with weight and input are logit value
        self.run_class_implementation_tests(
            metric=BinaryNormalizedEntropy(from_logits=True),
            state_names={"total_entropy", "num_examples", "num_positive"},
            update_kwargs={"input": input_logit, "target": target, "weight": weight},
            compute_result=torch.tensor([1.0060274419349144], dtype=torch.float64),
            num_total_updates=4,
            num_processes=2,
        )

        # multi-task
        input_multi_tasks = torch.tensor(
            [[[0.2, 0.3], [0.5, 0.1]], [[0.3, 0.5], [0.2, 0.4]]]
        )
        input_logit_multi_tasks = torch.logit(input_multi_tasks)
        target_multi_tasks = torch.tensor(
            [[[0.0, 0.0], [1.0, 0.0]], [[1.0, 0.0], [1.0, 1.0]]]
        )
        weight_multi_tasks = torch.tensor(
            [[[5.0, 1.0], [2.0, 3.0]], [[4.0, 7.0], [1.0, 1.0]]]
        )

        # without weight and input are probability value
        self.run_class_implementation_tests(
            metric=BinaryNormalizedEntropy(num_tasks=2),
            state_names={"total_entropy", "num_examples", "num_positive"},
            update_kwargs={"input": input_multi_tasks, "target": target_multi_tasks},
            compute_result=torch.tensor(
                [1.101184269229977, 1.477871490067013], dtype=torch.float64
            ),
            num_total_updates=2,
            num_processes=2,
        )

        # without weight and input are logit value
        self.run_class_implementation_tests(
            metric=BinaryNormalizedEntropy(num_tasks=2, from_logits=True),
            state_names={"total_entropy", "num_examples", "num_positive"},
            update_kwargs={
                "input": input_logit_multi_tasks,
                "target": target_multi_tasks,
            },
            compute_result=torch.tensor(
                [1.101184269229977, 1.477871490067013], dtype=torch.float64
            ),
            num_total_updates=2,
            num_processes=2,
        )

        # with weight and input are probability value
        self.run_class_implementation_tests(
            metric=BinaryNormalizedEntropy(num_tasks=2),
            state_names={"total_entropy", "num_examples", "num_positive"},
            update_kwargs={
                "input": input_multi_tasks,
                "target": target_multi_tasks,
                "weight": weight_multi_tasks,
            },
            compute_result=torch.tensor(
                [1.2010980734390073, 0.884474687519494], dtype=torch.float64
            ),
            num_total_updates=2,
            num_processes=2,
        )

        # with weight and input are logit value
        self.run_class_implementation_tests(
            metric=BinaryNormalizedEntropy(num_tasks=2, from_logits=True),
            state_names={"total_entropy", "num_examples", "num_positive"},
            update_kwargs={
                "input": input_logit_multi_tasks,
                "target": target_multi_tasks,
                "weight": weight_multi_tasks,
            },
            compute_result=torch.tensor(
                [1.2010980734390073, 0.884474687519494], dtype=torch.float64
            ),
            num_total_updates=2,
            num_processes=2,
        )

    def test_ne_with_invalid_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "`num_tasks` value should be greater"):
            metric = BinaryNormalizedEntropy(num_tasks=-1)

        metric = BinaryNormalizedEntropy()
        with self.assertRaisesRegex(ValueError, "is different from `target` shape"):
            metric.update(torch.rand((5,)), torch.randint(0, 2, (3,)))

        with self.assertRaisesRegex(ValueError, "is different from `input` shape"):
            metric.update(
                torch.rand((5,)),
                torch.randint(0, 2, (5,)),
                weight=torch.randint(0, 20, (3,)),
            )
        # Range check commented out in D98589563: input.max()/min() on GPU
        # tensors triggers cudaStreamSynchronize. Test removed since the check
        # is disabled. TODO(T262340351): restore when sync-free alternative lands.
        #
        # with self.assertRaisesRegex(
        #     ValueError,
        #     "`input` should be probability",
        # ):
        #     metric.update(
        #         torch.rand((5,)) * 10.0,
        #         torch.randint(0, 2, (5,)),
        #     )
