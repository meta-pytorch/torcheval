# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import torch
from torcheval.metrics.functional import binary_normalized_entropy
from torcheval.metrics.window import WindowedBinaryNormalizedEntropy
from torcheval.utils.test_utils.metric_class_tester import MetricClassTester


class TestWindowedBinaryNormalizedEntropy(MetricClassTester):
    def test_ne_with_valid_input(self) -> None:
        input = torch.rand(8, 16).to(torch.float64)
        input_logit = torch.logit(input)
        target = torch.randint(high=2, size=(8, 16)).to(torch.float64)
        weight = torch.rand(8, 16).to(torch.float64)

        # compute for lifetime
        compute_result = binary_normalized_entropy(
            input.reshape(-1), target.reshape(-1)
        ).reshape(-1)
        weighted_compute_result = binary_normalized_entropy(
            input.reshape(-1), target.reshape(-1), weight=weight.reshape(-1)
        ).reshape(-1)

        # compute for windowed
        input_tensor, target_tensor, weight_tensor = (
            input[-2:].reshape(-1),
            target[-2:].reshape(-1),
            weight[-2:].reshape(-1),
        )
        windowed_compute_result = binary_normalized_entropy(
            input_tensor, target_tensor
        ).reshape(-1)
        windowed_weighted_compute_result = binary_normalized_entropy(
            input_tensor, target_tensor, weight=weight_tensor
        ).reshape(-1)

        # compute for windowed merging
        input_tensor_merge, target_tensor_merge, weight_tensor_merge = (
            torch.cat([input[2:4], input[6:]], dim=1).reshape(-1),
            torch.cat([target[2:4], target[6:]], dim=1).reshape(-1),
            torch.cat([weight[2:4], weight[6:]], dim=1).reshape(-1),
        )
        windowed_merge_compute_result = binary_normalized_entropy(
            input_tensor_merge, target_tensor_merge
        ).reshape(-1)
        windowed_weighted_merge_compute_result = binary_normalized_entropy(
            input_tensor_merge,
            target_tensor_merge,
            weight=weight_tensor_merge,
        ).reshape(-1)

        # disable lifetime
        self.run_class_implementation_tests(
            metric=WindowedBinaryNormalizedEntropy(
                max_num_updates=2, enable_lifetime=False
            ),
            state_names={
                "max_num_updates",
                "total_updates",
                "max_num_updates",
                "total_updates",
                "windowed_total_entropy",
                "windowed_num_examples",
                "windowed_num_positive",
            },
            update_kwargs={"input": input, "target": target},
            compute_result=windowed_compute_result,
            merge_and_compute_result=windowed_merge_compute_result,
            num_processes=2,
        )

        # without weight and input are probability value
        self.run_class_implementation_tests(
            metric=WindowedBinaryNormalizedEntropy(max_num_updates=2),
            state_names={
                "max_num_updates",
                "total_updates",
                "total_entropy",
                "num_examples",
                "num_positive",
                "windowed_total_entropy",
                "windowed_num_examples",
                "windowed_num_positive",
            },
            update_kwargs={"input": input, "target": target},
            compute_result=(
                compute_result,
                windowed_compute_result,
            ),
            merge_and_compute_result=(
                compute_result,
                windowed_merge_compute_result,
            ),
            num_processes=2,
        )

        # with weight and input are probability value
        self.run_class_implementation_tests(
            metric=WindowedBinaryNormalizedEntropy(max_num_updates=2),
            state_names={
                "max_num_updates",
                "total_updates",
                "total_entropy",
                "num_examples",
                "num_positive",
                "windowed_total_entropy",
                "windowed_num_examples",
                "windowed_num_positive",
            },
            update_kwargs={"input": input, "target": target, "weight": weight},
            compute_result=(
                weighted_compute_result,
                windowed_weighted_compute_result,
            ),
            merge_and_compute_result=(
                weighted_compute_result,
                windowed_weighted_merge_compute_result,
            ),
            num_processes=2,
        )

        # compute for lifetime
        compute_result = binary_normalized_entropy(
            input_logit.reshape(-1),
            target.reshape(-1),
            from_logits=True,
        ).reshape(-1)
        weighted_compute_result = binary_normalized_entropy(
            input_logit.reshape(-1),
            target.reshape(-1),
            weight=weight.reshape(-1),
            from_logits=True,
        ).reshape(-1)

        # compute for windowed
        input_tensor = input_logit[-2:].reshape(-1)
        windowed_compute_result = binary_normalized_entropy(
            input_tensor,
            target_tensor,
            from_logits=True,
        ).reshape(-1)
        windowed_weighted_compute_result = binary_normalized_entropy(
            input_tensor,
            target_tensor,
            weight=weight_tensor,
            from_logits=True,
        ).reshape(-1)

        # compute for windowed merging
        input_tensor_merge = torch.cat(
            [input_logit[2:4], input_logit[6:]], dim=1
        ).reshape(-1)
        windowed_merge_compute_result = binary_normalized_entropy(
            input_tensor_merge,
            target_tensor_merge,
            from_logits=True,
        ).reshape(-1)
        windowed_weighted_merge_compute_result = binary_normalized_entropy(
            input_tensor_merge,
            target_tensor_merge,
            weight=weight_tensor_merge,
            from_logits=True,
        ).reshape(-1)

        # without weight and input are logit value
        self.run_class_implementation_tests(
            metric=WindowedBinaryNormalizedEntropy(max_num_updates=2, from_logits=True),
            state_names={
                "max_num_updates",
                "total_updates",
                "total_entropy",
                "num_examples",
                "num_positive",
                "windowed_total_entropy",
                "windowed_num_examples",
                "windowed_num_positive",
            },
            update_kwargs={"input": input_logit, "target": target},
            compute_result=(
                compute_result,
                windowed_compute_result,
            ),
            merge_and_compute_result=(
                compute_result,
                windowed_merge_compute_result,
            ),
            num_processes=2,
        )

        # with weight and input are logit value
        self.run_class_implementation_tests(
            metric=WindowedBinaryNormalizedEntropy(max_num_updates=2, from_logits=True),
            state_names={
                "max_num_updates",
                "total_updates",
                "total_entropy",
                "num_examples",
                "num_positive",
                "windowed_total_entropy",
                "windowed_num_examples",
                "windowed_num_positive",
            },
            update_kwargs={"input": input_logit, "target": target, "weight": weight},
            compute_result=(
                weighted_compute_result,
                windowed_weighted_compute_result,
            ),
            merge_and_compute_result=(
                weighted_compute_result,
                windowed_weighted_merge_compute_result,
            ),
            num_processes=2,
        )

        # multi-task
        input_multi_tasks = torch.rand(8, 2, 16).to(torch.float64)
        input_logit_multi_tasks = torch.logit(input_multi_tasks)
        target_multi_tasks = torch.randint(high=2, size=(8, 2, 16)).to(torch.float64)
        weight_multi_tasks = torch.rand(8, 2, 16).to(torch.float64)

        # compute for lifetime
        compute_result = binary_normalized_entropy(
            input_multi_tasks.permute(1, 0, 2).reshape(2, -1),
            target_multi_tasks.permute(1, 0, 2).reshape(2, -1),
            num_tasks=2,
        )
        weighted_compute_result = binary_normalized_entropy(
            input_multi_tasks.permute(1, 0, 2).reshape(2, -1),
            target_multi_tasks.permute(1, 0, 2).reshape(2, -1),
            weight=weight_multi_tasks.permute(1, 0, 2).reshape(2, -1),
            num_tasks=2,
        )

        # compute for windowed
        input_tensor, target_tensor, weight_tensor = (
            input_multi_tasks[-2:].permute(1, 0, 2).reshape(2, -1),
            target_multi_tasks[-2:].permute(1, 0, 2).reshape(2, -1),
            weight_multi_tasks[-2:].permute(1, 0, 2).reshape(2, -1),
        )
        windowed_compute_result = binary_normalized_entropy(
            input_tensor, target_tensor, num_tasks=2
        )
        windowed_weighted_compute_result = binary_normalized_entropy(
            input_tensor, target_tensor, weight=weight_tensor, num_tasks=2
        )

        # compute for windowed merging
        input_tensor_merge, target_tensor_merge, weight_tensor_merge = (
            torch.cat(
                [
                    input_multi_tasks[2],
                    input_multi_tasks[3],
                    input_multi_tasks[6],
                    input_multi_tasks[7],
                ],
                dim=1,
            ),
            torch.cat(
                [
                    target_multi_tasks[2],
                    target_multi_tasks[3],
                    target_multi_tasks[6],
                    target_multi_tasks[7],
                ],
                dim=1,
            ),
            torch.cat(
                [
                    weight_multi_tasks[2],
                    weight_multi_tasks[3],
                    weight_multi_tasks[6],
                    weight_multi_tasks[7],
                ],
                dim=1,
            ),
        )
        windowed_merge_compute_result = binary_normalized_entropy(
            input_tensor_merge, target_tensor_merge, num_tasks=2
        )
        windowed_weighted_merge_compute_result = binary_normalized_entropy(
            input_tensor_merge,
            target_tensor_merge,
            weight=weight_tensor_merge,
            num_tasks=2,
        )

        # without weight and input are probability value
        self.run_class_implementation_tests(
            metric=WindowedBinaryNormalizedEntropy(max_num_updates=2, num_tasks=2),
            state_names={
                "max_num_updates",
                "total_updates",
                "total_entropy",
                "num_examples",
                "num_positive",
                "windowed_total_entropy",
                "windowed_num_examples",
                "windowed_num_positive",
            },
            update_kwargs={"input": input_multi_tasks, "target": target_multi_tasks},
            compute_result=(
                compute_result,
                windowed_compute_result,
            ),
            merge_and_compute_result=(
                compute_result,
                windowed_merge_compute_result,
            ),
            num_processes=2,
        )

        # with weight and input are probability value
        self.run_class_implementation_tests(
            metric=WindowedBinaryNormalizedEntropy(max_num_updates=2, num_tasks=2),
            state_names={
                "max_num_updates",
                "total_updates",
                "total_entropy",
                "num_examples",
                "num_positive",
                "windowed_total_entropy",
                "windowed_num_examples",
                "windowed_num_positive",
            },
            update_kwargs={
                "input": input_multi_tasks,
                "target": target_multi_tasks,
                "weight": weight_multi_tasks,
            },
            compute_result=(
                weighted_compute_result,
                windowed_weighted_compute_result,
            ),
            merge_and_compute_result=(
                weighted_compute_result,
                windowed_weighted_merge_compute_result,
            ),
            num_processes=2,
        )

        # compute for lifetime
        compute_result = binary_normalized_entropy(
            input_logit_multi_tasks.permute(1, 0, 2).reshape(2, -1),
            target_multi_tasks.permute(1, 0, 2).reshape(2, -1),
            num_tasks=2,
            from_logits=True,
        )
        weighted_compute_result = binary_normalized_entropy(
            input_logit_multi_tasks.permute(1, 0, 2).reshape(2, -1),
            target_multi_tasks.permute(1, 0, 2).reshape(2, -1),
            weight=weight_multi_tasks.permute(1, 0, 2).reshape(2, -1),
            num_tasks=2,
            from_logits=True,
        )

        # compute for windowed
        input_tensor = input_logit_multi_tasks[-2:].permute(1, 0, 2).reshape(2, -1)
        windowed_compute_result = binary_normalized_entropy(
            input_tensor,
            target_tensor,
            num_tasks=2,
            from_logits=True,
        )
        windowed_weighted_compute_result = binary_normalized_entropy(
            input_tensor,
            target_tensor,
            weight=weight_tensor,
            num_tasks=2,
            from_logits=True,
        )

        # compute for windowed merging
        input_tensor_merge = torch.cat(
            [
                input_logit_multi_tasks[2],
                input_logit_multi_tasks[3],
                input_logit_multi_tasks[6],
                input_logit_multi_tasks[7],
            ],
            dim=1,
        )
        windowed_merge_compute_result = binary_normalized_entropy(
            input_tensor_merge,
            target_tensor_merge,
            num_tasks=2,
            from_logits=True,
        )
        windowed_weighted_merge_compute_result = binary_normalized_entropy(
            input_tensor_merge,
            target_tensor_merge,
            weight=weight_tensor_merge,
            num_tasks=2,
            from_logits=True,
        )

        # without weight and input are logit value
        self.run_class_implementation_tests(
            metric=WindowedBinaryNormalizedEntropy(
                max_num_updates=2, num_tasks=2, from_logits=True
            ),
            state_names={
                "max_num_updates",
                "total_updates",
                "total_entropy",
                "num_examples",
                "num_positive",
                "windowed_total_entropy",
                "windowed_num_examples",
                "windowed_num_positive",
            },
            update_kwargs={
                "input": input_logit_multi_tasks,
                "target": target_multi_tasks,
            },
            compute_result=(
                compute_result,
                windowed_compute_result,
            ),
            merge_and_compute_result=(
                compute_result,
                windowed_merge_compute_result,
            ),
            num_processes=2,
        )

        # with weight and input are logit value
        self.run_class_implementation_tests(
            metric=WindowedBinaryNormalizedEntropy(
                max_num_updates=2, num_tasks=2, from_logits=True
            ),
            state_names={
                "max_num_updates",
                "total_updates",
                "total_entropy",
                "num_examples",
                "num_positive",
                "windowed_total_entropy",
                "windowed_num_examples",
                "windowed_num_positive",
            },
            update_kwargs={
                "input": input_logit_multi_tasks,
                "target": target_multi_tasks,
                "weight": weight_multi_tasks,
            },
            compute_result=(
                weighted_compute_result,
                windowed_weighted_compute_result,
            ),
            merge_and_compute_result=(
                weighted_compute_result,
                windowed_weighted_merge_compute_result,
            ),
            num_processes=2,
        )

    def test_ne_with_invalid_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "`num_tasks` value should be greater"):
            metric = WindowedBinaryNormalizedEntropy(num_tasks=-1)

        with self.assertRaisesRegex(
            ValueError, "`max_num_updates` value should be greater"
        ):
            metric = WindowedBinaryNormalizedEntropy(max_num_updates=-1)

        metric = WindowedBinaryNormalizedEntropy()
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
