# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import torch
from torcheval.metrics.functional import binary_normalized_entropy


class TestBinaryNormalizedEntropy(unittest.TestCase):
    def test_ne_with_valid_input(self) -> None:
        input = torch.tensor([0.2, 0.3, 0.5, 0.1, 0.3, 0.5, 0.2, 0.4])
        input_logit = torch.logit(input)
        target = torch.tensor([0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0])
        weight = torch.tensor([5.0, 1.0, 2.0, 3.0, 4.0, 7.0, 1.0, 1.0])

        # without weight and input are probability value
        torch.testing.assert_close(
            binary_normalized_entropy(input, target, weight=None, from_logits=False),
            torch.tensor(1.046165732800875, dtype=torch.float64),
        )

        # with weight and input are probability value
        torch.testing.assert_close(
            binary_normalized_entropy(input, target, weight=weight, from_logits=False),
            torch.tensor(1.0060274419349144, dtype=torch.float64),
        )

        # without weight and input are logit value
        torch.testing.assert_close(
            binary_normalized_entropy(
                input_logit, target, weight=None, from_logits=True
            ),
            torch.tensor(1.046165732800875, dtype=torch.float64),
        )

        # with weight and input are logit value
        torch.testing.assert_close(
            binary_normalized_entropy(
                input_logit, target, weight=weight, from_logits=True
            ),
            torch.tensor(1.0060274419349144, dtype=torch.float64),
        )

        # multi-task
        input_multi_task = input.view(2, -1)
        input_logit_multi_task = torch.logit(input_multi_task)
        target_multi_task = target.view(2, -1)
        weight_multi_task = weight.view(2, -1)

        # without weight and input are probability value
        torch.testing.assert_close(
            binary_normalized_entropy(
                input_multi_task,
                target_multi_task,
                weight=None,
                num_tasks=2,
                from_logits=False,
            ),
            torch.tensor([0.6127690164269908, 1.9662867428699988], dtype=torch.float64),
        )

        # with weight and input are probability value
        torch.testing.assert_close(
            binary_normalized_entropy(
                input_multi_task,
                target_multi_task,
                weight=weight_multi_task,
                num_tasks=2,
                from_logits=False,
            ),
            torch.tensor([0.6087141981256933, 1.3590155586403683], dtype=torch.float64),
        )

        # without weight and input are logit value
        torch.testing.assert_close(
            binary_normalized_entropy(
                input_logit_multi_task,
                target_multi_task,
                weight=None,
                num_tasks=2,
                from_logits=True,
            ),
            torch.tensor([0.6127690164269908, 1.9662867428699988], dtype=torch.float64),
        )

        # with weight and input are logit value
        torch.testing.assert_close(
            binary_normalized_entropy(
                input_logit_multi_task,
                target_multi_task,
                weight=weight_multi_task,
                num_tasks=2,
                from_logits=True,
            ),
            torch.tensor([0.6087141981256933, 1.3590155586403683], dtype=torch.float64),
        )

    def test_ne_with_invalid_input(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "`num_tasks = 2`, `input`'s shape is expected to be"
        ):
            binary_normalized_entropy(
                torch.rand((5,)),
                torch.randint(0, 2, (5,)),
                num_tasks=2,
            )

        with self.assertRaisesRegex(
            ValueError,
            "`num_tasks = 1`, `input` is expected to be one-dimensional tensor",
        ):
            binary_normalized_entropy(
                torch.rand((2, 10)),
                torch.randint(0, 2, (2, 10)),
            )

        with self.assertRaisesRegex(ValueError, "is different from `target` shape"):
            binary_normalized_entropy(torch.rand((5,)), torch.randint(0, 2, (3,)))

        with self.assertRaisesRegex(ValueError, "is different from `input` shape"):
            binary_normalized_entropy(
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
        #     binary_normalized_entropy(
        #         torch.rand((5,)) * 10.0,
        #         torch.randint(0, 2, (5,)),
        #         weight=None,
        #         from_logits=False,
        #     )
