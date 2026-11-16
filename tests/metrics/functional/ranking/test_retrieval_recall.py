# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest
from typing import Any

import torch
from torcheval.metrics.functional import retrieval_recall


class TestRetrievalRecall(unittest.TestCase):
    def test_with_example_cases_one_task(self) -> None:
        input = torch.tensor([0.1, 0.4, 0.6, 0.2, 0.5, 0.7, 0.3])
        target = torch.tensor([0, 0, 1, 1, 1, 0, 1])

        torch.testing.assert_close(
            retrieval_recall(
                input,
                target,
            ),
            torch.tensor(4 / 4),
        )
        torch.testing.assert_close(
            retrieval_recall(
                input,
                target,
                k=2,
            ),
            torch.tensor(1 / 4),
        )
        torch.testing.assert_close(
            retrieval_recall(
                input,
                target,
                k=3,
            ),
            torch.tensor(2 / 4),
        )
        torch.testing.assert_close(
            retrieval_recall(
                input,
                target,
                k=4,
            ),
            torch.tensor(2 / 4),
        )
        torch.testing.assert_close(
            retrieval_recall(
                input,
                target,
                k=5,
            ),
            torch.tensor(3 / 4),
        )

    def test_with_example_cases_multi_tasks_k_None(self) -> None:
        input = torch.tensor([[0.1, 0.5, 0.9]]).repeat(7, 1)
        target = torch.tensor(
            [
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 1],
            ]
        )
        expected_rp = torch.tensor([1 / 1, 1 / 1, 2 / 2, 0 / 2, 1 / 2, 1 / 2, 2 / 3])
        torch.testing.assert_close(
            retrieval_recall(input, target, k=2, limit_k_to_size=False, num_tasks=7),
            expected_rp,
        )

    def test_retrieval_recall_with_invalid_parameters(self) -> None:
        def prec(args: dict[str, Any]) -> None:
            retrieval_recall(
                input=torch.tensor([1]),
                target=torch.tensor([1]),
                **args,
            )

        # check validations on parameter k
        for k in [-1, 0]:
            with self.assertRaisesRegex(
                ValueError, rf"k must be a positive integer, got k={k}\."
            ):
                prec({"k": k})

        # check parameters coupling between k and limit_k_to_size
        with self.assertRaisesRegex(
            ValueError,
            r"when limit_k_to_size is True, k must be a positive \(>0\) integer\.",
        ):
            prec({"k": None, "limit_k_to_size": True})

    def test_retrieval_recall_invalid_input(self) -> None:
        def prec(args: dict[str, Any]) -> None:
            retrieval_recall(
                k=1,
                limit_k_to_size=False,
                **args,
            )

        with self.assertRaisesRegex(
            ValueError,
            r"input and target should be one dimensional tensors, got input and target dimensions=3\.",
        ):
            prec(
                {
                    "input": torch.tensor([[[1], [1], [1]]]),
                    "target": torch.tensor([[[1], [1], [1]]]),
                }
            )
        with self.assertRaisesRegex(
            ValueError,
            r"input and target should be two dimensional tensors with 3 rows, got input and target shape=torch\.Size\(\[1, 3, 1\]\)\.",
        ):
            prec(
                {
                    "input": torch.tensor([[[1], [1], [1]]]),
                    "target": torch.tensor([[[1], [1], [1]]]),
                    "num_tasks": 3,
                }
            )

        with self.assertRaisesRegex(
            ValueError,
            r"input and target should be one dimensional tensors, got input and target dimensions=0\.",
        ):
            prec({"input": torch.tensor(1), "target": torch.tensor(1), "num_tasks": 1})

        with self.assertRaisesRegex(
            ValueError,
            r"input and target must be of the same shape, got input\.shape=torch\.Size\(\[2, 2\]\) and target\.shape=torch\.Size\(\[2, 3\]\)\.",
        ):
            prec(
                {
                    "target": torch.tensor([[0, 0, 0], [0, 0, 0]]),
                    "input": torch.tensor([[1, 2], [1, 2]]),
                }
            )
