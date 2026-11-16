# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest
from typing import Any

import torch
from torcheval.metrics.functional import retrieval_precision
from torcheval.utils import random_data as rd


class TestRetrievalPrecision(unittest.TestCase):
    def rp_1D(
        self, inp: torch.Tensor, target: torch.Tensor, k: int | None
    ) -> torch.Tensor:
        """
        1D version of _retrieval_precision_compute
        """
        k = inp.shape[0] if k is None else k
        topk_idx = inp.topk(min(k, inp.shape[0]))[1]
        return target[topk_idx].sum() / k

    def test_with_example_cases_one_task(self) -> None:
        input = torch.tensor([0.2, 0.3, 0.5, 0.1, 0.3, 0.5, 0.2])
        target = torch.tensor([0, 0, 1, 1, 1, 0, 1])

        torch.testing.assert_close(
            retrieval_precision(
                input,
                target,
            ),
            torch.tensor(4 / 7),
        )
        torch.testing.assert_close(
            retrieval_precision(
                input,
                target,
                k=2,
            ),
            torch.tensor(1 / 2),
        )
        torch.testing.assert_close(
            retrieval_precision(
                input,
                target,
                k=4,
            ),
            torch.tensor(1 / 2),
        )
        torch.testing.assert_close(
            retrieval_precision(
                input,
                target,
                k=10,
            ),
            torch.tensor(2 / 5),
        )
        torch.testing.assert_close(
            retrieval_precision(
                input,
                target,
                k=10,
                limit_k_to_size=True,
            ),
            torch.tensor(4 / 7),
        )

    def test_with_example_cases_multi_tasks_k_None(self) -> None:
        k = None
        input = torch.tensor([[0.1, 0.2, 0.3]]).repeat(8, 1)
        target = torch.tensor(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 1],
            ]
        )
        expected_rp = torch.tensor([0.0, 1 / 3, 1 / 3, 2 / 3, 1 / 3, 2 / 3, 2 / 3, 1.0])
        torch.testing.assert_close(
            retrieval_precision(input, target, k=k, num_tasks=8), expected_rp
        )
        torch.testing.assert_close(
            retrieval_precision(input, target, k=10, limit_k_to_size=True, num_tasks=8),
            expected_rp,
        )

    def test_with_randomized_data_getter(self) -> None:
        num_samples = 7
        num_tasks = 5
        for _ in range(100):
            input, target = rd.get_rand_data_binary(
                num_updates=num_samples, num_tasks=num_tasks, batch_size=1
            )
            target = target.reshape(num_tasks, num_samples)
            input = input.reshape(num_tasks, num_samples)

            for k in [1, num_samples // 2, num_samples, num_samples * 2, None]:
                actual_prec = retrieval_precision(
                    input,
                    target,
                    k=k,
                    num_tasks=num_tasks,
                )
                expected_prec = torch.stack(
                    [self.rp_1D(inp, tar, k=k) for inp, tar in zip(input, target)]
                )
                torch.testing.assert_close(actual_prec, expected_prec)

    def test_retrieval_precision_with_invalid_parameters(self) -> None:
        def prec(args: dict[str, Any]) -> None:
            retrieval_precision(
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

    def test_retrieval_precision_invalid_input(self) -> None:
        def prec(args: dict[str, Any]) -> None:
            retrieval_precision(
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
