# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# pyre-ignore-all-errors[56]: Pyre was not able to infer the type of argument

import unittest

import torch
from sklearn.metrics import auc as sklearn_auc
from torcheval.metrics.functional.aggregation import auc


class TestAUC(unittest.TestCase):
    def test_auc_base(self) -> None:
        x = torch.tensor(
            [
                [0.04, 0.06, 0.8],
                [0.1, 0.2, 0.7],
                [0.04, 0.06, 0.08],
                [0.10, 0.14, 0.18],
            ]
        )
        y = torch.tensor([[2, 0, 1], [0, 1, 0], [1, 1, 1], [2, 0, 1]])
        sklearn_result = torch.tensor(
            [sklearn_auc(a, b) for b, a in zip(y, x)], dtype=torch.float32
        )
        torch_auc_result = auc(x, y)
        torch.testing.assert_close(
            torch_auc_result,
            sklearn_result,
            atol=1e-8,
            rtol=1e-5,
        )

    def test_auc_reordered_input(self) -> None:
        x = torch.tensor(
            [
                [0.05, 0.02, 1.8],
                [0.6, 0.3, 0.7],
                [0.02, 0.01, 0.08],
                [0.15, 0.11, 0.18],
            ]
        )
        y = torch.tensor([[2, 0, 1], [0, 1, 0], [1, 1, 1], [2, 0, 1]])

        sklearn_x, sklearn_x_idx = x.sort(dim=1)
        sklearn_y = y.gather(1, sklearn_x_idx)
        sklearn_auc_result = torch.tensor(
            [sklearn_auc(m, n) for n, m in zip(sklearn_y, sklearn_x)],
            dtype=torch.float32,
        )
        torcheval_auc_result = auc(x, y, reorder=True)
        torch.testing.assert_close(
            torcheval_auc_result,
            sklearn_auc_result,
            atol=1e-8,
            rtol=1e-5,
        )

    def test_auc_input_dim_1(self) -> None:
        x = torch.rand(3)
        y = torch.rand(1, 3)

        sklearn_x, sklearn_x_idx = x.unsqueeze(0).sort(dim=1)
        sklearn_y = y.gather(1, sklearn_x_idx)
        sklearn_auc_result = torch.tensor(
            [sklearn_auc(m, n) for n, m in zip(sklearn_y, sklearn_x)],
            dtype=torch.float32,
        )
        torcheval_auc_result = auc(x, y, reorder=True)
        torch.testing.assert_close(
            torcheval_auc_result,
            sklearn_auc_result,
            atol=1e-8,
            rtol=1e-5,
        )

    def test_auc_no_reorder(self) -> None:
        x = torch.tensor(
            [
                [0.02, 0.05, 1.8],
                [0.3, 0.4, 0.7],
                [0.03, 0.05, 0.08],
                [0.10, 0.11, 0.18],
            ]
        )
        y = torch.tensor([[2, 0, 1], [0, 1, 0], [1, 1, 1], [2, 0, 1]])

        sklearn_auc_result = torch.tensor(
            [sklearn_auc(m, n) for n, m in zip(y, x)],
            dtype=torch.float32,
        )
        torcheval_auc_result = auc(x, y)
        torch.testing.assert_close(
            torcheval_auc_result,
            sklearn_auc_result,
            atol=1e-8,
            rtol=1e-5,
        )

    def test_auc_1d_input_no_reorder(self) -> None:
        x = torch.ones(3)
        y = torch.ones(1, 3)

        sklearn_x, sklearn_x_idx = x.unsqueeze(0).sort(dim=1)
        sklearn_y = y.gather(1, sklearn_x_idx)
        sklearn_auc_result = torch.tensor(
            [sklearn_auc(m, n) for n, m in zip(sklearn_y, sklearn_x)],
            dtype=torch.float32,
        )
        torcheval_auc_result = auc(x, y)
        torch.testing.assert_close(
            torcheval_auc_result,
            sklearn_auc_result,
            atol=1e-8,
            rtol=1e-5,
        )

    def test_auc_invalid_input(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            r"Expected the same shape in `x` and `y` tensor but got shapes torch.Size\(\[4, 2\]\) and torch.Size\(\[3\]\).",
        ):
            auc(torch.rand(4, 2), torch.rand(3))

        with self.assertRaisesRegex(
            ValueError,
            r"Expected the same shape in `x` and `y` tensor but got shapes torch.Size\(\[3, 2\]\) and torch.Size\(\[3, 3\]\).",
        ):
            auc(torch.rand(3, 2), torch.rand(3, 3))

        with self.assertRaisesRegex(
            ValueError,
            "The `x` and `y` should have atleast 1 element, "
            r"got shapes torch.Size\(\[0\]\) and torch.Size\(\[0\]\).",
        ):
            auc(torch.tensor([]), torch.tensor([]))
