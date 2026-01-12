#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import torch
from torcheval.utils import (
    get_rand_data_binary,
    get_rand_data_binned_binary,
    get_rand_data_multiclass,
)


class BinaryRandomDataTest(unittest.TestCase):
    cuda_avail: bool = torch.cuda.is_available()

    def test_get_rand_data_binary_shapes(self) -> None:
        # multi update/multi-task
        input, targets = get_rand_data_binary(num_updates=2, num_tasks=5, batch_size=10)
        self.assertEqual(input.size(), torch.Size([2, 5, 10]))
        self.assertEqual(targets.size(), torch.Size([2, 5, 10]))

        # single update/multi-task
        input, targets = get_rand_data_binary(num_updates=1, num_tasks=5, batch_size=10)
        self.assertEqual(input.size(), torch.Size([5, 10]))
        self.assertEqual(targets.size(), torch.Size([5, 10]))

        # single update/single-task
        input, targets = get_rand_data_binary(num_updates=1, num_tasks=1, batch_size=10)
        self.assertEqual(input.size(), torch.Size([10]))
        self.assertEqual(targets.size(), torch.Size([10]))

        # multi update/single-task
        input, targets = get_rand_data_binary(num_updates=3, num_tasks=1, batch_size=10)
        self.assertEqual(input.size(), torch.Size([3, 10]))
        self.assertEqual(targets.size(), torch.Size([3, 10]))

    @unittest.skipUnless(
        condition=cuda_avail, reason="This test needs a GPU host to run."
    )
    def test_get_rand_data_binary_GPU(self) -> None:
        device = torch.device("cuda")
        input, targets = get_rand_data_binary(
            num_updates=2, num_tasks=5, batch_size=10, device=device
        )
        self.assertEqual(input.size(), targets.size())
        self.assertTrue(input.is_cuda)
        self.assertTrue(targets.is_cuda)

    def test_get_rand_data_binned_binary_shapes(self) -> None:
        # multi update/multi-task
        input, targets, thresholds = get_rand_data_binned_binary(
            num_updates=2, num_tasks=5, batch_size=10, num_bins=20
        )
        self.assertEqual(input.size(), torch.Size([2, 5, 10]))
        self.assertEqual(targets.size(), torch.Size([2, 5, 10]))
        self.assertEqual(thresholds.size(), torch.Size([20]))

        # single update/multi-task
        input, targets, thresholds = get_rand_data_binned_binary(
            num_updates=1, num_tasks=5, batch_size=10, num_bins=20
        )
        self.assertEqual(input.size(), torch.Size([5, 10]))
        self.assertEqual(targets.size(), torch.Size([5, 10]))
        self.assertEqual(thresholds.size(), torch.Size([20]))

        # single update/single-task
        input, targets, thresholds = get_rand_data_binned_binary(
            num_updates=1, num_tasks=1, batch_size=10, num_bins=20
        )
        self.assertEqual(input.size(), torch.Size([10]))
        self.assertEqual(targets.size(), torch.Size([10]))
        self.assertEqual(thresholds.size(), torch.Size([20]))

        # multi update/single-task
        input, targets, thresholds = get_rand_data_binned_binary(
            num_updates=3, num_tasks=1, batch_size=10, num_bins=20
        )
        self.assertEqual(input.size(), torch.Size([3, 10]))
        self.assertEqual(targets.size(), torch.Size([3, 10]))
        self.assertEqual(thresholds.size(), torch.Size([20]))

    @unittest.skipUnless(
        condition=cuda_avail, reason="This test needs a GPU host to run."
    )
    def test_get_rand_data_binned_binary_GPU(self) -> None:
        device = torch.device("cuda")
        input, targets, thresholds = get_rand_data_binned_binary(
            num_updates=2, num_tasks=5, batch_size=10, num_bins=20, device=device
        )
        self.assertTrue(input.is_cuda)
        self.assertTrue(targets.is_cuda)
        self.assertTrue(thresholds.is_cuda)


class MulticlassRandomDataTest(unittest.TestCase):
    cuda_avail: bool = torch.cuda.is_available()

    def test_get_rand_data_multiclass_shapes(self) -> None:
        # multi update
        input, targets = get_rand_data_multiclass(
            num_updates=2, num_classes=5, batch_size=10
        )
        self.assertEqual(input.size(), torch.Size([2, 10, 5]))
        self.assertEqual(targets.size(), torch.Size([2, 10]))
        self.assertTrue(torch.all(torch.lt(targets, 5)))

        # single update
        input, targets = get_rand_data_multiclass(
            num_updates=1, num_classes=5, batch_size=10
        )
        self.assertEqual(input.size(), torch.Size([10, 5]))
        self.assertEqual(targets.size(), torch.Size([10]))
        self.assertTrue(torch.all(torch.lt(targets, 5)))

    @unittest.skipUnless(
        condition=cuda_avail, reason="This test needs a GPU host to run."
    )
    def test_get_rand_data_multiclass_GPU(self) -> None:
        device = torch.device("cuda")
        input, targets = get_rand_data_multiclass(
            num_updates=2, num_classes=5, batch_size=10, device=device
        )
        self.assertEqual(input.size(), torch.Size([2, 10, 5]))
        self.assertTrue(torch.all(torch.lt(targets, 5)))
        self.assertTrue(input.is_cuda)
        self.assertTrue(targets.is_cuda)
