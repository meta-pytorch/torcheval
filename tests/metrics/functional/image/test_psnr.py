# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import torch
from skimage.metrics import peak_signal_noise_ratio as skimage_psnr
from torcheval.metrics.functional import peak_signal_noise_ratio
from torcheval.utils.test_utils.metric_class_tester import (
    BATCH_SIZE,
    IMG_CHANNELS,
    IMG_HEIGHT,
    IMG_WIDTH,
)


class TestPeakSignalNoiseRatio(unittest.TestCase):
    def test_psnr_skimage_equivelant(self) -> None:
        input, target = self._get_random_data_peak_signal_to_noise_ratio(
            BATCH_SIZE, IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH
        )

        input_np = input.numpy().ravel()
        target_np = target.numpy().ravel()
        skimage_result = torch.tensor(
            skimage_psnr(target_np, input_np), dtype=torch.float32
        )

        torch.testing.assert_close(
            peak_signal_noise_ratio(input, target),
            skimage_result,
            atol=1e-3,
            rtol=1e-3,
        )

    def test_psnr_with_invalid_input(self) -> None:
        input = torch.rand(BATCH_SIZE, IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH)
        target = torch.rand(BATCH_SIZE, IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH + 1)
        with self.assertRaisesRegex(
            ValueError,
            r"^The `input` and `target` must have the same shape, "
            + rf"got shapes torch.Size\(\[{BATCH_SIZE}, {IMG_CHANNELS}, {IMG_HEIGHT}, {IMG_WIDTH}\]\) "
            + rf"and torch.Size\(\[{BATCH_SIZE}, {IMG_CHANNELS}, {IMG_HEIGHT}, {IMG_WIDTH + 1}\]\).",
        ):
            peak_signal_noise_ratio(input, target)

    def _get_random_data_peak_signal_to_noise_ratio(
        self, batch_size: int, num_channels: int, height: int, width: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input = torch.rand(
            size=(batch_size, num_channels, height, width),
        )
        target = torch.rand(
            size=(batch_size, num_channels, height, width),
        )
        return input, target
