# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import torch
from torch import Tensor
from torcheval.metrics.image.ssim import StructuralSimilarity
from torcheval.utils.test_utils.metric_class_tester import (
    BATCH_SIZE,
    IMG_CHANNELS,
    IMG_HEIGHT,
    IMG_WIDTH,
    MetricClassTester,
    NUM_TOTAL_UPDATES,
)

# pyre-ignore-all-errors[6]


class TestStructuralSimilarity(MetricClassTester):
    def setUp(self) -> None:
        super().setUp()
        torch.manual_seed(0)

    def _get_input_data(
        self,
        num_updates: int,
        batch_size: int,
        num_channels: int,
        height: int,
        width: int,
    ) -> dict[str, Tensor]:
        images = {
            "images_1": torch.rand(
                size=(num_updates, batch_size, num_channels, height, width)
            ),
            "images_2": torch.rand(
                size=(num_updates, batch_size, num_channels, height, width)
            ),
        }

        return images

    def test_ssim(
        self,
    ) -> None:
        images = self._get_input_data(
            NUM_TOTAL_UPDATES, BATCH_SIZE, IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH
        )

        expected_result = torch.tensor(0.022607240825891495)

        state_names = {
            "mssim_sum",
            "num_images",
        }

        self.run_class_implementation_tests(
            metric=StructuralSimilarity(),
            state_names=state_names,
            update_kwargs={
                "images_1": images["images_1"],
                "images_2": images["images_2"],
            },
            compute_result=expected_result,
            min_updates_before_compute=2,
            test_merge_with_one_update=False,
            atol=1e-4,
            rtol=1e-4,
            test_devices=["cpu"],
        )

    def test_ssim_invalid_input(self) -> None:
        metric = StructuralSimilarity()
        images_1 = torch.rand(BATCH_SIZE, IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH)
        images_2 = torch.rand(BATCH_SIZE, 4, IMG_HEIGHT, IMG_WIDTH)

        with self.assertRaisesRegex(
            RuntimeError, "The two sets of images must have the same shape."
        ):
            metric.update(images_1=images_1, images_2=images_2)
