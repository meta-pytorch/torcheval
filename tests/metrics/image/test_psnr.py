# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import torch
from skimage.metrics import peak_signal_noise_ratio as skimage_peak_signal_noise_ratio
from torcheval.metrics import PeakSignalNoiseRatio
from torcheval.utils.test_utils.metric_class_tester import (
    BATCH_SIZE,
    IMG_CHANNELS,
    IMG_HEIGHT,
    IMG_WIDTH,
    MetricClassTester,
    NUM_TOTAL_UPDATES,
)


class TestPeakSignalNoiseRatio(MetricClassTester):
    def _get_random_data_PeakSignalToNoiseRatio(
        self,
        num_updates: int,
        batch_size: int,
        num_channels: int,
        height: int,
        width: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = torch.rand(
            size=(num_updates, batch_size, num_channels, height, width),
        )
        targets = torch.rand(
            size=(num_updates, batch_size, num_channels, height, width),
        )
        return inputs, targets

    def _test_psnr_skimage_equivelant(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        data_range: float | None = None,
    ) -> None:
        input_np = input.numpy().ravel()
        target_np = target.numpy().ravel()

        skimage_result = torch.tensor(
            skimage_peak_signal_noise_ratio(
                image_true=target_np, image_test=input_np, data_range=data_range
            )
        )

        state_names = {
            "num_observations",
            "sum_squared_error",
            "data_range",
            "min_target",
            "max_target",
        }

        self.run_class_implementation_tests(
            metric=PeakSignalNoiseRatio(data_range=data_range),
            state_names=state_names,
            update_kwargs={"input": input, "target": target},
            compute_result=skimage_result.to(torch.float32),
        )

    def test_psnr_with_random_data(self) -> None:
        input, target = self._get_random_data_PeakSignalToNoiseRatio(
            NUM_TOTAL_UPDATES, BATCH_SIZE, IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH
        )
        self._test_psnr_skimage_equivelant(input, target)

    def test_psnr_with_random_data_and_data_range(self) -> None:
        input, target = self._get_random_data_PeakSignalToNoiseRatio(
            NUM_TOTAL_UPDATES, BATCH_SIZE, IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH
        )
        self._test_psnr_skimage_equivelant(input, target, data_range=0.5)

    def test_psnr_class_invalid_input(self) -> None:
        metric = PeakSignalNoiseRatio()
        with self.assertRaisesRegex(
            ValueError,
            "The `input` and `target` must have the same shape, "
            r"got shapes torch.Size\(\[4, 3, 4, 4\]\) and torch.Size\(\[4, 3, 4, 6\]\).",
        ):
            metric.update(torch.rand(4, 3, 4, 4), torch.rand(4, 3, 4, 6))

    def test_psnr_class_invalid_data_range(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "`data_range needs to be either `None` or `float`."
        ):
            PeakSignalNoiseRatio(data_range=5)

        with self.assertRaisesRegex(ValueError, "`data_range` needs to be positive."):
            PeakSignalNoiseRatio(data_range=-1.0)
