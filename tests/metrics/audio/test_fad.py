# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import numpy as np
import torch
from torcheval.metrics import FrechetAudioDistance
import pytest

pytest.importorskip("torchaudio.prototype.pipelines")



# pyre-fixme[24]: Generic type `np.ndarray` expects 2 type parameters.
def add_noise(x: np.ndarray, std_dev: float) -> np.ndarray:
    """Adds Gaussian noise to the samples.
    Args:
    data: 1d Numpy array containing floating point samples. Not necessarily
        normalized.
    stddev: The standard deviation of the added noise.
    Returns:
        1d Numpy array containing the provided floating point samples with added
        Gaussian noise.
    Raises:
    ValueError: When data is not a 1d numpy array.
    """
    if len(x.shape) != 1:
        raise ValueError("expected 1d numpy array.")
    max_value = np.amax(np.abs(x))
    num_samples = x.shape[0]
    gauss = np.random.normal(0, std_dev, (num_samples)) * max_value
    return x + gauss


def gen_sine_wave(
    freq: float = 600,
    length_seconds: float = 6,
    sample_rate: int = 16_000,
    std_dev: float | None = None,
) -> torch.Tensor:
    """Creates sine wave of the specified frequency, sample_rate and length."""
    t = np.linspace(0, length_seconds, int(length_seconds * sample_rate))
    samples = np.sin(2 * np.pi * t * freq)
    if std_dev:
        samples = add_noise(samples, std_dev)
    # Clipping is important since we are converting to 16-bit signed integers
    # Overflow seems to be handled differently with different build modes.
    # However, clipping should be the correct way to deal with this.
    samples = np.clip(samples, -1, 0.999999)
    return torch.from_numpy(np.asarray(2**15 * samples, dtype=np.int16)).float()


def gen_fad_test_batch(num_files: int, std_dev: float | None) -> torch.Tensor:
    """Creates a tensor representing a batch of sine waves with optional
    gaussian noise added in."""
    frequencies = np.linspace(100, 1000, num_files).tolist()
    sines_np = [gen_sine_wave(freq, std_dev=std_dev) for freq in frequencies]
    sines_torch = torch.stack(sines_np)
    assert sines_torch.shape[0] == num_files
    return sines_torch


def preprocess(batch: torch.Tensor) -> torch.Tensor:
    def normalize(waveform: torch.Tensor) -> torch.Tensor:
        min_ratio = 0.1  # = 10^(max_db/-20) with max_db = 20
        return waveform / torch.clamp(torch.max(torch.abs(waveform)), min=min_ratio)

    sines_np = [normalize(batch[idx]) for idx in range(batch.size(0))]

    sample_rate = 16_000
    step_size = int(0.5 * sample_rate)
    samples_splits = []
    for audio in sines_np:
        sample_len = audio.shape[-1]
        for i in range(0, sample_len - sample_rate + 1, step_size):
            samples_splits.append(audio[i : i + sample_rate])

    sines_torch = torch.stack(samples_splits)
    return sines_torch


class TestFAD(unittest.TestCase):
    def test_vggish_fad(self) -> None:
        """FrechetAudioDistance correctly computes distances using TorchAudio's pretrained VGGish model."""
        fad = FrechetAudioDistance.with_vggish()

        np.random.seed(23487621)
        background_audio = gen_fad_test_batch(10, None)
        test1_audio = gen_fad_test_batch(5, 0.0001)
        test2_audio = gen_fad_test_batch(5, 0.00001)

        ref_fad1 = 2.784474
        ref_fad2 = 1.324203

        fad.update(
            preds=preprocess(test1_audio),
            targets=preprocess(background_audio),
        )
        fad1 = fad.compute().cpu().numpy()
        assert np.isclose(
            fad1, ref_fad1, atol=0.05
        ), f"Calculated FAD1 is {fad1} where as expected is {ref_fad1}."

        fad.reset()
        fad.update(
            preds=preprocess(test2_audio[0:2, ...]),
            targets=preprocess(background_audio[0:2, ...]),
        )
        fad.update(
            preds=preprocess(test2_audio[2:, ...]),
            targets=preprocess(background_audio[2:, ...]),
        )
        fad2 = fad.compute().cpu().numpy()
        assert np.isclose(
            fad2, ref_fad2, atol=0.05
        ), f"Calculated FAD2 is {fad2} where as expected is {ref_fad2}."

    def test_vggish_fad_merge(self) -> None:
        """FrechetAudioDistance correctly computes distances using TorchAudio's pretrained VGGish model
        when merging states across multiple instances.
        """
        fads = [FrechetAudioDistance.with_vggish().to("cpu") for _ in range(3)]

        np.random.seed(23487621)
        background_audio = gen_fad_test_batch(10, None)
        test_audio = gen_fad_test_batch(5, 0.00001)

        ref_fad = 1.324203

        fads[0].update(
            preds=preprocess(test_audio[0:2, ...]),
            targets=preprocess(background_audio[0:2, ...]),
        )
        fads[1].update(
            preds=preprocess(test_audio[2:3, ...]),
            targets=preprocess(background_audio[2:3, ...]),
        )
        fads[2].update(
            preds=preprocess(test_audio[3:, ...]),
            targets=preprocess(background_audio[3:, ...]),
        )
        fads[0].merge_state(fads[1:])
        fad = fads[0].compute().cpu().numpy()
        assert np.isclose(
            fad, ref_fad, atol=0.05
        ), f"Calculated FAD of {fad}; expected {ref_fad}."
