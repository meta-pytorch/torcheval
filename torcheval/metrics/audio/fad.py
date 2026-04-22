# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
from collections.abc import Callable, Iterable
from typing import Any

import torch
from torcheval.metrics.functional.frechet import gaussian_frechet_distance
from torcheval.metrics.metric import Metric

# pyre-ignore-all-errors[16]: Undefined attribute of metric states.


class FrechetAudioDistance(Metric[torch.Tensor]):
    """Computes the Fréchet distance between predicted and target audio waveforms.

    Original paper: https://arxiv.org/abs/1812.08466

    Args:
        preproc (Callable[[torch.Tensor], torch.Tensor]): Callable for preprocessing waveforms prior to passing to model.
        model (torch.nn.Module): Model for generating embeddings from preprocessed waveforms.
        embedding_dim (int): Size of embedding.
        device (torch.device or None, optional): Device where computations will be performed.
            If `None`, the default device will be used. (Default: `None`)
    """

    def __init__(
        self,
        preproc: Callable[[torch.Tensor], torch.Tensor],
        model: torch.nn.Module,
        embedding_dim: int,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(device=device)

        self.preproc = preproc
        # pyre-ignore
        self.model = model.to(device)
        self.model.eval()
        self.model.requires_grad_(False)

        self._add_state("pred_mean_partial", torch.zeros(1, embedding_dim))
        self._add_state("pred_cov_partial", torch.zeros(embedding_dim, embedding_dim))
        self._add_state("pred_n", 0)
        self._add_state("target_mean_partial", torch.zeros(1, embedding_dim))
        self._add_state("target_cov_partial", torch.zeros(embedding_dim, embedding_dim))
        self._add_state("target_n", 0)

    def _compute_embedding(self, waveform: torch.Tensor) -> torch.Tensor:
        model_input = self.preproc(waveform)
        model_input = model_input.to(self.device)
        return self.model(model_input)

    def _update_state(self, state_prefix: str, waveforms: torch.Tensor) -> None:
        n = getattr(self, f"{state_prefix}_n")
        mean_partial = getattr(self, f"{state_prefix}_mean_partial")
        cov_partial = getattr(self, f"{state_prefix}_cov_partial")

        for idx in range(waveforms.size(0)):
            embedding = self._compute_embedding(
                waveforms[idx]
            )  # (n_example, embedding_dim)
            n += embedding.size(0)
            mean_partial += embedding.sum(0).unsqueeze(0)
            cov_partial += embedding.T @ embedding

        setattr(self, f"{state_prefix}_n", n)
        setattr(self, f"{state_prefix}_mean_partial", mean_partial)
        setattr(self, f"{state_prefix}_cov_partial", cov_partial)

    @torch.inference_mode()
    # pyre-ignore[14]: inconsistent override on *_:Any, **__:Any
    def update(
        self, preds: torch.Tensor, targets: torch.Tensor
    ) -> "FrechetAudioDistance":
        """Update states with a batch of predicted and target waveforms.

        Args:
            preds (torch.Tensor): Predicted waveforms, with shape (B, T)
            targets (torch.Tensor): Target waveforms, with shape (C, U)

        """
        self._update_state("pred", preds)
        self._update_state("target", targets)
        return self

    @torch.inference_mode()
    def compute(self: "FrechetAudioDistance") -> torch.Tensor:
        """Computes the Fréchet distance on the current set of internal states.

        Returns:
            torch.Tensor: the Fréchet distance between the accumulated predicted and target waveforms.
        """
        # pyrefly: ignore [missing-attribute]
        target_mean = self.target_mean_partial / self.target_n
        # pyrefly: ignore [missing-attribute]
        target_cov = self.target_cov_partial / (self.target_n - 1) - target_mean.T @ (
            target_mean
            # pyrefly: ignore [missing-attribute]
        ) * self.target_n / (self.target_n - 1)
        # pyrefly: ignore [missing-attribute]
        pred_mean = self.pred_mean_partial / self.pred_n
        # pyrefly: ignore [missing-attribute]
        pred_cov = self.pred_cov_partial / (self.pred_n - 1) - pred_mean.T @ (
            pred_mean
            # pyrefly: ignore [missing-attribute]
        ) * self.pred_n / (self.pred_n - 1)
        return gaussian_frechet_distance(
            pred_mean.squeeze(0),
            # pyre-fixme[6]: For 2nd argument expected `Tensor` but got `float`.
            pred_cov,
            target_mean.squeeze(0),
            # pyre-fixme[6]: For 4th argument expected `Tensor` but got `float`.
            target_cov,
        )

    @torch.inference_mode()
    # pyre-ignore[14]: inconsistent override on *_:Any, **__:Any
    def merge_state(
        self, fads: Iterable["FrechetAudioDistance"]
    ) -> "FrechetAudioDistance":
        """Merges the states of other `FrechetAudioDistance` instances into those of the current instance.

        Args:
            fads (Iterable[FrechetAudioDistance]): The other `FrechetAudioDistance` instances to merge states from.
        """
        for fad in fads:
            # pyrefly: ignore [missing-attribute]
            self.pred_mean_partial += fad.pred_mean_partial
            # pyrefly: ignore [missing-attribute]
            self.pred_cov_partial += fad.pred_cov_partial
            # pyrefly: ignore [missing-attribute]
            self.pred_n += fad.pred_n
            # pyrefly: ignore [missing-attribute]
            self.target_mean_partial += fad.target_mean_partial
            # pyrefly: ignore [missing-attribute]
            self.target_cov_partial += fad.target_cov_partial
            # pyrefly: ignore [missing-attribute]
            self.target_n += fad.target_n
        return self

    def to(
        self,
        device: str | torch.device,
        *args: Any,
        **kwargs: Any,
    ) -> "FrechetAudioDistance":
        super().to(device=device)
        self.model.to(self.device)
        return self

    @staticmethod
    def with_vggish(device: torch.device | None = None) -> "FrechetAudioDistance":
        """Builds an instance of FrechetAudioDistance with TorchAudio's pretrained VGGish model.
        The returned instance expects batches of waveforms of shape `(B, T)` and sampled at a rate of 16KHZ.

        Args:
            device (torch.device or None, optional): Device where computations will be performed.
                If `None`, the default device will be used. (Default: `None`)

        Returns:
            FrechetAudioDistance: Instance of FrechetAudioDistance preloaded with TorchAudio's pretrained VGGish model.
        """
        try:
            from torchaudio.prototype.pipelines import VGGISH
        except ImportError:
            raise RuntimeError(
                "Using the pretrained VGGish model requires the TorchAudio nightly binary as it is a prototype feature. "
                "Please install the latest nightly version of ``torchaudio``."
            )
        model = copy.deepcopy(VGGISH.get_model())
        model.embedding_network = torch.nn.Sequential(
            *list(model.embedding_network.children())[:-1]
        )
        return FrechetAudioDistance(
            VGGISH.get_input_processor(), model, 128, device=device
        )
