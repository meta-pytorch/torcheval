# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from collections.abc import Iterable
from typing import TypeAlias, TypeVar, Union

import torch
from torcheval.metrics.metric import Metric
from typing_extensions import Self

# TODO: use a NamedTuple?
_T = TypeVar("_T", bound=Union[torch.Tensor, int])
_Output: TypeAlias = tuple[torch.Tensor, torch.Tensor]  # mean, cov


class Covariance(Metric[_Output]):
    """Fit sample mean + covariance to empirical distribution"""

    def __init__(self, *, device: torch.device | None = None) -> None:
        super().__init__(device=device)
        self.sum: torch.Tensor = self._add_state_and_return(
            "sum", default=torch.as_tensor(0.0)
        )
        self.ss_sum: torch.Tensor = self._add_state_and_return(
            "ss_sum", default=torch.as_tensor(0.0)
        )
        self.n: int = self._add_state_and_return("n", default=0)

    def _add_state_and_return(self, name: str, default: _T) -> _T:
        # Helper function for pyre
        self._add_state(name, default)
        return getattr(self, name)

    def _update(self, sum: torch.Tensor, ss_sum: torch.Tensor, n: int) -> None:
        if n == 0:
            return
        elif self.n == 0:
            self.n = n
            self.ss_sum = ss_sum
            self.sum = sum
        else:
            # Welford's algorithm for numerical stability
            delta = (self.sum / self.n) - (sum / n)
            outer = torch.outer(delta, delta)

            scale = n * self.n / (self.n + n)
            self.ss_sum += ss_sum + outer * scale
            self.sum += sum
            self.n += n

    # pyre-fixme[14]
    def update(self, obs: torch.Tensor) -> Self:
        assert obs.ndim == 2
        with torch.inference_mode():
            demeaned = obs - obs.mean(dim=0, keepdim=True)
            ss_sum = torch.einsum("ni,nj->ij", demeaned, demeaned)
            self._update(obs.sum(dim=0), ss_sum, len(obs))
        return self

    # pyre-fixme[14]
    def merge_state(self, metrics: Iterable[Self]) -> Self:
        with torch.inference_mode():
            for other in metrics:
                if self.n == 0 and other.n > 0:
                    # Adopt independent storage before subsequent in-place updates.
                    self._update(other.sum.clone(), other.ss_sum.clone(), other.n)
                else:
                    self._update(other.sum, other.ss_sum, other.n)
        return self

    def compute(self) -> _Output:
        if self.n < 2:
            msg = f"Not enough samples to estimate covariance (found {self.n})"
            raise ValueError(msg)
        with torch.inference_mode():
            mean = self.sum / self.n
            # TODO: make degress of freedom configurable?
            cov = self.ss_sum / (self.n - 1)
            return mean, cov
