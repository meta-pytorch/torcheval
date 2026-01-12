# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# pyre-ignore-all-errors[16]: Undefined attribute of metric states.


from collections.abc import Iterable
from typing import TypeVar

import torch
from torcheval.metrics.metric import Metric

TCat = TypeVar("TCat")


class Cat(Metric[torch.Tensor]):
    """
    Concatenate all input tensors along dimension dim. Its functional
    version is ``torch.cat(input)``.

    All input tensors to ``Cat.update()`` must either have the same shape
    (except in the concatenating dimension) or be empty.

    Zero-dimensional tensor is not a valid input of ``Cat.update()``.
    ``torch.flatten()`` can be used to flatten zero-dimensional into
    an one-dimensional tensor before passing in ``Cat.update()``.

    Examples::

        >>> import torch
        >>> from torcheval.metrics import Cat
        >>> metric = Cat(dim=1)
        >>> metric.update(torch.tensor([[1, 2], [3, 4]]))
        >>> metric.compute()
        tensor([[1, 2],
                [3, 4]]))

        >>> metric.update(torch.tensor([[5, 6], [7, 8]]))).compute()
        tensor([[1, 2, 5, 6],
                [3, 4, 7, 8]]))

        >>> metric.reset()
        >>> metric.update(torch.tensor([0])).compute()
        tensor([0])
    """

    def __init__(
        self: "Cat",
        *,
        dim: int = 0,
        device: torch.device | None = None,
    ) -> None:
        """
        Initialize a Cat metric object.

        Args:
            dim: The dimension along which to concatenate, as in ``torch.cat()``.
        """
        super().__init__(device=device)
        self._add_state("dim", dim)
        self._add_state("inputs", [])

    @torch.inference_mode()
    # pyre-ignore[14]: inconsistent override on *_:Any, **__:Any
    def update(self: TCat, input: torch.Tensor) -> TCat:
        self.inputs.append(input)
        return self

    @torch.inference_mode()
    def compute(self: TCat) -> torch.Tensor:
        """
        Return the concatenated inputs.

        If no calls to ``update()`` are made before ``compute()`` is called,
        the function returns ``torch.empty(0)``.
        """
        if not self.inputs:
            return torch.empty(0)
        return torch.cat(self.inputs, dim=self.dim)

    @torch.inference_mode()
    def merge_state(self: TCat, metrics: Iterable[TCat]) -> TCat:
        for metric in metrics:
            if metric.inputs:
                self.inputs.append(
                    torch.cat(metric.inputs, dim=metric.dim).to(self.device)
                )
        return self

    @torch.inference_mode()
    def _prepare_for_merge_state(self: TCat) -> None:
        if self.inputs:
            self.inputs = [torch.cat(self.inputs, dim=self.dim)]
