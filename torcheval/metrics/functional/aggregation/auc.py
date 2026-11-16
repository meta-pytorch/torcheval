# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch


def _auc_compute(
    x: torch.Tensor, y: torch.Tensor, reorder: bool = False
) -> torch.Tensor:
    """Computes area under the curve using the trapezoidal rule.
    Args:
        x: x-coordinates,
        y: y-coordinates
        reorder: sorts the x input tensor in order, default value is False
    Return:
        Tensor containing AUC score (float)
    """
    if x.numel() == 0 or y.numel() == 0:
        return torch.tensor([])

    if x.ndim == 1:
        x = x.unsqueeze(0)
    if y.ndim == 1:
        y = y.unsqueeze(0)

    if reorder:
        x, x_idx = torch.sort(x, dim=1, stable=True)
        y = y.gather(1, x_idx)

    return torch.trapz(y, x)


def _auc_update_input_check(x: torch.Tensor, y: torch.Tensor, n_tasks: int = 1) -> None:
    """
    Checks if the 2 input tensors have the same shape
    Checks if the 2 input tensors have atleast 1 elements.
    Args:
        x: x-coordinates
        y: y-coordinates
        n_tasks: Number of tasks that need AUC calculation. Default value is 1.
    """

    size_x = x.size()
    size_y = y.size()

    if x.ndim == 1:
        x = x.unsqueeze(0)
    if y.ndim == 1:
        y = y.unsqueeze(0)

    if x.numel() == 0 or y.numel() == 0:
        raise ValueError(
            f"The `x` and `y` should have atleast 1 element, got shapes {size_x} and {size_y}."
        )
    if x.size() != y.size():
        raise ValueError(
            f"Expected the same shape in `x` and `y` tensor but got shapes {size_x} and {size_y}."
        )

    if x.size(0) != n_tasks or y.size(0) != n_tasks:
        raise ValueError(
            f"Expected `x` dim_1={x.size(0)} and `y` dim_1={y.size(0)} have first dimension equals to n_tasks={n_tasks}."
        )


def auc(x: torch.Tensor, y: torch.Tensor, reorder: bool = False) -> torch.Tensor:
    """Computes Area Under the Curve (AUC) using the trapezoidal rule.
    Args:
        x: x-coordinates
        y: y-coordinates
        reorder: sorts the x input tensor in order, default value is False
    Return:
        Tensor containing AUC score (float)
    Raises:
        ValueError:
            If both ``x`` and ``y`` don't have the same shape.
            If both ``x`` and ``y`` have atleast 1 element.
    Example:
        >>> from torcheval.metrics.functional.aggregation.auc import auc
        >>> x = torch.tensor([0,.1,.2,.3])
        >>> y = torch.tensor([1,1,1,1])
        >>> auc(x, y)
        tensor([0.3000])
        >>> y = torch.tensor([[0, 4, 0, 4, 3],
                        [1, 1, 2, 1, 1],
                        [4, 3, 1, 4, 4],
                        [1, 0, 0, 3, 0]])
        >>> x = torch.tensor([[0.2535, 0.1138, 0.1324, 0.1887, 0.3117],
                        [0.1434, 0.4404, 0.1100, 0.1178, 0.1883],
                        [0.2344, 0.1743, 0.3110, 0.0393, 0.2410],
                        [0.1381, 0.1564, 0.0320, 0.2220, 0.4515]])
        >>> auc(x, y, reorder=True) # Reorders X and calculates AUC.
        tensor([0.3667, 0.3343, 0.8843, 0.5048])
    """
    n_tasks = 1
    if x.ndim > 1:
        n_tasks = x.size(0)
    _auc_update_input_check(x, y, n_tasks)
    return _auc_compute(x, y, reorder)
