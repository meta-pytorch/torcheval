# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch


@torch.inference_mode()
def wasserstein_1d(
    x: torch.Tensor,
    y: torch.Tensor,
    x_weights: torch.Tensor | None = None,
    y_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""
    The Wasserstein distance, also called the Earth Mover's Distance, is a
    measure of the similarity between two distributions.

    The Wasserstein distance between two distributions is intuitively the
    minimum weight of soil (times distance moved) that would need to be moved
    if the two distributions were represented by two piles of soil. [1]_

    Args
    ----------
    x, y (Tensor) : 1D Tensor values observed in the distribution.
    x_weights, y_weights (Tensor): Optional tensor weights for each value.
        If unspecified, each value is assigned the same value.
        `x_weights` (resp. `y_weights`) must have the same length as
        `x` (resp. `y`). If the weight sum differs from 1, it
        must still be positive and finite so that the weights can be normalized
        to sum to 1.

    Returns
    -------
    distance : Tensor value
        The computed distance between the distributions.

    Notes
    -----
    The first Wasserstein distance between the distributions :math:`x` and
    :math:`x` is:

    .. math::

        W_1 (x, y) = \inf_{\pi \in \Gamma (x, y)} \int_{\mathbb{R} \times
        \mathbb{R}} |p-q| \mathrm{d} \pi (p, q)

    where :math:`\Gamma (x, y)` is the set of (probability) distributions on
    :math:`\mathbb{R} \times \mathbb{R}` whose marginals are :math:`x` and
    :math:`y` on the first and second factors respectively.

    If :math:`X` and :math:`Y` are the respective CDFs of :math:`x` and
    :math:`y`, this distance also equals to:

    .. math::

        W_1(x, y) = \int_{-\infty}^{+\infty} |X-Y|

    See [2]_ for a proof of the equivalence of both definitions.

    The input distributions can be empirical, therefore coming from samples
    whose values are effectively inputs of the function, or they can be seen as
    generalized functions, in which case they are weighted sums of Dirac delta
    functions located at the specified values.

    References
    ----------
    .. [1] "Wasserstein metric", https://en.wikipedia.org/wiki/Wasserstein_metric
    .. [2] Ramdas, Garcia, Cuturi "On Wasserstein Two Sample Testing and Related
           Families of Nonparametric Tests" (2015), https://arxiv.org/abs/1509.02237

    Examples
    --------
    >>> from torcheval.metrics.functional import wasserstein_1d
    >>> wasserstein_1d(torch.tensor([0,1,2]), torch.tensor([0,1,1]))
    torch.tensor([0.33333333333333337])
    >>> wasserstein_1d(torch.tensor([0,1,2]), torch.tensor([0,1,1]), torch.tensor([1,2,0]), torch.tensor([1,1,1]))
    torch.tensor([0.0])
    >>> wasserstein_1d(torch.tensor([0,1,2,2]), torch.tensor([0,1]))
    torch.tensor([0.75])

    """
    _wasserstein_update_input_check(x, y, x_weights, y_weights)
    return _wasserstein_compute(x, y, x_weights, y_weights)


def _wasserstein_update_input_check(
    x: torch.Tensor,
    y: torch.Tensor,
    x_weights: torch.Tensor | None = None,
    y_weights: torch.Tensor | None = None,
) -> None:
    if x.nelement() == 0 or y.nelement() == 0:
        raise ValueError("Distribution cannot be empty.")
    if x.dim() > 1 or y.dim() > 1:
        raise ValueError("Distribution has to be one dimensional.")
    if not x.device == y.device:
        raise ValueError("Expected all the tensors to be on the same device.")
    if x_weights is not None:
        if x_weights.nelement() == 0:
            raise ValueError("Weights cannot be empty.")
        if not torch.all(x_weights > 0):
            raise ValueError("All weights must be non-negative.")
        if not (0 < torch.sum(x_weights) < torch.inf):
            raise ValueError("Weight tensor sum must be positive-finite.")
        if not x_weights.device == x.device:
            raise ValueError("Expected values and weights to be on the same device.")
        if x_weights.shape != x.shape:
            raise ValueError(
                "Distribution values and weight tensors must be of the same shape, "
                f"got shapes {x.shape} and {x_weights.shape}."
            )
    if y_weights is not None:
        if y_weights.nelement() == 0:
            raise ValueError("Weights cannot be empty.")
        if not torch.all(y_weights > 0):
            raise ValueError("All weights must be non-negative.")
        if not (0 < torch.sum(y_weights) < torch.inf):
            raise ValueError("Weight tensor sum must be positive-finite.")
        if not y_weights.device == y.device:
            raise ValueError("Expected values and weights to be on the same device.")
        if y_weights.shape != y.shape:
            raise ValueError(
                "Distribution values and weight tensors must be of the same shape, "
                f"got shapes {y.shape} and {y_weights.shape}."
            )


def _wasserstein_compute(
    x: torch.Tensor,
    y: torch.Tensor,
    x_weights: torch.Tensor | None,
    y_weights: torch.Tensor | None,
) -> torch.Tensor:
    # Assigning device per input
    device = x.device

    # Finding the sorted values
    x_sorter = torch.argsort(x)
    y_sorter = torch.argsort(y)

    # Bringing all the values on a central number line
    all_values = torch.concatenate((x, y))
    all_values, _ = torch.sort(all_values)

    # Compute the differences between successive values of x and y
    deltas = torch.diff(all_values)

    # Obtain respective positions of the x and y values among all_values
    x_cdf_indices = torch.searchsorted(x[x_sorter], all_values[:-1], right=True)
    y_cdf_indices = torch.searchsorted(y[y_sorter], all_values[:-1], right=True)

    # Calculate the CDF of x and y using their weights, if specified
    if x_weights is None:
        x_cdf = x_cdf_indices.to(device) / x.size(0)
    else:
        x_sorted_cum_weights = torch.cat(
            (torch.Tensor([0]).to(device), torch.cumsum(x_weights[x_sorter], dim=0))
        )
        x_cdf = x_sorted_cum_weights[x_cdf_indices] / x_sorted_cum_weights[-1]

    if y_weights is None:
        y_cdf = y_cdf_indices.to(device) / y.size(0)
    else:
        y_sorted_cum_weights = torch.cat(
            (torch.Tensor([0]).to(device), torch.cumsum(y_weights[y_sorter], dim=0))
        )
        y_cdf = y_sorted_cum_weights[y_cdf_indices] / y_sorted_cum_weights[-1]

    return torch.sum(
        torch.multiply(torch.abs(x_cdf - y_cdf), deltas), dim=0, keepdim=True
    ).to(device)
