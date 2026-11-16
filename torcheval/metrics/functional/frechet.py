# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch


def gaussian_frechet_distance(
    mu_x: torch.Tensor, cov_x: torch.Tensor, mu_y: torch.Tensor, cov_y: torch.Tensor
) -> torch.Tensor:
    r"""Computes the Fréchet distance between two multivariate normal distributions :cite:`dowson1982frechet`.

    The Fréchet distance is also known as the Wasserstein-2 distance.

    Concretely, for multivariate Gaussians :math:`X(\mu_X, \cov_X)`
    and :math:`Y(\mu_Y, \cov_Y)`, the function computes and returns :math:`F` as

    .. math::
        F(X, Y) = || \mu_X - \mu_Y ||_2^2
        + \text{Tr}\left( \cov_X + \cov_Y - 2 \sqrt{\cov_X \cov_Y} \right)

    Args:
        mu_x (torch.Tensor): mean :math:`\mu_X` of multivariate Gaussian :math:`X`, with shape `(N,)`.
        cov_x (torch.Tensor): covariance matrix :math:`\cov_X` of :math:`X`, with shape `(N, N)`.
        mu_y (torch.Tensor): mean :math:`\mu_Y` of multivariate Gaussian :math:`Y`, with shape `(N,)`.
        cov_y (torch.Tensor): covariance matrix :math:`\cov_Y` of :math:`Y`, with shape `(N, N)`.

    Returns:
        torch.Tensor: the Fréchet distance between :math:`X` and :math:`Y`.
    """
    if mu_x.ndim != 1:
        msg = f"Input mu_x must be one-dimensional; got dimension {mu_x.ndim}."
        raise ValueError(msg)
    if mu_y.ndim != 1:
        msg = f"Input mu_y must be one-dimensional; got dimension {mu_y.ndim}."
        raise ValueError(msg)
    if cov_x.ndim != 2:
        msg = f"Input cov_x must be two-dimensional; got dimension {cov_x.ndim}."
        raise ValueError(msg)
    if cov_y.ndim != 2:
        msg = f"Input cov_x must be two-dimensional; got dimension {cov_y.ndim}."
        raise ValueError(msg)
    if mu_x.shape != mu_y.shape:
        msg = f"Inputs mu_x and mu_y must have the same shape; got {mu_x.shape} and {mu_y.shape}."
        raise ValueError(msg)
    if cov_x.shape != cov_y.shape:
        msg = f"Inputs cov_x and cov_y must have the same shape; got {cov_x.shape} and {cov_y.shape}."
        raise ValueError(msg)

    a = (mu_x - mu_y).square().sum()
    b = cov_x.trace() + cov_y.trace()
    c = torch.linalg.eigvals(cov_x @ cov_y).sqrt().real.sum()
    return a + b - 2 * c
