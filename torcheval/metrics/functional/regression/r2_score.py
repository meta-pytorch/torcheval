# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# pyre-ignore-all-errors[16]: Undefined attribute of metric states.


import torch
from torcheval.metrics.functional._jit import _script_on_use


@torch.inference_mode()
def r2_score(
    input: torch.Tensor,
    target: torch.Tensor,
    *,
    multioutput: str = "uniform_average",
    num_regressors: int = 0,
) -> torch.Tensor:
    """
    Compute R-squared score, which is the proportion of variance in the dependent variable that can be explained by the independent variable.
    Its class version is ``torcheval.metrics.R2Score``.

    Args:
        input:
            Tensor of predicted values with shape of (n_sample, n_output).
        target:
            Tensor of ground truth values with shape of (n_sample, n_output).
        multioutput (Optional):
            - ``'uniform_average'`` [default]:
                Return scores of all outputs are averaged with uniform weight.
            - ``'raw_values'``:
                Return a full set of scores.
            - ``variance_weighted``:
                Return scores of all outputs are averaged with weighted by the variances of each individual output.
        num_regressors (Optional):
            Number of independent variables used, applied to adjusted R-squared score. Defaults to zero (standard R-squared score).
    Raises:
        ValueError:
            - If value of multioutput does not exist in (``raw_values``, ``uniform_average``, ``variance_weighted``).
            - If value of num_regressors is not an ``integer`` in the range of [0, n_samples - 1].

    Examples::

        >>> import torch
        >>> from torcheval.metrics.functional import r2_score
        >>> input = torch.tensor([0, 2, 1, 3])
        >>> target = torch.tensor([0, 1, 2, 3])
        >>> r2_score(input, target)
        tensor(0.6)

        >>> input = torch.tensor([[0, 2], [1, 6]])
        >>> target = torch.tensor([[0, 1], [2, 5]])
        >>> r2_score(input, target)
        tensor(0.6250)

        >>> input = torch.tensor([[0, 2], [1, 6]])
        >>> target = torch.tensor([[0, 1], [2, 5]])
        >>> r2_score(input, target, multioutput="raw_values")
        tensor([0.5000, 0.7500])

        >>> input = torch.tensor([[0, 2], [1, 6]])
        >>> target = torch.tensor([[0, 1], [2, 5]])
        >>> r2_score(input, target, multioutput="variance_weighted")
        tensor(0.7000)

        >>> input = torch.tensor([1.2, 2.5, 3.6, 4.5, 6])
        >>> target = torch.tensor([1, 2, 3, 4, 5])
        >>> r2_score(input, target, multioutput="raw_values", num_regressors=2)
        tensor(0.6200)
    """

    _r2_score_param_check(multioutput, num_regressors)
    sum_squared_obs, sum_obs, sum_squared_residual, num_obs = _r2_score_update(
        input, target
    )
    return _r2_score_compute(
        sum_squared_obs,
        sum_obs,
        sum_squared_residual,
        num_obs,
        multioutput,
        num_regressors,
    )


def _r2_score_update(
    input: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _r2_score_update_input_check(input, target)
    return _script_on_use(_update)(input, target)


def _update(
    input: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sum_squared_obs = torch.sum(torch.square(target), dim=0)
    sum_obs = torch.sum(target, dim=0)
    sum_squared_residual = torch.sum(torch.square(target - input), dim=0)
    num_obs = torch.tensor(target.size(0))
    return sum_squared_obs, sum_obs, sum_squared_residual, num_obs


def _r2_score_compute(
    sum_squared_obs: torch.Tensor,
    sum_obs: torch.Tensor,
    rss: torch.Tensor,
    num_obs: torch.Tensor,
    multioutput: str,
    num_regressors: int,
) -> torch.Tensor:
    if num_obs < 2:
        raise ValueError(
            "There is no enough data for computing. Needs at least two samples to calculate r2 score."
        )
    if num_regressors >= num_obs - 1:
        raise ValueError(
            "The `num_regressors` must be smaller than n_samples - 1, "
            f"got num_regressors={num_regressors}, n_samples={num_obs}.",
        )
    return _script_on_use(_compute)(
        sum_squared_obs,
        sum_obs,
        rss,
        num_obs,
        multioutput,
        num_regressors,
    )


def _compute(
    sum_squared_obs: torch.Tensor,
    sum_obs: torch.Tensor,
    rss: torch.Tensor,
    num_obs: torch.Tensor,
    multioutput: str,
    num_regressors: int,
) -> torch.Tensor:
    tss = sum_squared_obs - torch.square(sum_obs) / num_obs
    # Calculate R2 score when multioutput is equal to raw_values.
    r_squared = 1 - (rss / tss)
    if multioutput == "uniform_average":
        r_squared = torch.mean(r_squared)
    elif multioutput == "variance_weighted":
        r_squared = torch.sum(r_squared * tss / torch.sum(tss))

    # If num_regressors is not equal to 0, adjusted R2 applies.
    if num_regressors != 0:
        r_squared = 1 - (1 - r_squared) * (num_obs - 1) / (num_obs - num_regressors - 1)
    return r_squared


def _r2_score_param_check(
    multioutput: str,
    num_regressors: int,
) -> None:
    if multioutput not in ("raw_values", "uniform_average", "variance_weighted"):
        raise ValueError(
            "The `multioutput` must be either `raw_values` or `uniform_average` or `variance_weighted`, "
            f"got multioutput={multioutput}."
        )
    if not isinstance(num_regressors, int) or num_regressors < 0:
        raise ValueError(
            "The `num_regressors` must an integer larger or equal to zero, "
            f"got num_regressors={num_regressors}."
        )


def _r2_score_update_input_check(
    input: torch.Tensor,
    target: torch.Tensor,
) -> None:
    if input.ndim >= 3 or target.ndim >= 3:
        raise ValueError(
            "The dimension `input` and `target` should be 1D or 2D, "
            f"got shapes {input.shape} and {target.shape}."
        )
    if input.size() != target.size():
        raise ValueError(
            "The `input` and `target` should have the same size, "
            f"got shapes {input.shape} and {target.shape}."
        )
