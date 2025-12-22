# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from warnings import warn
import torch


@torch.inference_mode()
def pearson_correlation_coefficient(
    input: torch.Tensor,
    target: torch.Tensor,
    *,
    multioutput: str = "uniform_average",
) -> torch.Tensor:
    """
    Compute Pearson Correlation Coefficient.
    Its class version is ``torcheval.metrics.PearsonCorrelationCoefficient``.

    Args:
        input (Tensor): Tensor of predicted values with shape of (n_sample, n_output).
        target (Tensor): Tensor of ground truth values with shape of (n_sample, n_output).
        multioutput (Optional):
            - ``'uniform_average'`` [default]:
                Return scores of all outputs are averaged with uniform weight.
            - ``'raw_values'``:
                Return a full set of scores.
    Raises:
        ValueError:
            - If value of multioutput does not exist in (``raw_values``, ``uniform_average``).
            - If the dimension of `input` or `target` is not 1D or 2D.
            - If the `input` and `target` do not have the same size.

    Examples::

        >>> import torch
        >>> from torcheval.metrics.functional import pearson_correlation_coefficient
        >>> input = torch.tensor([0.9, 0.5, 0.3, 0.5])
        >>> target = torch.tensor([0.5, 0.8, 0.2, 0.8])
        >>> pearson_correlation_coefficient(input, target)
        tensor(0.2075)

        >>> input = torch.tensor([[0.9, 0.5], [0.3, 0.5]])
        >>> target = torch.tensor([[0.5, 0.8], [0.2, 0.8]])
        >>> pearson_correlation_coefficient(input, target)
        tensor(0.5000)

        >>> input = torch.tensor([[0.9, 0.5], [0.3, 0.5]])
        >>> target = torch.tensor([[0.5, 0.8], [0.2, 0.8]])
        >>> pearson_correlation_coefficient(input, target, multioutput="raw_values")
        tensor([1.0000, 0.0000])
    """
    _pearson_correlation_coefficient_param_check(multioutput)
    sum_input, sum_target, sum_input_target, sum_input_squared, sum_target_squared, num_samples = (
        _pearson_correlation_coefficient_update(input, target)
    )
    return _pearson_correlation_coefficient_compute(
        sum_input,
        sum_target,
        sum_input_target,
        sum_input_squared,
        sum_target_squared,
        num_samples,
        multioutput,
    )


def _pearson_correlation_coefficient_update(
    input: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Update the internal states for computing Pearson correlation coefficient.

    Args:
        input (Tensor): Tensor of predicted values with shape of (n_sample, n_output).
        target (Tensor): Tensor of ground truth values with shape of (n_sample, n_output).

    Returns:
        Tuple of tensors containing the sum of products, sum of inputs, sum of targets,
        sum of squared inputs, sum of squared targets, and count of samples.
    """
    _pearson_correlation_coefficient_update_input_check(input, target)
    return _update(input, target)


@torch.jit.script
def _update(
    input: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if input.ndim == 1:
        input = input.unsqueeze(1)
        target = target.unsqueeze(1)

    num_samples = torch.tensor(input.size(0), device=input.device)
    sum_input = input.sum(dim=0)
    sum_target = target.sum(dim=0)
    sum_input_target = (input * target).sum(dim=0)
    sum_input_squared = (input**2).sum(dim=0)
    sum_target_squared = (target**2).sum(dim=0)

    return (
        sum_input,
        sum_target,
        sum_input_target,
        sum_input_squared,
        sum_target_squared,
        num_samples,
    )


def _pearson_correlation_coefficient_compute(
    sum_input: torch.Tensor,
    sum_target: torch.Tensor,
    sum_input_target: torch.Tensor,
    sum_input_squared: torch.Tensor,
    sum_target_squared: torch.Tensor,
    num_samples: torch.Tensor,
    multioutput: str,
) -> torch.Tensor:
    """
    Compute the Pearson correlation coefficient based on saved statistics.

    Args:
        sum_input: Sum of inputs
        sum_target: Sum of targets
        sum_input_target: Sum of the product of input and target
        sum_input_squared: Sum of squared inputs
        sum_target_squared: Sum of squared targets
        num_samples: Number of samples
        multioutput: How to handle multiple outputs

    Returns:
        Pearson correlation coefficient
    """

    # Pearson correlation formula
    eps = torch.finfo(torch.float64).eps
    numerator = num_samples * sum_input_target - sum_input * sum_target
    denominator_input = torch.sqrt(num_samples * sum_input_squared - sum_input**2)
    denominator_target = torch.sqrt(num_samples * sum_target_squared - sum_target**2)
    denominator = denominator_input * denominator_target

    # Handle division by zero
    if (denominator_input < eps).any():
        warn("The variance of predictions is close to zero. This may result in numerical instability.")
    if (denominator_target < eps).any():
        warn("The variance of targets is close to zero. This may result in numerical instability.")
    correlation = torch.zeros_like(numerator)
    valid_indices = denominator > eps
    correlation[valid_indices] = numerator[valid_indices] / denominator[valid_indices]

    # Ensure results are within [-1, 1] due to potential numerical issues
    if (correlation < -1.0).any():
        warn("The Pearson correlation coefficient is less than -1. This may be due to numerical instability.")
    if (correlation > 1.0).any():
        warn("The Pearson correlation coefficient is greater than 1. This may be due to numerical instability.")
    correlation = torch.clamp(correlation, -1.0, 1.0)

    if multioutput == "raw_values":
        return correlation
    else:
        return correlation.mean()


def _pearson_correlation_coefficient_update_input_check(
    input: torch.Tensor,
    target: torch.Tensor,
) -> None:
    """Check input and target tensors for compatibility."""
    if input.ndim >= 3 or target.ndim >= 3:
        raise ValueError(
            "The dimension of `input` and `target` should be 1D or 2D, "
            f"got shapes {input.shape} and {target.shape}."
        )
    if input.size() != target.size():
        raise ValueError(
            "The `input` and `target` should have the same size, "
            f"got shapes {input.shape} and {target.shape}."
        )


def _pearson_correlation_coefficient_param_check(multioutput: str) -> None:
    """Check multioutput parameter for valid values."""
    if multioutput not in ("raw_values", "uniform_average"):
        raise ValueError(
            "The `multioutput` must be either `raw_values` or `uniform_average`, "
            f"got multioutput={multioutput}."
        )
