# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch


@torch.inference_mode()
def throughput(
    num_processed: int = 0,
    elapsed_time_sec: float = 0.0,
) -> torch.Tensor:
    """
    Calculate the throughput value which is the number of elements processed per second.
    Its class version is ``torcheval.metrics.Throughput``.

    Args:
        num_processed (int): Number of items processed.
        elapsed_time_sec (float): Total elapsed time in seconds to process ``num_processed`` items.
    Raises:
        ValueError:
                If ``num_processed`` is a negative number.
                If ``elapsed_time_sec`` is a non-positive number.

    Examples::

        >>> import torch
        >>> from torcheval.metrics.functional import throughput
        >>> throughput(64, 2.0)
        tensor(32.)
    """
    return _throughput_compute(num_processed, elapsed_time_sec)


def _throughput_compute(num_processed: int, elapsed_time_sec: float) -> torch.Tensor:
    if num_processed < 0:
        raise ValueError(
            f"Expected num_processed to be a non-negative number, but received {num_processed}."
        )
    if elapsed_time_sec <= 0:
        raise ValueError(
            f"Expected elapsed_time_sec to be a positive number, but received {elapsed_time_sec}."
        )
    return torch.tensor(num_processed / elapsed_time_sec)
