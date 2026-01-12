# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# pyre-ignore-all-errors[16]: Undefined attribute of metric states.

import logging
from collections.abc import Iterable
from typing import TypeVar

import torch
from torcheval.metrics.metric import Metric

TThroughput = TypeVar("TThroughput")

_logger: logging.Logger = logging.getLogger(__name__)


class Throughput(Metric[float]):
    """
    Calculate the throughput value which is the number of elements processed per second.

    Note: In a distributed setting, it's recommended to use `world_size * metric.compute()`
    to get an approximation of total throughput. While using `sync_and_compute(metric)` requires
    state sync. Additionally, `sync_and_compute(metric)` will give a slightly different value compared
    to `world_size * metric.compute()`.

    Examples::

        >>> import time
        >>> import torch
        >>> from torcheval.metrics import Throughput
        >>> metric = Throughput()
        >>> items_processed = 64
        >>> ts = time.monotonic()
        >>> time.sleep(2.0)  # simulate executing the program for 2 seconds
        >>> elapsed_time_sec = time.monotonic() - ts
        >>> metric.update(items_processed, elapsed_time_sec)
        >>> metric.compute()
        tensor(32.)
    """

    def __init__(
        self: TThroughput,
        *,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(device=device)
        self._add_state("num_total", 0.0)
        self._add_state("elapsed_time_sec", 0.0)

    @torch.inference_mode()
    # pyre-ignore[14]: inconsistent override on *_:Any, **__:Any
    def update(
        self: TThroughput,
        num_processed: int,
        elapsed_time_sec: float,
    ) -> TThroughput:
        """
        Update states with the values and weights.

        Args:
            num_processed: Number of items processed
            elapsed_time_sec: Total elapsed time in seconds to process ``num_processed`` items
        Raises:
            ValueError:
                If ``num_processed`` is a negative number.
                If ``elapsed_time_sec`` is a non-positive number.
        """
        if num_processed < 0:
            raise ValueError(
                f"Expected num_processed to be a non-negative number, but received {num_processed}."
            )
        if elapsed_time_sec <= 0:
            raise ValueError(
                f"Expected elapsed_time_sec to be a positive number, but received {elapsed_time_sec}."
            )

        self.elapsed_time_sec += elapsed_time_sec
        self.num_total += num_processed
        return self

    @torch.inference_mode()
    def compute(self: TThroughput) -> float:
        if not self.elapsed_time_sec:
            _logger.warning("No calls to update() have been made - returning 0.0")
            return 0.0

        return self.num_total / self.elapsed_time_sec

    @torch.inference_mode()
    def merge_state(self: TThroughput, metrics: Iterable[TThroughput]) -> TThroughput:
        for metric in metrics:
            self.num_total += metric.num_total
            # this assumes the metric is used within a fully-synchronous program.
            # In this scenario, the slowest process becomes the bottleneck for the
            # program's execution. As a result, we use the max, as the overall throughput
            # is gated based on the rank that takes the longest to complete.
            # TODO: should this be configurable?
            self.elapsed_time_sec = max(self.elapsed_time_sec, metric.elapsed_time_sec)
        return self
