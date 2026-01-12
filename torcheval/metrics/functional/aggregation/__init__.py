# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from torcheval.metrics.functional.aggregation.auc import auc
from torcheval.metrics.functional.aggregation.mean import mean
from torcheval.metrics.functional.aggregation.sum import sum
from torcheval.metrics.functional.aggregation.throughput import throughput


__all__ = ["auc", "mean", "sum", "throughput"]
__doc_name__ = "Aggregation Metrics"
