# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from torcheval.metrics.regression.mean_squared_error import MeanSquaredError
from torcheval.metrics.regression.r2_score import R2Score

__all__ = ["MeanSquaredError", "R2Score"]
__doc_name__ = "Regression Metrics"
