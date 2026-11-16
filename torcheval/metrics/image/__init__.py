# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from torcheval.metrics.image.fid import FrechetInceptionDistance
from torcheval.metrics.image.psnr import PeakSignalNoiseRatio
from torcheval.metrics.image.ssim import StructuralSimilarity

__all__ = [
    "FrechetInceptionDistance",
    "PeakSignalNoiseRatio",
    "StructuralSimilarity",
]
__doc_name__ = "Image Metrics"
