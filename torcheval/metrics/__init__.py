# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from torcheval.metrics import functional
from torcheval.metrics.aggregation import (
    AUC,
    Cat,
    Covariance,
    Max,
    Mean,
    Min,
    Sum,
    Throughput,
)
from torcheval.metrics.audio import FrechetAudioDistance
from torcheval.metrics.classification import (
    BinaryAccuracy,
    BinaryAUPRC,
    BinaryAUROC,
    BinaryBinnedAUPRC,
    BinaryBinnedAUROC,
    BinaryBinnedPrecisionRecallCurve,
    BinaryConfusionMatrix,
    BinaryF1Score,
    BinaryNormalizedEntropy,
    BinaryPrecision,
    BinaryPrecisionRecallCurve,
    BinaryRecall,
    BinaryRecallAtFixedPrecision,
    MulticlassAccuracy,
    MulticlassAUPRC,
    MulticlassAUROC,
    MulticlassBinnedAUPRC,
    MulticlassBinnedAUROC,
    MulticlassBinnedPrecisionRecallCurve,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassPrecisionRecallCurve,
    MulticlassRecall,
    MultilabelAccuracy,
    MultilabelAUPRC,
    MultilabelBinnedAUPRC,
    MultilabelBinnedPrecisionRecallCurve,
    MultilabelPrecisionRecallCurve,
    MultilabelRecallAtFixedPrecision,
    TopKMultilabelAccuracy,
)

from torcheval.metrics.image import (
    FrechetInceptionDistance,
    PeakSignalNoiseRatio,
    StructuralSimilarity,
)
from torcheval.metrics.metric import Metric

from torcheval.metrics.ranking import (
    ClickThroughRate,
    HitRate,
    ReciprocalRank,
    RetrievalPrecision,
    RetrievalRecall,
    WeightedCalibration,
)
from torcheval.metrics.regression import MeanSquaredError, R2Score, PearsonCorrelationCoefficient

from torcheval.metrics.text import (
    BLEUScore,
    Perplexity,
    WordErrorRate,
    WordInformationLost,
    WordInformationPreserved,
)
from torcheval.metrics.window import (
    WindowedBinaryAUROC,
    WindowedBinaryNormalizedEntropy,
    WindowedClickThroughRate,
    WindowedMeanSquaredError,
    WindowedWeightedCalibration,
)

__all__ = [
    # base interface
    "Metric",
    # functional metrics
    "functional",
    # class metrics
    "AUC",
    "BinaryAccuracy",
    "BinaryAUPRC",
    "BinaryAUROC",
    "BinaryBinnedAUPRC",
    "BinaryBinnedAUROC",
    "BinaryBinnedPrecisionRecallCurve",
    "BinaryConfusionMatrix",
    "BinaryF1Score",
    "BinaryNormalizedEntropy",
    "BinaryPrecision",
    "BinaryPrecisionRecallCurve",
    "BinaryRecall",
    "BinaryRecallAtFixedPrecision",
    "BLEUScore",
    "Cat",
    "Covariance",
    "ClickThroughRate",
    "FrechetAudioDistance",
    "FrechetInceptionDistance",
    "HitRate",
    "Max",
    "Mean",
    "MeanSquaredError",
    "Min",
    "MulticlassAccuracy",
    "MulticlassAUPRC",
    "MulticlassAUROC",
    "MulticlassBinnedAUPRC",
    "MulticlassBinnedAUROC",
    "MulticlassBinnedPrecisionRecallCurve",
    "MulticlassConfusionMatrix",
    "MulticlassF1Score",
    "MulticlassPrecision",
    "MulticlassPrecisionRecallCurve",
    "MulticlassRecall",
    "MultilabelAccuracy",
    "MultilabelAUPRC",
    "MultilabelBinnedAUPRC",
    "MultilabelBinnedPrecisionRecallCurve",
    "MultilabelPrecisionRecallCurve",
    "MultilabelRecallAtFixedPrecision",
    "PearsonCorrelationCoefficient",
    "PeakSignalNoiseRatio",
    "Perplexity",
    "TopKMultilabelAccuracy",
    "R2Score",
    "ReciprocalRank",
    "RetrievalPrecision",
    "RetrievalRecall",
    "StructuralSimilarity",
    "Sum",
    "Throughput",
    "WeightedCalibration",
    "WindowedBinaryAUROC",
    "WindowedBinaryNormalizedEntropy",
    "WindowedClickThroughRate",
    "WindowedMeanSquaredError",
    "WindowedWeightedCalibration",
    "WordErrorRate",
    "WordInformationPreserved",
    "WordInformationLost",
]
