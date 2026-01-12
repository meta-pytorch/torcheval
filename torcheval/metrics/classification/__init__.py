# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from torcheval.metrics.classification.accuracy import (
    BinaryAccuracy,
    MulticlassAccuracy,
    MultilabelAccuracy,
    TopKMultilabelAccuracy,
)
from torcheval.metrics.classification.auprc import (
    BinaryAUPRC,
    MulticlassAUPRC,
    MultilabelAUPRC,
)
from torcheval.metrics.classification.auroc import BinaryAUROC, MulticlassAUROC
from torcheval.metrics.classification.binary_normalized_entropy import (
    BinaryNormalizedEntropy,
)
from torcheval.metrics.classification.binned_auprc import (
    BinaryBinnedAUPRC,
    MulticlassBinnedAUPRC,
    MultilabelBinnedAUPRC,
)
from torcheval.metrics.classification.binned_auroc import (
    BinaryBinnedAUROC,
    MulticlassBinnedAUROC,
)
from torcheval.metrics.classification.binned_precision_recall_curve import (
    BinaryBinnedPrecisionRecallCurve,
    MulticlassBinnedPrecisionRecallCurve,
    MultilabelBinnedPrecisionRecallCurve,
)
from torcheval.metrics.classification.confusion_matrix import (
    BinaryConfusionMatrix,
    MulticlassConfusionMatrix,
)
from torcheval.metrics.classification.f1_score import BinaryF1Score, MulticlassF1Score
from torcheval.metrics.classification.precision import (
    BinaryPrecision,
    MulticlassPrecision,
)
from torcheval.metrics.classification.precision_recall_curve import (
    BinaryPrecisionRecallCurve,
    MulticlassPrecisionRecallCurve,
    MultilabelPrecisionRecallCurve,
)
from torcheval.metrics.classification.recall import BinaryRecall, MulticlassRecall
from torcheval.metrics.classification.recall_at_fixed_precision import (
    BinaryRecallAtFixedPrecision,
    MultilabelRecallAtFixedPrecision,
)

__all__ = [
    "BinaryAccuracy",
    "BinaryAUPRC",
    "BinaryAUROC",
    "BinaryBinnedAUROC",
    "BinaryBinnedAUPRC",
    "BinaryBinnedPrecisionRecallCurve",
    "BinaryConfusionMatrix",
    "BinaryF1Score",
    "BinaryNormalizedEntropy",
    "BinaryPrecision",
    "BinaryPrecisionRecallCurve",
    "BinaryRecall",
    "BinaryRecallAtFixedPrecision",
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
    "TopKMultilabelAccuracy",
]

__doc_name__ = "Classification Metrics"
