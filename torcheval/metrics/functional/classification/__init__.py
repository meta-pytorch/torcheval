# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from torcheval.metrics.functional.classification.accuracy import (
    binary_accuracy,
    multiclass_accuracy,
    multilabel_accuracy,
    topk_multilabel_accuracy,
)
from torcheval.metrics.functional.classification.auprc import (
    binary_auprc,
    multiclass_auprc,
    multilabel_auprc,
)
from torcheval.metrics.functional.classification.auroc import (
    binary_auroc,
    multiclass_auroc,
)
from torcheval.metrics.functional.classification.binary_normalized_entropy import (
    binary_normalized_entropy,
)
from torcheval.metrics.functional.classification.binned_auprc import (
    binary_binned_auprc,
    multiclass_binned_auprc,
    multilabel_binned_auprc,
)
from torcheval.metrics.functional.classification.binned_auroc import (
    binary_binned_auroc,
    multiclass_binned_auroc,
)
from torcheval.metrics.functional.classification.binned_precision_recall_curve import (
    binary_binned_precision_recall_curve,
    multiclass_binned_precision_recall_curve,
    multilabel_binned_precision_recall_curve,
)
from torcheval.metrics.functional.classification.confusion_matrix import (
    binary_confusion_matrix,
    multiclass_confusion_matrix,
)
from torcheval.metrics.functional.classification.f1_score import (
    binary_f1_score,
    multiclass_f1_score,
)
from torcheval.metrics.functional.classification.precision import (
    binary_precision,
    multiclass_precision,
)
from torcheval.metrics.functional.classification.precision_recall_curve import (
    binary_precision_recall_curve,
    multiclass_precision_recall_curve,
    multilabel_precision_recall_curve,
)
from torcheval.metrics.functional.classification.recall import (
    binary_recall,
    multiclass_recall,
)
from torcheval.metrics.functional.classification.recall_at_fixed_precision import (
    binary_recall_at_fixed_precision,
    multilabel_recall_at_fixed_precision,
)

__all__ = [
    "binary_accuracy",
    "binary_auprc",
    "binary_auroc",
    "binary_binned_auprc",
    "binary_binned_auroc",
    "binary_binned_precision_recall_curve",
    "binary_confusion_matrix",
    "binary_f1_score",
    "binary_normalized_entropy",
    "binary_precision",
    "binary_precision_recall_curve",
    "binary_recall",
    "binary_recall_at_fixed_precision",
    "multiclass_accuracy",
    "multiclass_auprc",
    "multiclass_auroc",
    "multiclass_binned_auprc",
    "multiclass_binned_auroc",
    "multiclass_binned_precision_recall_curve",
    "multiclass_confusion_matrix",
    "multiclass_f1_score",
    "multiclass_precision",
    "multiclass_precision_recall_curve",
    "multiclass_recall",
    "multilabel_accuracy",
    "multilabel_auprc",
    "multilabel_binned_auprc",
    "multilabel_binned_precision_recall_curve",
    "multilabel_precision_recall_curve",
    "multilabel_recall_at_fixed_precision",
    "topk_multilabel_accuracy",
]
__doc_name__ = "Classification Metrics"
