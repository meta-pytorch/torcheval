# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from torcheval.metrics.functional.text.bleu import bleu_score
from torcheval.metrics.functional.text.perplexity import perplexity
from torcheval.metrics.functional.text.word_error_rate import word_error_rate
from torcheval.metrics.functional.text.word_information_lost import (
    word_information_lost,
)
from torcheval.metrics.functional.text.word_information_preserved import (
    word_information_preserved,
)

__all__ = [
    "bleu_score",
    "perplexity",
    "word_error_rate",
    "word_information_preserved",
    "word_information_lost",
]
__doc_name__ = "Text Metrics"
