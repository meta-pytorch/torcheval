# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from torcheval.metrics.text.bleu import BLEUScore
from torcheval.metrics.text.perplexity import Perplexity
from torcheval.metrics.text.word_error_rate import WordErrorRate
from torcheval.metrics.text.word_information_lost import WordInformationLost
from torcheval.metrics.text.word_information_preserved import WordInformationPreserved

__all__ = [
    "BLEUScore",
    "Perplexity",
    "WordErrorRate",
    "WordInformationLost",
    "WordInformationPreserved",
]
__doc_name__ = "Text Metrics"
