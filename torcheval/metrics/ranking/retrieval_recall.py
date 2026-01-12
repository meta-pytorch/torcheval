# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# pyre-ignore-all-errors[16]: Undefined attribute of metric states.

from collections.abc import Iterable
from typing import Literal, TypeVar

import torch
from torcheval.metrics.functional.ranking.retrieval_recall import (
    _retrieval_recall_param_check,
    _retrieval_recall_update_input_check,
    get_topk,
    retrieval_recall,
)
from torcheval.metrics.metric import Metric


TRetrievalRecall = TypeVar("RetrievalRecall")


class RetrievalRecall(Metric[torch.Tensor]):
    """
    Compute the retrieval recall.
    Its functional version is :func:`torcheval.metrics.functional.retrieval_recall`.
    (Here, `input` and `target` refer to the arguments of `update` function.)

    Args:
        k (int, optional):
            the number of elements considered as being retrieved. Only the top (sorted in decreasing order) `k` elements of `input` are considered.
            if `k` is None, all the `input` elements are considered.
        limit_k_to_size (bool, default value: False):
            When set to `True`, limits `k` to be at most the length of `input`, i.e. replaces `k` by `k=min(k, len(input))`.
            This parameter can only be set to `True` if `k` is not None.
        empty_target_action (str, choose among ["neg", "pos", "skip", "err"], default: "neg"):
            Choose the behaviour of `update` function when `target` does not contain at least one positive element:
            - when 'neg': retrieval recall is equal to 0.0,
            - when 'pos': retrieval recall is equal to 1.0,
            - when 'skip': retrieval recall is equal to NaN.
            - when 'err': raise a ValueError.
        num_queries (int, default value: 1):
            If >1, `inputs` and `targets` can contain entries related to different queries.
            An `indexes` tensor must be passed during updates which associates each `input` and `target` to an integer between 0 and `num_queries`-1.
            Outputs for each query are computed independently and `.compute()` will return a tensor of shape `(num_queries,)`.
        avg (str, choose among ["macro", "none", None], default: "None"):
            Choose the averaging method over all queries:
            - when "none" or None: `.compute()` returns a tensor of shape `(num_queries,)`, which ith value is equal to the retrieval recall of ith query.
            - when "macro": `.compute()` returns the average retrieval recall over all queries.
        device: Optional[torch.device]:
            choose the torch device to be used.

    Examples:
        >>> import torch
        >>> from torcheval.metrics import RetrievalRecall

        >>> input = torch.tensor([0.2, 0.3, 0.5, 0.1, 0.3, 0.5, 0.2])
        >>> target = torch.tensor([0, 0, 1, 1, 1, 0, 1])

        >>> metric = RetrievalRecall(k=2)
        >>> metric.update(input, target)
        >>> metric.compute()
        tensor(0.25)

    Raises:
        ValueError:
            if `empty_target_action` is not one of "neg", "pos", "skip", "err".
        ValueError:
            if `limit_k_to_size` is True and `k` is None.
        ValueError:
            if `k` is not a positive integer.
        ValueError:
            if `empty_target_action` == "err" and self.update is called with a target which entries are all equal to 0.
        ValueError:
            if input or target arguments of self.update are Tensors with different dimensions or dimension != 1.
        ValueError:
            if `num_queries` > 1 and argument `indexes` of function .update() is `None`.
    """

    def __init__(
        self: TRetrievalRecall,
        *,
        empty_target_action: (
            Literal["neg"] | Literal["pos"] | Literal["skip"] | Literal["err"]
        ) = "neg",
        k: int | None = None,
        limit_k_to_size: bool = False,
        num_queries: int = 1,
        avg: Literal["macro"] | Literal["none"] | None = None,
        device: torch.device | None = None,
    ) -> None:
        _retrieval_recall_param_check(k, limit_k_to_size)
        super().__init__(device=device)
        self.empty_target_action = empty_target_action
        self.num_queries = num_queries
        self.k = k
        self.limit_k_to_size = limit_k_to_size
        self.avg = avg
        self._add_state("topk", [torch.empty(0) for _ in range(num_queries)])
        self._add_state("target", [torch.empty(0) for _ in range(num_queries)])

    @torch.inference_mode()
    # pyre-ignore[14]: `update` overrides method defined in `Metric` inconsistently.
    def update(
        self: TRetrievalRecall,
        input: torch.Tensor,
        target: torch.Tensor,
        indexes: torch.Tensor | None = None,
    ) -> TRetrievalRecall:
        """
        Update the metric state with ground truth labels and predictions.
        """
        _retrieval_recall_update_input_check(
            input, target, num_queries=self.num_queries, indexes=indexes
        )
        if self.num_queries == 1:
            self.update_single_query(0, input, target)
            return self

        if indexes is None:
            raise ValueError(
                "`indexes` must be passed during update() when num_queries > 1."
            )
        for i in range(self.num_queries):
            if i in indexes:
                self.update_single_query(i, input[indexes == i], target[indexes == i])

        return self

    def update_single_query(
        self, i: int, input: torch.Tensor, target: torch.Tensor
    ) -> None:
        batch_preds = torch.cat([self.topk[i], input])
        batch_targets = torch.cat([self.target[i], target])
        preds_topk = get_topk(batch_preds, self.k)
        self.topk[i] = preds_topk[0]
        self.target[i] = batch_targets.gather(dim=-1, index=preds_topk[1])

    @torch.inference_mode()
    def compute(self: TRetrievalRecall) -> torch.Tensor:
        rp = []
        for i in range(self.num_queries):
            if not len(self.target[i]):
                rp.append(torch.tensor([torch.nan]))
            elif 1 not in self.target[i]:
                if self.empty_target_action == "pos":
                    rp.append(torch.tensor([1.0]))
                elif self.empty_target_action == "neg":
                    rp.append(torch.tensor([0.0]))
                elif self.empty_target_action == "skip":
                    rp.append(torch.tensor([torch.nan]))
                elif self.empty_target_action == "err":
                    raise ValueError(
                        f"no positive value found in target={self.target[i]}."
                    )
            else:
                rp.append(
                    retrieval_recall(
                        self.topk[i], self.target[i], self.k, self.limit_k_to_size
                    ).reshape(-1)
                )
        rp = torch.cat(rp).to(self.device)
        if self.avg == "macro":
            return rp.nanmean()
        else:
            return rp

    @torch.inference_mode()
    def merge_state(
        self: TRetrievalRecall, metrics: Iterable[TRetrievalRecall]
    ) -> TRetrievalRecall:
        """
        Merge the metric state with its counterparts from other metric instances.

        Args:
            metrics (Iterable[Metric]): metric instances whose states are to be merged.
        """

        for i in range(self.num_queries):
            self.topk[i] = torch.cat([self.topk[i]] + [m.topk[i] for m in metrics]).to(
                self.device
            )
            self.target[i] = torch.cat(
                [self.target[i]] + [m.target[i] for m in metrics]
            ).to(self.device)

        return self
