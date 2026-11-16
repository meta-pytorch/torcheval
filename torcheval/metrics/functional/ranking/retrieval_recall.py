# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch


@torch.inference_mode()
def retrieval_recall(
    input: torch.Tensor,
    target: torch.Tensor,
    k: int | None = None,
    limit_k_to_size: bool = False,
    num_tasks: int = 1,
) -> torch.Tensor:
    """
    Retrieval Recall is a metric that measures the proportion of relevant items retrieved out of the all relevant items.

    It is defined as:
    Retrieval Recall = (Number of Relevant Items Retrieved) / (Total Number of All Relevant Items)
    This metric is also known as Recall at k, where k is the number of elements considered as being retrieved.

    Its class version is :class:`torcheval.metrics.ranking.RetrievalRecall`.

    Args:
        input (Tensor):
            Predicted scores for each document (the higher the more relevant), with shape (num_sample,) or (num_tasks, num_samples).
        target (Tensor):
            0 and 1 valued Tensor of ground truth identifying relevant element, with shape (num_sample,) or (num_tasks, num_samples).
        k (int, optional):
            the number of elements considered as being retrieved. Only the top (sorted in decreasing order) `k` elements of `input` are considered.
            if `k` is None, all the `input` elements are considered.
        limit_k_to_size (bool, default value: False):
            When set to `True`, limits `k` to be at most the length of `input`, i.e. replaces `k` by `k=min(k, len(input))`.
            This parameter can only be set to `True` if `k` is not None.
        num_tasks (int, default value: 1): Number of tasks that need retrieval_recall calculation.
    Returns:
       (Tensor):
            - If input and target are 1D: returns a tensor of dimension 0, containing the retrieval recall value.
            - When input and target are 2D with shape (num_tasks, num_samples): returns a tensor of shape (num_tasks,) containing the retrieval recall, computed row by row.

    Examples:
        >>> import torch
        >>> from torcheval.metrics.functional.ranking import retrieval_recall

        >>> input = torch.Tensor([0.2, 0.3, 0.5, 0.1, 0.3, 0.5, 0.2])
        >>> target = torch.Ttensor([0, 0, 1, 1, 1, 0, 1])
        >>> retrieval_recall(input, target)
        torch.Tensor(1.0)
        >>> retrieval_recall(input, target, k=2)
        torch.Tensor(0.25)

    Raises:
        ValueError:
            if `limit_k_to_size` is True and `k` is None.
        ValueError:
            if `k` is not a positive integer.
        ValueError:
            if input or target arguments of self._retrieval_recall_compute are Tensors with dimension 0 or > 2.
    """
    _retrieval_recall_param_check(k, limit_k_to_size)
    _retrieval_recall_update_input_check(input, target, num_tasks)
    return _retrieval_recall_compute(
        input=input,
        target=target,
        k=k,
        limit_k_to_size=limit_k_to_size,
    )


def _retrieval_recall_param_check(
    k: int | None = None, limit_k_to_size: bool = False
) -> None:
    if k is not None and k <= 0:
        raise ValueError(f"k must be a positive integer, got k={k}.")

    if limit_k_to_size and k is None:
        raise ValueError(
            "when limit_k_to_size is True, k must be a positive (>0) integer."
        )


def _retrieval_recall_update_input_check(
    input: torch.Tensor,
    target: torch.Tensor,
    num_tasks: int = 1,
    indexes: torch.Tensor | None = None,
    num_queries: int = 1,
) -> None:
    if input.shape != target.shape:
        raise ValueError(
            f"input and target must be of the same shape, got input.shape={input.shape} and target.shape={target.shape}."
        )
    if num_tasks == 1:
        if input.dim() != 1:
            raise ValueError(
                f"input and target should be one dimensional tensors, got input and target dimensions={input.dim()}."
            )
    else:
        if input.dim() != 2 or input.shape[0] != num_tasks:
            raise ValueError(
                f"input and target should be two dimensional tensors with {num_tasks} rows, got input and target shape={input.shape}."
            )


def _retrieval_recall_compute(
    input: torch.Tensor,
    target: torch.Tensor,
    k: int | None = None,
    limit_k_to_size: bool = False,
) -> torch.Tensor:
    nb_relevant_items = compute_nb_relevant_items_retrieved(input, k, target)
    return nb_relevant_items / target.sum(dim=-1)


def compute_nb_relevant_items_retrieved(
    input: torch.Tensor,
    k: int | None,
    target: torch.Tensor,
) -> torch.Tensor:
    return target.gather(dim=-1, index=get_topk(input, k)[1]).sum(dim=-1)


def get_topk(t: torch.Tensor, k: int | None) -> tuple[torch.Tensor, torch.LongTensor]:
    nb_samples = t.size(-1)
    if k is None:
        k = nb_samples
    # take the topk values of input. /!\ Ties are sorted in an unpredictable way.
    return t.topk(min(k, nb_samples), dim=-1)
