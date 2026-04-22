# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
This file contains functions to synchronize metric states across processes, to be called within toolkit.py

The overall process goes as follows:

1) Metrics are converted to their state dict representations
2) A traversal order is established through these state dicts (by alphabetical order of keys)
3) An empty dictionary to store states received from other processes is created (this is `gathered_states` in the code)
4) The traversal begins, for each metric's state:
    - validates it is one of Tensor, List[Tensor], Dict[str, Tensor], int, or float type
    - calls the appropriate helper sync function to receive and store that state from all other processes
5) Returns these states to toolkit
"""

import logging
from typing import Any

import torch
from pyre_extensions import none_throws
from torch import distributed as dist, Tensor
from torch.nn import functional as F
from torcheval.metrics.metric import TState

_logger: logging.Logger = logging.getLogger(__name__)


def _simple_send_tensors(
    tensor: Tensor,
    world_size: int,
    group: dist.ProcessGroup | None,
    rank: int | None,
) -> list[Tensor] | None:
    """
    Helper function that sends tensor to specified rank (or all ranks), and
    returns the received results. Assumes tensors have same dimensions.

    Return:
        gathered_result: list with size equal to the process group where
            gathered_result[i] corresponds to result tensor from process i
    """
    gathered_result = None
    local_rank = dist.get_rank(group=group)

    if rank is None or local_rank == rank:
        # only construct ``gathered_result`` if rank is going to receive data
        stacked_result_sizes = [world_size] + list(tensor.size())
        gathered_result = list(
            torch.empty(stacked_result_sizes, dtype=tensor.dtype, device=tensor.device)
        )

    if rank is None:
        # sync tensors to all ranks
        dist.all_gather(gathered_result, tensor, group=group)
    else:
        # sync tensors only to specified rank
        dist.gather(tensor, gathered_result, dst=rank, group=group)

    return gathered_result


def _send_uneven_tensors(
    tensor: Tensor,
    world_size: int,
    group: dist.ProcessGroup | None,
    rank: int | None,
) -> list[Tensor] | None:
    """
    Helper function that sends tensor to specified rank (or all ranks), and
    returns the received results. If tensor dimensions differ across ranks,
    tensors are padded, gathered and then trimmed to secure equal workload
    for all processes.

    Return:
        gathered_result: list with size equal to the process group where
            gathered_result[i] corresponds to result tensor from process i
    """
    # TODO investigate why this stopped working (can cause hangs)
    # if the backend is NCCL, we can gather the differently sized tensors without padding
    # if dist.get_backend(group) == "nccl":
    #     return _simple_send_tensors(tensor, world_size, group, rank=rank)

    # gather sizes of all tensors on all ranks
    local_size = torch.tensor(tensor.shape, device=tensor.device)
    local_sizes = _simple_send_tensors(local_size, world_size, group, rank=None)
    assert local_sizes is not None

    # if shapes are all the same, then do a simple gather:
    stacked_sizes = torch.stack(local_sizes)
    max_size = stacked_sizes.max(dim=0).values
    min_size = stacked_sizes.min(dim=0).values
    all_sizes_equal = torch.equal(max_size, min_size)
    if all_sizes_equal:
        return _simple_send_tensors(tensor, world_size, group, rank=rank)

    # if not, we need to pad each local tensor to maximum size, gather and then truncate
    pad_dims = []
    pad_by = (max_size - local_size).detach().cpu()
    for val in reversed(pad_by):
        pad_dims.append(0)
        pad_dims.append(val.item())
    result_padded = F.pad(tensor, pad_dims)

    gathered_result = _simple_send_tensors(result_padded, world_size, group, rank=rank)

    if gathered_result:
        for idx, item_size in enumerate(local_sizes):
            slice_param = [slice(dim_size) for dim_size in item_size]
            gathered_result[idx] = gathered_result[idx][slice_param]

    return gathered_result


def send_tensors(
    result: Tensor,
    group: dist.ProcessGroup | None = None,
    rank: int | None = None,
) -> list[Tensor] | None:
    """Function to gather tensors from several distributed processes onto a list that is broadcasted to specified processes.
    Works on tensors that have the same number of dimensions, but where each dimension may differ. In this case
    tensors are padded, gathered and then trimmed to secure equal workload for all processes.

    Args:
        result: the value to sync
        group: the process group to gather results from. Defaults to all processes (world)
        rank: the rank to gather results from. If none, gathers on all ranks

    Return:
        gathered_result: list with size equal to the process group where
            gathered_result[i] corresponds to result tensor from process i
    """
    # if torch.distributed is not available or not initialized
    # return single-item list containing the result
    if not dist.is_available() or not dist.is_initialized():
        return [result]

    # convert tensors to contiguous format
    result = result.contiguous()
    world_size = dist.get_world_size(group)

    # if the tensor is scalar, things are easy
    if result.ndim == 0:
        return _simple_send_tensors(result, world_size, group, rank=rank)

    return _send_uneven_tensors(result, world_size, group, rank=rank)


def metrics_traversal_order(
    state_dict: dict[str, dict[str, TState]],
) -> list[tuple[str, str]]:
    """
    Args:
        state_dict: a dictionary of metric states (metric name -> metric's state dict).

    Returns: a list of tuple defining how to traverse a group of metrics deterministically.
        ie ((metric1 name, metric1 state1), (metric1 name, metric1 state2), (metric2 name, metric2 state1), ...)
    """
    dict_items = []
    for outer_key in sorted(state_dict.keys()):
        inner_dict = state_dict[outer_key]
        for inner_key in sorted(inner_dict.keys()):
            dict_items.append((outer_key, inner_key))
    return dict_items


def _get_empty_metric_state_collection(
    metrics_traversal_order: list[tuple[str, str]],
) -> dict[str, dict[str, Any]]:
    metric_state_collection = {}
    for metric_name, state_name in metrics_traversal_order:
        if metric_name not in metric_state_collection.keys():
            metric_state_collection[metric_name] = {}
        metric_state_collection[metric_name][state_name] = {}
    return metric_state_collection


def _sync_tensor_states(
    metric_name: str,
    state_name: str,
    my_state_data: torch.Tensor,
    gathered_states: list[dict[str, dict[str, Any]]],
    process_group: dist.ProcessGroup | None,
    rank: int | None,
) -> None:
    gathered_state_data = send_tensors(my_state_data, group=process_group, rank=rank)

    if gathered_state_data is None:
        return

    for i, state_tensor in enumerate(gathered_state_data):
        gathered_states[i][metric_name][state_name] = state_tensor


def _sync_dtype_and_shape(
    tensor: torch.Tensor | None,
    process_group: dist.ProcessGroup | None,
) -> tuple[torch.dtype, torch.Size] | None:
    my_rank = dist.get_rank(group=process_group)
    world_size = dist.get_world_size(group=process_group)

    rank_with_dtype = -1
    if tensor is not None:
        rank_with_dtype = my_rank  # send its rank

    object_list = [None for _ in range(world_size)]
    dist.all_gather_object(object_list, rank_with_dtype, group=process_group)
    # pyre-ignore: Incompatible parameter type [6]
    rank_with_dtype = max(object_list)  # choose highest rank to broadcast dtype

    if rank_with_dtype == -1:
        # all ranks have passed None for tensor
        return None

    # Broadcast the dtype from the rank that knows it to all other processes
    if my_rank == rank_with_dtype:
        # pyre-ignore: Undefined attribute [16]
        object_list = [(tensor.dtype, tensor.shape)]
    else:
        object_list = [None]

    dist.broadcast_object_list(object_list, src=rank_with_dtype, group=process_group)
    # pyrefly: ignore [not-iterable]
    dtype, shape = object_list[0]
    return dtype, shape


def _sync_list_length(
    state_data: list[torch.Tensor],
    process_group: dist.ProcessGroup | None,
) -> list[int]:
    my_length = len(state_data)
    world_size = dist.get_world_size(group=process_group)

    lengths = [None for _ in range(world_size)]
    dist.all_gather_object(lengths, my_length, group=process_group)

    # pyre-ignore: Incompatible parameter type [7]
    return lengths


def _generate_dummy_tensor(
    shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    return torch.empty(shape, dtype=dtype, device=device)


def _sync_list_tensor_states(
    metric_name: str,
    state_name: str,
    my_state_data: list[torch.Tensor],
    device: torch.device,
    gathered_states: list[dict[str, dict[str, Any]]],
    process_group: dist.ProcessGroup | None,
    rank: int | None,
) -> None:
    # get length of lists across all ranks
    gathered_list_lengths = _sync_list_length(
        my_state_data, process_group=process_group
    )
    if any(length == 0 for length in gathered_list_lengths):
        # one or more ranks has empty list, need to sync dtype and shape
        # so those ranks can send appropriate dummy tensor for allgather

        # if at least one element, send your tensor dtype and shape
        # otherwise send None to indicate your list is empty
        result = _sync_dtype_and_shape(
            my_state_data[0] if len(my_state_data) > 0 else None,
            process_group=process_group,
        )
        if result is None:
            # all ranks state_data is empty, no need to sync
            return
        dtype, shape = result  # unpack results
    else:
        # pull dtype and ndim from local data
        shape = my_state_data[0].shape
        dtype = my_state_data[0].dtype

    max_length = max(gathered_list_lengths)

    # go through each element in list and sync the tensors
    for i in range(max_length):
        # if local rank exceeded length of its list, send dummy tensor instead
        tensor_to_send = (
            _generate_dummy_tensor(shape, dtype, device)
            if i >= len(my_state_data)
            else my_state_data[i]
        )

        gathered_state_data = send_tensors(
            tensor_to_send, group=process_group, rank=rank
        )

        if gathered_state_data is None:
            continue

        for _rank, state_tensor in enumerate(gathered_state_data):
            if len(gathered_states[_rank][metric_name][state_name]) == 0:
                # initialize to list
                gathered_states[_rank][metric_name][state_name] = []
            # append the received state tensor to rank unless its list length is reached
            if i < gathered_list_lengths[_rank]:
                gathered_states[_rank][metric_name][state_name].append(state_tensor)


def _sync_dict_tensor_states(
    metric_name: str,
    state_name: str,
    my_state_data: dict[str, torch.Tensor],
    device: torch.device,
    gathered_states: list[dict[str, dict[str, Any]]],
    process_group: dist.ProcessGroup | None,
    rank: int | None,
) -> None:
    sorted_keys = sorted(my_state_data.keys())
    tensor_list = [my_state_data[key] for key in sorted_keys]
    _sync_list_tensor_states(
        metric_name,
        state_name,
        tensor_list,
        device,
        gathered_states,
        process_group,
        rank,
    )

    if rank is None or dist.get_rank(group=process_group) == rank:
        # pack ``gathered_states`` into dictionary
        for rank in range(len(gathered_states)):
            tensor_list = gathered_states[rank][metric_name][state_name]
            gathered_states[rank][metric_name][state_name] = dict(
                zip(sorted_keys, tensor_list)
            )


# TODO: Remove use of gather_object and use tensors instead to avoid unnecessary gpu -> cpu
def _sync_obj_states(
    metric_name: str,
    state_name: str,
    # pyre-ignore: Missing parameter annotation [2]
    my_state_data: Any,
    gathered_states: list[dict[str, dict[str, Any]]],
    process_group: dist.ProcessGroup | None,
    rank: int | None,
) -> None:
    local_rank = dist.get_rank(group=process_group)
    world_size = dist.get_world_size(group=process_group)

    create_obj_gather_list = rank is None or rank == local_rank
    gathered_obj_data = [None] * world_size if create_obj_gather_list else None

    if rank is None:
        # if rank not specified, sync all ranks
        dist.all_gather_object(gathered_obj_data, my_state_data, group=process_group)
    else:
        # if rank is specified, send object only to that rank
        dist.gather_object(
            my_state_data, gathered_obj_data, dst=rank, group=process_group
        )

    if create_obj_gather_list:
        gathered_obj_data = none_throws(gathered_obj_data)
        for i, obj in enumerate(gathered_obj_data):
            gathered_states[i][metric_name][state_name] = obj


def sync_states(
    states: dict[str, dict[str, Any]],
    devices: dict[str, torch.device],
    metrics_traversal_order: list[tuple[str, str]],
    process_group: dist.ProcessGroup | None = None,
    rank: int | None = None,
) -> list[dict[str, dict[str, Any]]] | None:
    """
    Retrieves metric states across all ranks.

    Args:
        states: local metric state dict
        devices: mapping from metric name to device on which metric resides
        metrics_traversal_order: list of tuple (metric_name, state_name) defining the order of traversal through the metric state dicts
        rank: The rank to sync and compute metric values on. If None, all ranks will be synced on

    Returns:
        Metric state dict data from all ranks.
    """
    gathered_states = [
        _get_empty_metric_state_collection(
            metrics_traversal_order=metrics_traversal_order
        )
        for _ in range(dist.get_world_size())
    ]

    for metric_name, state_name in metrics_traversal_order:
        my_state_data = states[metric_name][
            state_name
        ]  # tensor, list, dict, int, or float
        if isinstance(my_state_data, torch.Tensor):
            _sync_tensor_states(
                metric_name,
                state_name,
                my_state_data,
                gathered_states,
                process_group=process_group,
                rank=rank,
            )
        elif isinstance(my_state_data, list):
            _sync_list_tensor_states(
                metric_name,
                state_name,
                my_state_data,
                devices[metric_name],
                gathered_states,
                process_group=process_group,
                rank=rank,
            )
        elif isinstance(my_state_data, dict):
            _sync_dict_tensor_states(
                metric_name,
                state_name,
                my_state_data,
                devices[metric_name],
                gathered_states,
                process_group=process_group,
                rank=rank,
            )
        elif isinstance(
            my_state_data,
            (
                int,
                float,
            ),
        ):
            _sync_obj_states(
                metric_name,
                state_name,
                my_state_data,
                gathered_states,
                process_group=process_group,
                rank=rank,
            )
        else:
            raise RuntimeError(
                f"Do not know how to sync state of type: {type(my_state_data)} for state {metric_name} {state_name}"
            )

    if rank is None or dist.get_rank(group=process_group) == rank:
        return gathered_states
    return None
