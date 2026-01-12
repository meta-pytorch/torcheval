# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
from collections.abc import Iterable, MutableMapping
from copy import deepcopy
from typing import Any, overload, TypeVar

import torch
import torch.distributed as dist
from pyre_extensions import none_throws
from torcheval.metrics import Metric
from torcheval.metrics.metric import TComputeReturn, TState
from torcheval.metrics.synclib import metrics_traversal_order, sync_states

log: logging.Logger = logging.getLogger(__name__)

_TMetrics = TypeVar("_TMetrics", bound=Iterable[Metric])

_TMP: str = "tmp"


def _get_world_size(process_group: dist.ProcessGroup | None) -> int:
    if not dist.is_available() or not dist.is_initialized():
        # dist is not initialized or available, return 1 for world size
        return 1
    return dist.get_world_size(group=process_group)


def _get_rank(process_group: dist.ProcessGroup | None) -> int:
    if not dist.is_available() or not dist.is_initialized():
        # dist is not initialized or available, return 0 for rank
        return 0
    return dist.get_rank(group=process_group)


def sync_and_compute(
    metric: Metric[TComputeReturn],
    process_group: dist.ProcessGroup | None = None,
) -> TComputeReturn:
    """
    Sync metric states and returns the ``metric.compute()`` result of
    synced metric on all ranks.

    Args:
        metric: The metric object to be synced and computed.
        process_group: The process group on which the metric states are
            gathered. default: ``None`` (the entire world)

    Examples::

        >>> # Assumes world_size of 3.
        >>> # Process group initialization omitted on each rank.
        >>> import torch
        >>> import torch.distributed as dist
        >>> from torcheval.metrics import Max
        >>> max = Max()
        >>> max.update(torch.tensor(dist.get_rank())).compute()
        tensor(0.) # Rank 0
        tensor(1.) # Rank 1
        tensor(2.) # Rank 2
        >>> sync_and_compute(max)
        tensor(2.) # Rank 0
        tensor(2.) # Rank 1
        tensor(2.) # Rank 2
    """
    synced_metric = get_synced_metric(metric, process_group)
    compute_result = synced_metric.compute()

    return compute_result


def sync_and_compute_collection(
    metrics: MutableMapping[str, Metric],
    process_group: dist.ProcessGroup | None = None,
) -> dict[str, Any]:
    """
    Sync metric states across a dict of metrics and returns the
    ``metric.compute()`` result of synced metrics on all ranks

    Args:
        metrics: The dict of metric objects to be synced and computed.
        process_group: The process group on which the metric states are
            gathered. default: ``None`` (the entire world)

    Examples::

        >>> # Assumes world_size of 3.
        >>> # Process group initialization omitted on each rank.
        >>> import torch
        >>> import torch.distributed as dist
        >>> from torcheval.metrics import Max, Min
        >>> metrics = {"max" : Max(), "min": Min()}
        >>> metrics["max"].update(torch.tensor(dist.get_rank())).compute()
        tensor(0.) # Rank 0
        tensor(1.) # Rank 1
        tensor(2.) # Rank 2
        >>> metrics["min"].update(torch.tensor(dist.get_rank())).compute()
        tensor(0.) # Rank 0
        tensor(1.) # Rank 1
        tensor(2.) # Rank 2
        >>> sync_and_compute_collection(metrics)
        {"max" : tensor(2.), "min": tensor(0.)} # Rank 0
        {"max" : tensor(2.), "min": tensor(0.)} # Rank 1
        {"max" : tensor(2.), "min": tensor(0.)} # Rank 2
    """
    synced_metrics = get_synced_metric_collection(metrics, process_group)
    compute_result = {key: m.compute() for key, m in synced_metrics.items()}

    return compute_result


def get_synced_state_dict(
    metric: Metric,
    process_group: dist.ProcessGroup | None = None,
) -> dict[str, Any]:
    """
    Return the state dict of a metric after syncing on all ranks.
    Return an empty dict on other ranks.

    Args:
        metric: The metric object to sync and get ``state_dict()``
        process_group: The process group on which the metric states are
            gathered. default: ``None`` (the entire world)
    Returns:
        state dict of synced metric

    Examples::

        >>> # Assumes world_size of 3.
        >>> # Process group initialization omitted on each rank.
        >>> import torch
        >>> import torch.distributed as dist
        >>> from torcheval import Max
        >>> max = Max()
        >>> max.update(torch.tensor(dist.get_rank()))
        >>> get_synced_state_dict(max)
        {"max", tensor(2.)} # Rank 0
        {"max", tensor(2.)} # Rank 1
        {"max", tensor(2.)} # Rank 2
    """
    synced_metric = get_synced_metric(metric, process_group)
    return synced_metric.state_dict() if synced_metric else {}


def get_synced_state_dict_collection(
    metric_collection: MutableMapping[str, Metric],
    process_group: dist.ProcessGroup | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Return the state dict of a collection of metrics after syncing on all ranks.
    Return an None on other ranks.

    Args:
        metric_collection (Dict[str, Metric]): The metric objects to sync and get ``state_dict()``
        process_group: The process group on which the metric states are
            gathered. default: ``None`` (the entire world)
    Returns:
        Bundle of state dicts of for the synced metrics

    Examples::

        >>> # Assumes world_size of 3.
        >>> # Process group initialization omitted on each rank.
        >>> import torch
        >>> import torch.distributed as dist
        >>> from torcheval import Max, Min
        >>> maximum = Max()
        >>> maximum.update(torch.tensor(dist.get_rank()))
        >>> minimum = Min()
        >>> minimum.update(torch.tensor(dist.get_rank()))
        >>> get_synced_state_dict({"max rank": maximum, "min rank": minimum})
        {"max rank": {"max", tensor(2.)}, "min rank": {"min", tensor(0.)}} # Rank 0
        {"max rank": {"max", tensor(2.)}, "min rank": {"min", tensor(0.)}} # Rank 1
        {"max rank": {"max", tensor(2.)}, "min rank": {"min", tensor(0.)}} # Rank 2
    """
    synced_metrics = get_synced_metric_collection(
        metric_collection,
        process_group,
    )

    return {key: metric.state_dict() for key, metric in synced_metrics.items()}


def clone_metric(metric: Metric) -> Metric:
    """
    Return a new metric instance which is cloned from the input metric.

    Args:
        metric: The metric object to clone
    Returns:
        A new metric instance from cloning
    """
    return deepcopy(metric)


def clone_metrics(metrics: _TMetrics) -> list[Metric]:
    """
    Return a list of new metric instances which are cloned from the input metrics.

    Args:
        metrics: The metric objects to clone
    Returns:
        A list of metric instances from cloning
    """
    return [clone_metric(metric) for metric in metrics]


def get_synced_metric(
    metric: Metric,
    process_group: dist.ProcessGroup | None = None,
) -> Metric:
    """
    Returns a metric object on all ranks whose internal state
    variables are synced across processes in the process_group.

    Args:
        metric: The metric object to sync.
        process_group: The process group on which the metric states are
            gathered. default: ``None`` (the entire world)

    Examples::

        >>> # Assumes world_size of 3.
        >>> # Process group initialization omitted on each rank.
        >>> import torch
        >>> import torch.distributed as dist
        >>> from torcheval import Max
        >>> max = Max()
        >>> max.update(torch.tensor(dist.get_rank())).compute()
        tensor(0.) # Rank 0
        tensor(1.) # Rank 1
        tensor(2.) # Rank 2
        >>> synced_metric = get_synced_metric(max)  # by default sync metric states to Rank 0
        >>> synced_metric.compute() if synced_metric else None
        tensor(2.)     # Rank 0
        None # Rank 1 -- synced_metric is None
        None # Rank 2 -- synced_metric is None
        >>> synced_metric = get_synced_metric(max)
        >>> synced_metric.compute()
        tensor(2.) # Rank 0
        tensor(2.) # Rank 1
        tensor(2.) # Rank 2
    """
    world_size = _get_world_size(process_group)
    _validate_rank_and_world_size(world_size)

    if world_size == 1:
        return metric

    gathered_metric_list = _sync_metric_object(
        metric,
        # pyre-fixme[6]: For 2nd param expected `ProcessGroup` but got `Union[None,
        #  dist.ProcessGroup, _distributed_c10d.ProcessGroup]`.
        process_group if process_group else dist.group.WORLD,
        world_size,
    )

    local_rank = _get_rank(process_group)
    other_rank_metrics: list[Metric] = [
        gathered_metric_list[rank] for rank in range(world_size) if rank != local_rank
    ]

    return clone_metric(metric).to(metric.device).merge_state(other_rank_metrics)


def get_synced_metric_collection(
    metric_collection: MutableMapping[str, Metric],
    process_group: dist.ProcessGroup | None = None,
) -> dict[str, Metric] | MutableMapping[str, Metric]:
    """
    Returns a dict of metric objects to all ranks whose
    internal state variables are synced across processes in the process_group.

    The data transfer is batched to maximize efficiency.

    Args:
        metric_collection (Dict[str, Metric]): The dict of metric objects to sync.
        process_group (int): The process group on which the metric states are
            gathered. default: ``None`` (the entire world)

    Examples::

        >>> # Assumes world_size of 3.
        >>> # Process group initialization omitted on each rank.
        >>> import torch
        >>> import torch.distributed as dist
        >>> from torcheval.metrics import Max, Min
        >>> metrics = {"max" : Max(), "min": Min()}
        >>> metrics["max"].update(torch.tensor(dist.get_rank()))
        >>> metrics["min"].update(torch.tensor(dist.get_rank()))
        >>> synced_metrics = get_synced_metric_collection(metrics)
        >>> synced_metrics["max"].compute()
        tensor(2.) # Rank 0
        tensor(2.) # Rank 1
        tensor(2.) # Rank 2
        >>> synced_metrics["min"].compute()
        tensor(0.) # Rank 0
        tensor(0.) # Rank 1
        tensor(0.) # Rank 2
    """
    world_size = _get_world_size(process_group)
    _validate_rank_and_world_size(world_size)

    if world_size == 1:
        return metric_collection

    list_of_metric_collections = _sync_metric_object(
        metric_collection,
        # pyre-fixme[6]: For 2nd param expected `ProcessGroup` but got `Union[None,
        #  dist.ProcessGroup, _distributed_c10d.ProcessGroup]`.
        process_group if process_group else dist.group.WORLD,
        world_size,
    )

    if isinstance(list_of_metric_collections[0], MutableMapping):
        local_rank = dist.get_rank(process_group)

        # metric bundles are dicts.
        synced_metric_dict: dict[str, Metric] = {}

        for metric_key in metric_collection.keys():
            base_metric = metric_collection[metric_key]
            base_metric = clone_metric(base_metric).to(base_metric.device)
            other_rank_metrics: list[Metric] = [
                list_of_metric_collections[rank][metric_key]
                for rank in range(world_size)
                if rank != local_rank
            ]
            synced_metric_dict[metric_key] = base_metric.merge_state(other_rank_metrics)
        return synced_metric_dict


def _validate_rank_and_world_size(
    world_size: int,
) -> None:
    if world_size == 1:
        log.warning(
            "World size is 1, and metric(s) not synced. returning the input metric(s)."
        )
    elif world_size == -1:
        raise RuntimeError("The current process is not part of the process group")
    if world_size < 1:
        raise RuntimeError(
            f"Unexpected world_size {world_size} is seen when syncing metrics!"
        )


@overload
def _sync_metric_object(
    local_metric_data: Metric,
    process_group: dist.ProcessGroup,
    world_size: int,
) -> list[Metric]: ...


@overload
def _sync_metric_object(
    local_metric_data: MutableMapping[str, Metric],
    process_group: dist.ProcessGroup,
    world_size: int,
) -> list[MutableMapping[str, Metric]]: ...


def _apply_device_to_tensor_states(
    state_dict: dict[str, Any], device: torch.device
) -> None:
    for state_name, state_value in state_dict.items():
        if isinstance(state_value, torch.Tensor):
            state_dict[state_name] = state_value.to(device)
        elif isinstance(state_value, list):
            state_dict[state_name] = [tensor.to(device) for tensor in state_value]
        elif isinstance(state_value, dict):
            state_dict[state_name] = {
                key: tensor.to(device) for key, tensor in state_value.items()
            }


def _sync_metric_object(
    local_metric_data: Metric | MutableMapping[str, Metric],
    process_group: dist.ProcessGroup,
    world_size: int,
) -> list[Metric] | list[MutableMapping[str, Metric]]:
    unpack = False  # unpack the dictionary into a single metric when returned. Only used when metric_data is a metric and not a dict of metrics.
    if isinstance(local_metric_data, Metric):
        local_metric_data = {_TMP: local_metric_data}
        unpack = True

    # Allow metrics to run some pre-processing before syncing states
    # for example, a common optimization is to concat tensors in a list into a single tensor
    for m in local_metric_data.values():
        m._prepare_for_merge_state()

    # create a dict of state dicts for each metric in the collection, i.e. extract the state dicts from the Metric objects
    metric_state_data: dict[str, dict[str, TState]] = {}
    metric_to_device: dict[str, torch.device] = {}
    backend = dist.get_backend(group=process_group)
    for metric_name, metric in local_metric_data.items():
        metric_state_data[metric_name] = metric.state_dict()
        if backend == "nccl" and metric.device.type == "cpu":
            log.warning(
                "Metric tensor states are on CPU, but NCCL process group detected. "
                "These tensors will be moved to GPU prior to syncing. "
                "It is recommended for efficiency reasons to either use a gloo process group to sync metrics on CPU, or move the metrics to GPU and use the NCCL process group."
            )
            # move tensors to gpu since on nccl
            _apply_device_to_tensor_states(
                metric_state_data[metric_name],
                # pyre-ignore: Incompatible parameter type [6]
                torch.cuda.current_device(),
            )
            # pyre-ignore: Incompatible parameter type [6]
            metric_to_device[metric_name] = torch.cuda.current_device()
        else:
            metric_to_device[metric_name] = metric.device

    metric_state_traversal_order = metrics_traversal_order(metric_state_data)

    world_metric_data = sync_states(
        metric_state_data,
        metric_to_device,
        metric_state_traversal_order,
        process_group=process_group,
    )
    world_metric_data = none_throws(world_metric_data)

    # Repack states into Metrics or Dict[str, Metric]s
    if unpack:
        # if users passed in one metric, read it from the "tmp" key and return a list of metrics.
        gathered_data_list = []
        for rank_data in world_metric_data:
            gathered_data_list.append(_convert_to_psuedo_metric(rank_data[_TMP]))
    else:
        # if users passed in a dict[str, metric], return a list of dict[str, metric] populated with the gathered state dicts.
        gathered_data_list = []
        for rank_data in world_metric_data:
            rank_dict = {}
            for metric_name in local_metric_data.keys():
                rank_dict[metric_name] = _convert_to_psuedo_metric(
                    rank_data[metric_name]
                )
            gathered_data_list.append(rank_dict)
    return gathered_data_list


def _convert_to_psuedo_metric(metric_state_dict: dict[str, Any]) -> Any:
    """
    Converts dictionary to object with attributes set according to key-value.
    """
    return type("", (), metric_state_dict)


def reset_metrics(metrics: _TMetrics) -> _TMetrics:
    """
    Reset input metrics and returns the reset collection back to users.

    Args:
        metrics: The metrics to be reset

    Examples::

        >>> from torcheval.metrics import Max, Min
        >>> max = Max()
        >>> min = Min()
        >>> max.update(torch.tensor(1)).compute()
        >>> min.update(torch.tensor(2)).compute()
        >>> max, min = reset_metrics((max, min))
        >>> max.compute()
        tensor(0.)
        >>> min.compute()
        tensor(0.)
    """

    for metric in metrics:
        metric.reset()
    return metrics


def to_device(
    metrics: _TMetrics, device: torch.device, *args: Any, **kwargs: Any
) -> _TMetrics:
    """
    Moves input metrics to the target device and returns the moved metrics back to users.

    Args:
        metrics: The metrics to be moved to the device
        device: the device to move te metrics to
        *args: Variadic arguments forwarded to ``Metric.to``
        **kwargs: Named arguments forwarded to ``Metric.to``

    Examples::

        >>> from torcheval.metrics import Max, Min
        >>> max = Max()
        >>> min = Min()
        >>> max, min = to_device((max, min), torch.device("cuda"))
        >>> max.device
        torch.device("cuda")
        >>> min.device
        torch.device("cuda")
    """
    for metric in metrics:
        metric.to(device, *args, **kwargs)
    return metrics


def classwise_converter(
    input: torch.Tensor, name: str, labels: list[str] | None = None
) -> dict[str, torch.Tensor]:
    """
    Converts an unaveraged metric result tensor into a dictionary with
    each key being 'metricname_classlabel' and value being the data
    associated with that class.

    Args:
        input (torch.Tensor): The tensor to be split along its first dimension.
        name (str): Name of the metric.
        labels (List[str], Optional): Optional list of strings indicating the different classes.

    Raises:
        ValueError: When the length of `labels` is not equal to the number of classes.
    """
    if labels is None:
        return {f"{name}_{i}": val for i, val in enumerate(input)}

    if input.size(dim=0) != len(labels):
        raise ValueError(
            f"Number of labels {len(labels)} must be equal to the number of classes {input.size(dim=0)}!"
        )
    return {f"{name}_{label}": val for label, val in zip(labels, input)}
