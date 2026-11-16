# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch


def get_rand_data_binary(
    num_updates: int,
    num_tasks: int,
    batch_size: int,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generates a random binary dataset. For the returned tensors, shape[0] will correspond to the update, shape[1] will correspond to the task, and shape[2] will correspond to the sample.

    Notes:
        - If num_tasks is 1 the task dimension will be omitted; tensors will have shape (num_updates, batch_size) or (batch_size, ) depending on whether num_updates=1.
        - If num_updates is 1, the update dimension will be omitted; tensors will have shape (num_tasks, batch_size) or (batch_size, ) depending on whether num_tasks=1.
        - If both num_updates and num_tasks are not 1, the returned tensors will have shape (num_updates, num_tasks, batch_size).

    Args:
        num_updates: the number of calls to update on each rank.
        num_tasks: the number of tasks for the metric.
        batch_size: batch size of the dataset.
        device: device for the returned Tensors

    Returns:
        torch.Tensor: random feature data
        torch.Tensor: random targets
    """
    if device is None:
        device = torch.device("cpu")

    shape = [num_updates, num_tasks, batch_size]
    if num_tasks == 1 and num_updates == 1:
        shape = [batch_size]
    elif num_updates == 1:
        shape = [num_tasks, batch_size]
    elif num_tasks == 1:
        shape = [num_updates, batch_size]

    input = torch.rand(size=shape)
    targets = torch.randint(low=0, high=2, size=shape)
    return input.to(device), targets.to(device)


def get_rand_data_multiclass(
    num_updates: int,
    num_classes: int,
    batch_size: int,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generates a random multiclass dataset.

    Notes:
        - If num_updates is 1, the update dimension will be omitted; input tensors will have shape (batch_size, num_classes) and target tensor will have shape (batch_size, ).

    Args:
        num_updates: the number of calls to update on each rank.
        num_classes: the number of classes for the dataset.
        batch_size: batch size of the dataset.
        device: device for the returned Tensors

    Returns:
        torch.Tensor: random feature data
        torch.Tensor: random targets
    """
    if device is None:
        device = torch.device("cpu")

    input_shape = [num_updates, batch_size, num_classes]
    targets_shape = [num_updates, batch_size]
    if num_updates == 1:
        input_shape = [batch_size, num_classes]
        targets_shape = [batch_size]

    input = torch.rand(size=input_shape)
    targets = torch.randint(low=0, high=num_classes, size=targets_shape)
    return input.to(device), targets.to(device)


def get_rand_data_multilabel(
    num_updates: int,
    num_labels: int,
    batch_size: int,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generates a random multilabel dataset.

    Notes:
        - If num_updates is 1, the update dimension will be omitted; input tensors will have shape (batch_size, num_labels) and target tensor will have shape (batch_size, ).

    Args:
        num_updates: the number of calls to update on each rank.
        num_labels: the number of labels for the dataset.
        batch_size: batch size of the dataset.

    Returns:
        torch.Tensor: random feature data
        torch.Tensor: random targets
    """
    if device is None:
        device = torch.device("cpu")

    input_shape = [num_updates, batch_size, num_labels]
    targets_shape = [num_updates, batch_size, num_labels]
    if num_updates == 1:
        input_shape = [batch_size, num_labels]
        targets_shape = [batch_size, num_labels]

    input = torch.rand(size=input_shape)
    targets = torch.randint(low=0, high=2, size=targets_shape)
    return input.to(device), targets.to(device)


def get_rand_data_binned_binary(
    num_updates: int,
    num_tasks: int,
    batch_size: int,
    num_bins: int,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Get random binary dataset, along with a threshold for binned data.

    Notes:
        - If num_tasks is 1 the task dimension will be omitted; input and target tensors will have shape (num_updates, batch_size) or (batch_size, ) depending on whether num_updates=1.
        - If num_updates is 1, the update dimension will be omitted; input and target tensors will have shape (num_tasks, batch_size) or (batch_size, ) depending on whether num_tasks=1.
        - If both num_updates and num_tasks are not 1, the returned input and target tensors will have shape (num_updates, num_tasks, batch_size).
        - thresholds tensor always has shape (num_bins, ).

    Args:
        num_updates: the number of calls to update on each rank.
        num_tasks: the number of tasks for the metric.
        batch_size: batch size of the dataset.
        num_bins: The number of bins.
        device: device of the returned Tensors

    Returns:
        torch.Tensor: random feature data
        torch.Tensor: random targets
        torch.Tensor: thresholds
    """
    if device is None:
        device = torch.device("cpu")

    input, target = get_rand_data_binary(
        num_updates, num_tasks, batch_size, device=device
    )

    threshold = torch.cat([torch.tensor([0, 1]), torch.rand(num_bins - 2)])
    threshold, _ = torch.sort(threshold)
    threshold = torch.unique(threshold)
    return input, target, threshold.to(device)


def get_rand_data_wasserstein1d(
    num_updates: int,
    batch_size: int,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generates a random distribution dataset.

    Notes:
        - If num_updates is 1, the update dimension will be omitted; tensors will have shape (batch_size,).

    Args:
        num_updates: the number of calls to update on each rank.
        batch_size: batch size of the dataset.
        device: device for the returned Tensors

    Returns:
        torch.Tensor: distribution values first distribution
        torch.Tensor: distribution values second distribution
        torch.Tensor: weight values first distribution
        torch.Tensor: weight values second distribution
    """
    if device is None:
        device = torch.device("cpu")

    shape = [num_updates, batch_size]
    if num_updates == 1:
        shape = [batch_size]

    x = torch.rand(size=shape)
    y = torch.rand(size=shape)
    x_weights = torch.randint(low=1, high=10, size=shape)
    y_weights = torch.randint(low=1, high=10, size=shape)

    return x.to(device), y.to(device), x_weights.to(device), y_weights.to(device)
