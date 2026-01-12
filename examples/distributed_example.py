# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Torcheval Distributed Example Overview
============================
This tutorial will demonstrate how to use Torcheval metrics in distributed computing environment.

Take metric ``MulticlassAccuracy`` as an example.
First, let's import all necessary imports.

"""

import os
import time
import uuid

import torch
from torch import distributed as dist
from torch.distributed import launcher as pet
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.dataset import TensorDataset
from torcheval.metrics import MulticlassAccuracy, Throughput
from torcheval.metrics.toolkit import sync_and_compute

#####################################################################
# Then prepare the global variables. For this tutorial, we plan to use 4 processes (`NUM_PROCESSES` = 4),
# but you can change it based on your computing resource setup.

NUM_PROCESSES = 4
NUM_EPOCHS = 4
NUM_BATCHES = 16
BATCH_SIZE = 8
COMPUTE_FREQUENCY = 4

#####################################################################
# Next, prepare a toy model and dataloader.


class Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 2),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.layers(X)


def prepare_dataloader(device: torch.device) -> torch.utils.data.DataLoader:
    num_samples = NUM_BATCHES * BATCH_SIZE
    data = torch.randn(num_samples, 128, device=device)
    labels = torch.randint(low=0, high=2, size=(num_samples,), device=device)
    return torch.utils.data.DataLoader(
        TensorDataset(data, labels), batch_size=BATCH_SIZE
    )


######################################################################
# Define the training step for each batch of input data.
# Before the training loop starts, let's initialize metric ``MulticlassAccuracy``
# and move it to the corresponding device.


def train() -> None:
    # Set directly in case hostname isn't identifiable and routable.
    os.environ["MASTER_ADDR"] = "localhost"
    if torch.cuda.is_available() and torch.cuda.device_count() >= NUM_PROCESSES:
        dist.init_process_group(backend="nccl")
    else:
        dist.init_process_group(backend="gloo")
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    if torch.cuda.is_available() and torch.cuda.device_count() >= NUM_PROCESSES:
        device = torch.device(f"cuda:{local_rank}")
        # Need specifically set current device
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    torch.manual_seed(42)

    print(f"Running basic DDP example on device {device}.")
    model = Model().to(device)
    ddp_model = DDP(model)
    optim = torch.optim.Adagrad(ddp_model.parameters(), lr=0.001)

    train_dataloader = prepare_dataloader(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    metric = MulticlassAccuracy(device=device)
    throughtput = Throughput(device=device)

    num_epochs_completed = 0
    while num_epochs_completed < NUM_EPOCHS:
        t0 = time.monotonic()
        data_iter = iter(train_dataloader)
        batch_idx = 0
        while True:
            try:
                # get the next batch from data iterator
                input, target = next(data_iter)
                output = ddp_model(input)

                loss = loss_fn(output, target)
                optim.zero_grad()
                loss.backward()
                optim.step()
                # metric.update() updates the metric state with new data
                metric.update(output, target)
                if (batch_idx + 1) % COMPUTE_FREQUENCY == 0:
                    # sync_and_compute() returns metric value from all seen data in all ranks
                    # It needs to be invoked on all ranks of the process_group
                    compute_result = sync_and_compute(metric)
                    if global_rank == 0:
                        print(
                            "Epoch {}/{}, Batch {}/{} --- loss: {:.4f}, acc: {:.4f}".format(
                                num_epochs_completed + 1,
                                NUM_EPOCHS,
                                batch_idx + 1,
                                NUM_BATCHES,
                                loss,
                                compute_result,
                            )
                        )
                batch_idx += 1
                t1 = time.monotonic()
                throughtput.update(batch_idx * BATCH_SIZE, t1 - t0)
            except StopIteration:
                break

        # metric.reset() cleans up all seen data
        metric.reset()

        num_epochs_completed += 1

    # throughput calcuation option 1: explicitly synchronize states across processes using `sync_and_compute()`
    global_throughput = sync_and_compute(throughtput)

    # throughput calcuation option 2: approximate the global throughput by scaling the local throughput value by the total number of processes
    local_throughput = throughtput.compute()
    approx_global_throughput = local_throughput * NUM_PROCESSES
    if global_rank == 0:
        print(
            f"Epoch{num_epochs_completed}/{NUM_EPOCHS} -- synced throughput:{global_throughput}"
        )
        print(
            f"Epoch{num_epochs_completed}/{NUM_EPOCHS} -- local throughput:{local_throughput}, approximate global throughput: {approx_global_throughput}"
        )
    dist.destroy_process_group()


def main() -> None:
    lc = pet.LaunchConfig(
        min_nodes=1,
        max_nodes=1,
        nproc_per_node=NUM_PROCESSES,
        run_id=str(uuid.uuid4()),
        rdzv_backend="c10d",
        rdzv_endpoint="localhost:0",
        max_restarts=0,
        monitor_interval=1,
    )

    pet.elastic_launch(lc, entrypoint=train)()


if __name__ == "__main__":
    main()  # pragma: no cover


######################################################################
# ``TorchEval`` provided two functions for compute: ``metric.compute()`` and ``sync_and_compute(metric)``.
# -  ``metric.compute()`` returns the compute result for the local process.
# -  ``sync_and_compute(metric)`` syncs the metric objects from all processes, and return the compute result based on all data. When the training script is running on a single process (``world_size = 1``), ``sync_and_compute(metric)`` returns exact same result as `metric.compute()`


# The Class version of metrics provides a function ` ``merge_state()``, which is used to update the current metric's state variables to be the merged states of the current metric and input metrics.
# This method can be used as a building block for syncing metric states in distributed training. For example, ``sync_and_compute`` in the metric toolkit will use this method to merge metric objects gathered from the process group.
