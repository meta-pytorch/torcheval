# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# pyre-ignore-all-errors[5]: Undefined variable type

import torch
from torch.utils.data.dataset import TensorDataset
from torcheval.metrics import MulticlassAccuracy

NUM_EPOCHS = 4
NUM_BATCHES = 16
BATCH_SIZE = 8


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


def prepare_dataloader() -> torch.utils.data.DataLoader:
    num_samples = NUM_BATCHES * BATCH_SIZE
    data = torch.randn(num_samples, 128)
    labels = torch.randint(low=0, high=2, size=(num_samples,))
    return torch.utils.data.DataLoader(
        TensorDataset(data, labels), batch_size=BATCH_SIZE
    )


def main() -> None:
    torch.random.manual_seed(42)

    model = Model()
    optim = torch.optim.Adagrad(model.parameters(), lr=0.001)

    train_dataloader = prepare_dataloader()

    loss_fn = torch.nn.CrossEntropyLoss()
    metric = MulticlassAccuracy()

    compute_frequency = 4
    num_epochs_completed = 0

    while num_epochs_completed < NUM_EPOCHS:
        data_iter = iter(train_dataloader)
        batch_idx = 0
        while True:
            try:
                # get the next batch from data iterator
                input, target = next(data_iter)
                output = model(input)

                # metric.update() updates the metric state with new data
                metric.update(output, target)

                loss = loss_fn(output, target)
                optim.zero_grad()
                loss.backward()
                optim.step()

                if (batch_idx + 1) % compute_frequency == 0:
                    print(
                        "Epoch {}/{}, Batch {}/{} --- loss: {:.4f}, acc: {:.4f}".format(
                            num_epochs_completed + 1,
                            NUM_EPOCHS,
                            batch_idx + 1,
                            NUM_BATCHES,
                            loss.item(),
                            # metric.compute() returns metric value from all seen data
                            metric.compute(),
                        )
                    )
                batch_idx += 1
            except StopIteration:
                break

        # metric.reset() cleans up all seen data
        metric.reset()

        num_epochs_completed += 1


if __name__ == "__main__":
    main()  # pragma: no cover
