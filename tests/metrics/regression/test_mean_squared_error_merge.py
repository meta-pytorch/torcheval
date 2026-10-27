# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from itertools import product

import torch
from torcheval.metrics import MeanSquaredError


class TestMeanSquaredErrorMerge(unittest.TestCase):
    def test_merge_preserves_donors(self) -> None:
        for multioutput, weighted, generator, outputs in product(
            ("raw_values", "uniform_average"), (False, True), (False, True), (1, 2)
        ):
            with self.subTest(
                multioutput=multioutput,
                weighted=weighted,
                generator=generator,
                outputs=outputs,
            ):
                input = torch.tensor([[1.0, 2.0], [3.0, 4.0], [2.0, 5.0]])
                target = torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.0, 2.0]])
                if outputs == 1:
                    input, target = input[:, 0], target[:, 0]
                weights = torch.tensor([1.0, 2.0, 3.0]) if weighted else None
                donors = [MeanSquaredError(multioutput=multioutput) for _ in range(2)]
                for donor, batch in zip(donors, (slice(0, 2), slice(2, 3))):
                    donor.update(
                        input[batch],
                        target[batch],
                        sample_weight=weights[batch] if weighted else None,
                    )
                states = [donor.state_dict() for donor in donors]
                scores = [donor.compute().clone() for donor in donors]
                recipient = MeanSquaredError(multioutput=multioutput)
                self.assertIs(
                    recipient.merge_state(iter(donors) if generator else donors),
                    recipient,
                )
                error = (input - target).square()
                if weighted:
                    error = error * (weights[:, None] if outputs == 2 else weights)
                expected = error.sum(dim=0) / (weights.sum() if weighted else 3)
                if multioutput == "uniform_average":
                    expected = expected.mean()
                torch.testing.assert_close(recipient.compute(), expected)
                for donor, state, score in zip(donors, states, scores):
                    for name, value in state.items():
                        torch.testing.assert_close(donor.state_dict()[name], value)
                    torch.testing.assert_close(donor.compute(), score)

    def test_updates_after_merge_are_independent(self) -> None:
        for multioutput, outputs in product(("raw_values", "uniform_average"), (1, 2)):
            with self.subTest(multioutput=multioutput, outputs=outputs):
                input = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
                if outputs == 1:
                    input = input[:, 0]
                target = torch.zeros_like(input)
                donor = MeanSquaredError(multioutput=multioutput).update(input, target)
                donor_state = donor.state_dict()
                recipient = MeanSquaredError(multioutput=multioutput)
                recipient.merge_state([donor]).update(input, target)
                for name, value in donor_state.items():
                    torch.testing.assert_close(donor.state_dict()[name], value)
                expected = input.square().mean(dim=0).to(donor.compute().dtype)
                if multioutput == "uniform_average":
                    expected = expected.mean()
                torch.testing.assert_close(recipient.compute(), expected)
                recipient_state = recipient.state_dict()
                donor.update(input * 2, target)
                for name, value in recipient_state.items():
                    torch.testing.assert_close(recipient.state_dict()[name], value)
                torch.testing.assert_close(donor.compute(), expected * 2.5)

    def test_reset_and_checkpoint_after_merge(self) -> None:
        input = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        target = torch.zeros_like(input)
        donor = MeanSquaredError(multioutput="raw_values").update(input, target)
        recipient = MeanSquaredError(multioutput="raw_values").merge_state([donor])
        restored = MeanSquaredError(multioutput="raw_values")
        restored.load_state_dict(recipient.state_dict())
        restored.update(input, target)
        torch.testing.assert_close(restored.compute(), input.square().mean(dim=0))
        self.assertIs(recipient.reset(), recipient)
        recipient.update(input * 2, target)
        torch.testing.assert_close(recipient.compute(), input.square().mean(dim=0) * 4)
        torch.testing.assert_close(donor.compute(), input.square().mean(dim=0))
