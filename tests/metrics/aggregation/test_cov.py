# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from copy import deepcopy

import torch
from torcheval.metrics import Covariance
from torcheval.utils.test_utils.metric_class_tester import MetricClassTester


class TestCovariance(MetricClassTester):
    def _test_covariance_with_input(self, batching: list[int]) -> None:
        gen = torch.Generator()
        gen.manual_seed(3)
        X = torch.randn(sum(batching), 4, generator=gen)
        self.run_class_implementation_tests(
            metric=Covariance(),
            state_names={"n", "sum", "ss_sum"},
            update_kwargs={"obs": torch.split(X, batching, dim=0)},
            compute_result=(X.mean(dim=0), torch.cov(X.T)),
            num_total_updates=len(batching),
            min_updates_before_compute=1,
            num_processes=4,
        )

    def test_covariance_all_at_once(self) -> None:
        self._test_covariance_with_input([100, 100, 100, 100])

    def test_covariance_one_by_one(self) -> None:
        self._test_covariance_with_input(list(range(2, 22)))

    def test_covariance_overflow(self) -> None:
        cov = Covariance()
        s = torch.zeros(10)
        ss_sum = torch.ones(10, 10)

        cov._update(s, ss_sum, torch.iinfo(torch.uint64).max)
        cov._update(s, ss_sum, 1)  # No overflow!

    def test_merge_state_preserves_independent_states(self) -> None:
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                batches = [
                    torch.tensor([[1, 2], [3, 4]], dtype=dtype),
                    torch.tensor([[5, 6], [7, 8]], dtype=dtype),
                    torch.tensor([[9, 10], [11, 12]], dtype=dtype),
                ]
                donors = [Covariance().update(batch) for batch in batches[:2]]
                states = [deepcopy(donor.state_dict()) for donor in donors]
                results = [donor.compute() for donor in donors]
                merged = Covariance().merge_state(donors)
                pooled = torch.cat(batches[:2])
                torch.testing.assert_close(
                    merged.compute(), (pooled.mean(0), torch.cov(pooled.T))
                )
                for donor, state, result in zip(donors, states, results):
                    torch.testing.assert_close(donor.state_dict(), state)
                    torch.testing.assert_close(donor.compute(), result)

                merged.update(batches[2])
                pooled = torch.cat(batches)
                torch.testing.assert_close(
                    merged.compute(), (pooled.mean(0), torch.cov(pooled.T))
                )
                for donor, state in zip(donors, states):
                    torch.testing.assert_close(donor.state_dict(), state)
                merged_state = deepcopy(merged.state_dict())
                donors[0].update(batches[2])
                torch.testing.assert_close(merged.state_dict(), merged_state)

    def test_merge_state_single_donor_then_update(self) -> None:
        batch = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        donor = Covariance().update(batch)
        state = deepcopy(donor.state_dict())
        merged = Covariance().merge_state([donor]).update(batch + 4)
        torch.testing.assert_close(donor.state_dict(), state)
        pooled = torch.cat([batch, batch + 4])
        torch.testing.assert_close(
            merged.compute(), (pooled.mean(0), torch.cov(pooled.T))
        )
