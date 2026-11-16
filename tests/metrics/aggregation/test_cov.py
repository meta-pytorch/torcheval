# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


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
