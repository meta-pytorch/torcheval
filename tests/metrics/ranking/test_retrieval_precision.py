# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import torch
from torcheval.metrics.ranking import RetrievalPrecision
from torcheval.utils.test_utils.metric_class_tester import MetricClassTester


class TestRetrievalPrecision(MetricClassTester):
    # test update and compute functions

    def test_retrieval_precision_single_updates_1_query_avg_none(self) -> None:
        input = torch.tensor([0.2, 0.3, 0.5, 0.1, 0.3, 0.5, 0.2])
        target = torch.tensor([0, 0, 1, 1, 1, 0, 1])

        metric = RetrievalPrecision(k=2)
        metric.update(input, target)
        actual_result = metric.compute()
        expected_result = torch.tensor([0.5])
        torch.testing.assert_close(actual_result, expected_result)

        # reset metric and compute a new value with a new target
        metric.reset()

        target[2] = 0
        metric.update(input, target)
        actual_result = metric.compute()
        expected_result = torch.tensor([0.0])
        torch.testing.assert_close(actual_result, expected_result)

    def test_retrieval_precision_single_updates_1_query_avg_macro(self) -> None:
        input = torch.tensor([0.2, 0.3, 0.5, 0.1, 0.3, 0.5, 0.2])
        target = torch.tensor([0, 0, 1, 1, 1, 0, 1])

        metric = RetrievalPrecision(k=2, avg="macro")
        metric.update(input, target)
        actual_result = metric.compute()
        expected_result = torch.tensor(0.5)
        torch.testing.assert_close(actual_result, expected_result)

        # reset metric and compute a new value with a new target
        metric.reset()

        target[2] = 0
        metric.update(input, target)
        actual_result = metric.compute()
        expected_result = torch.tensor(0.0)
        torch.testing.assert_close(actual_result, expected_result)

    def test_retrieval_precision_single_updates_1_queries_avg_none_other_example(
        self,
    ) -> None:
        metric = RetrievalPrecision(k=3)
        metric.update(
            torch.tensor([0.1, 0.2, 0.3, 0.3, 0.4]), torch.tensor([1, 0, 0, 1, 1])
        )
        metric.update(
            torch.tensor([0.9, 0.9, 0.3, 0.1, 0.1]), torch.tensor([1, 1, 0, 0, 0])
        )

        expected_result = torch.tensor([1.0])
        actual_result = metric.compute()
        torch.testing.assert_close(actual_result, expected_result)

    def test_retrieval_precision_single_updates_n_queries_avg_none(self) -> None:
        input = torch.tensor([0.2, 0.3, 0.5, 0.1, 0.3, 0.5, 0.2])
        target = torch.tensor([0, 0, 1, 1, 1, 0, 1])

        metric = RetrievalPrecision(k=2, num_queries=4)
        indexes = torch.tensor([0, 0, 0, 1, 1, 1, 1])
        metric.update(input, target, indexes)
        actual_result = metric.compute()
        expected_result = torch.tensor([1 / 2, 1 / 2, torch.nan, torch.nan])
        torch.testing.assert_close(actual_result, expected_result, equal_nan=True)

        input2 = torch.tensor([0.4, 0.1, 0.6, 0.8, 0.7, 0.9, 0.3])
        target2 = torch.tensor([1, 0, 1, 0, 1, 1, 0])
        indexes = torch.tensor([2, 2, 2, 3, 3, 3, 3])
        metric.update(input2, target2, indexes)
        actual_result = metric.compute()
        expected_result = torch.tensor([1 / 2, 1 / 2, 1.0, 1 / 2])
        torch.testing.assert_close(actual_result, expected_result)

    def test_retrieval_precision_single_updates_n_queries_avg_macro(self) -> None:
        input = torch.tensor([0.2, 0.3, 0.5, 0.1, 0.3, 0.5, 0.2])
        target = torch.tensor([0, 0, 1, 1, 1, 0, 1])
        indexes = torch.tensor([0, 0, 0, 1, 1, 1, 1])

        metric = RetrievalPrecision(k=2, num_queries=4, avg="macro")
        metric.update(input, target, indexes)
        actual_result = metric.compute()
        expected_result = torch.tensor(0.5)
        torch.testing.assert_close(actual_result, expected_result)

        input2 = torch.tensor([0.4, 0.1, 0.6, 0.8, 0.7, 0.9, 0.3])
        target2 = torch.tensor([1, 0, 1, 0, 1, 1, 0])
        indexes = torch.tensor([2, 2, 2, 3, 3, 3, 3])
        # add value (1 + 1 / 2) / 2 = 0.750 to scores
        metric.update(input2, target2, indexes)
        actual_result = metric.compute()
        expected_result = torch.tensor((0.5 + 0.750) / 2)
        torch.testing.assert_close(actual_result, expected_result)

    def test_retrieval_precision_single_updates_n_queries_avg_none_docstring_example(
        self,
    ) -> None:
        input = torch.tensor([0.2, 0.3, 0.5, 0.1, 0.3, 0.5, 0.2])
        target = torch.tensor([0, 0, 1, 1, 1, 0, 1])
        indexes = torch.tensor([0, 0, 0, 1, 1, 1, 1])
        metric = RetrievalPrecision(k=2, num_queries=2)

        metric.update(input, target, indexes)
        actual_result = metric.compute()
        expected_result = torch.tensor([1 / 2, 1 / 2])
        torch.testing.assert_close(actual_result, expected_result)

        target2 = torch.tensor([1, 0, 1, 0, 1, 1, 0])
        input2 = torch.tensor([0.4, 0.1, 0.6, 0.8, 0.7, 0.9, 0.3])
        metric.update(input2, target2, indexes)
        actual_result = metric.compute()
        expected_result = torch.tensor([1.0, 1 / 2])
        torch.testing.assert_close(actual_result, expected_result)

    def test_retrieval_precision_multiple_updates_1_query(self) -> None:
        k = 2
        num_updates = 4
        input = torch.tensor(
            [
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
                [0.1, 0.2, 0.1, 0.3, 0.1, 0.2],
                [0.1, 0.2, 0.3, 0.3, 0.2, 0.1],
            ]
        )
        target = torch.tensor(
            [
                [0, 0, 0, 0, 1, 1],
                [0, 1, 1, 0, 1, 1],
                [0, 0, 0, 1, 1, 0],
                [0, 0, 0, 0, 1, 0],
            ]
        )

        expected_result = torch.tensor(1 / 2)

        self.run_class_implementation_tests(
            metric=RetrievalPrecision(k=k, avg="macro"),
            state_names={"topk", "target"},
            update_kwargs={"input": input, "target": target},
            compute_result=expected_result,
            num_total_updates=num_updates,
            num_processes=2,
        )

    def test_retrieval_precision_multiple_updates_n_queries_without_nan(
        self,
    ) -> None:
        num_updates = 4
        input = torch.tensor(
            [
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
                [0.1, 0.2, 0.1, 0.2, 0.1, 0.2],
                [0.1, 0.2, 0.4, 0.3, 0.2, 0.1],
            ]
        )
        target = torch.tensor(
            [
                [1, 1, 1, 1, 1, 0],
                [0, 1, 1, 0, 1, 1],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 1, 0],
            ]
        )
        indexes = torch.tensor(
            [
                [0, 1, 2, 3, 4, 5],
                [0, 0, 0, 1, 1, 1],
                [2, 2, 3, 3, 3, 3],
                [4, 4, 4, 4, 4, 5],
            ]
        )
        k = 2
        for empty_target_action in ["pos", "neg", "skip"]:
            for avg in ["macro", None]:
                value_query_5 = -1
                if empty_target_action == "neg":
                    value_query_5 = 0
                elif empty_target_action == "pos":
                    value_query_5 = 1
                elif empty_target_action == "skip":
                    value_query_5 = torch.nan

                expected_result = torch.tensor(
                    [
                        1 / 2,  # query 0
                        1 / 2,  # query 1
                        1 / 2,  # query 2
                        1 / 2,  # query 3
                        1,  # query 4
                        value_query_5,  # query 5
                    ]
                )
                if avg == "macro":
                    expected_result = expected_result.nanmean()
                self.run_class_implementation_tests(
                    metric=RetrievalPrecision(
                        k=k,
                        limit_k_to_size=True,
                        num_queries=6,
                        # pyre-ignore[6]: In call `torcheval.metrics.ranking.retrieval_precision.RetrievalPrecision.__init__`, for argument `empty_target_action`, expected `Union[typing_extensions.Literal['err'], typing_extensions.Literal['neg'], typing_extensions.Literal['pos']]` but got `str`.
                        empty_target_action=empty_target_action,
                        # pyre-ignore[6]: In call `torcheval.metrics.ranking.retrieval_precision.RetrievalPrecision.__init__`, for argument `avg`, expected `Union[typing_extensions.Literal['macro'], typing_extensions.Literal['none'], None]` but got `Optional[str]`.
                        avg=avg,
                    ),
                    state_names={"topk", "target"},
                    update_kwargs={
                        "input": input,
                        "target": target,
                        "indexes": indexes,
                    },
                    compute_result=expected_result,
                    num_total_updates=num_updates,
                    num_processes=2,
                )

    # TODO: investigate why this test fails
    def test_retrieval_precision_multiple_updates_n_queries_with_nan(
        self,
    ) -> None:
        num_updates = 4
        input = torch.tensor(
            [
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.1],
                [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
                [0.1, 0.2, 0.1, 0.2, 0.1, 0.2],
                [0.1, 0.2, 0.4, 0.3, 0.2, 0.1],
            ]
        )
        target = torch.tensor(
            [
                [1, 1, 1, 1, 1, 0],
                [0, 1, 1, 0, 1, 1],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 1, 0],
            ]
        )
        indexes = torch.tensor(
            [
                [0, 1, 2, 3, 4, 4],
                [0, 0, 0, 1, 1, 1],
                [2, 2, 3, 3, 3, 3],
                [4, 4, 4, 4, 4, 4],
            ]
        )
        k = 2
        for empty_target_action in ["pos", "neg", "skip"]:
            for avg in ["macro", None]:
                value_query_5 = torch.nan

                expected_result = torch.tensor(
                    [
                        1 / 2,  # query 0
                        1 / 2,  # query 1
                        1 / 2,  # query 2
                        1 / 2,  # query 3
                        1,  # query 4
                        value_query_5,  # query 5
                    ]
                )
                if avg == "macro":
                    expected_result = expected_result.nanmean()
                self.run_class_implementation_tests(
                    metric=RetrievalPrecision(
                        k=k,
                        limit_k_to_size=True,
                        num_queries=6,
                        # pyre-ignore[6]: In call `torcheval.metrics.ranking.retrieval_precision.RetrievalPrecision.__init__`, for argument `empty_target_action`, expected `Union[typing_extensions.Literal['err'], typing_extensions.Literal['neg'], typing_extensions.Literal['pos']]` but got `str`.
                        empty_target_action=empty_target_action,
                        # pyre-ignore[6]: In call `torcheval.metrics.ranking.retrieval_precision.RetrievalPrecision.__init__`, for argument `avg`, expected `Union[typing_extensions.Literal['macro'], typing_extensions.Literal['none'], None]` but got `Optional[str]`.
                        avg=avg,
                    ),
                    state_names={"topk", "target"},
                    update_kwargs={
                        "input": input,
                        "target": target,
                        "indexes": indexes,
                    },
                    compute_result=expected_result,
                    num_total_updates=num_updates,
                    num_processes=2,
                )

    # test empty_target_action argument

    def test_retrieval_precision_empty_target_action(self) -> None:
        input = torch.tensor([0.2, 0.3])
        target = torch.tensor([0, 0])

        metric = RetrievalPrecision(k=2, empty_target_action="neg")
        metric.update(input, target)
        actual_result = metric.compute()
        expected_result = torch.tensor([0.0])
        torch.testing.assert_close(actual_result, expected_result)

        metric = RetrievalPrecision(k=2, empty_target_action="pos")
        metric.update(input, target)
        actual_result = metric.compute()
        expected_result = torch.tensor([1.0])
        torch.testing.assert_close(actual_result, expected_result)

        metric = RetrievalPrecision(k=2, empty_target_action="err")
        with self.assertRaisesRegex(
            ValueError, r"no positive value found in target=tensor\(\[0\.\, 0\.\]\)\."
        ):
            metric.update(input, target)
            metric.compute()

    def test_retrieval_precision_no_update(self) -> None:
        metric = RetrievalPrecision()
        expected_result = torch.tensor([torch.nan])
        actual_result = metric.compute()
        torch.testing.assert_close(actual_result, expected_result, equal_nan=True)

        metric = RetrievalPrecision(num_queries=2)
        expected_result = torch.tensor([torch.nan] * 2)
        actual_result = metric.compute()
        torch.testing.assert_close(actual_result, expected_result, equal_nan=True)

    # test input checks

    def test_retrieval_precision_with_invalid_input_in_update(self) -> None:
        metric = RetrievalPrecision()
        with self.assertRaisesRegex(
            ValueError,
            r"input and target must be of the same shape, got input\.shape=torch\.Size\(\[2\, 1\]\) and target\.shape=torch.Size\(\[2\]\)\.",
        ):
            target = torch.tensor([1.0, 0.0])
            input = torch.tensor([[0.0], [0.0]])
            metric.update(input, target)

        with self.assertRaisesRegex(
            ValueError,
            r"input and target should be one dimensional tensors, got input and target dimensions=2.",
        ):
            target = torch.tensor([[1.0], [0.0]])
            input = torch.tensor([[0.0], [0.0]])
            metric.update(input, target)

    def test_retrieval_precision_with_invalid_attributes(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            r"when limit_k_to_size is True, k must be a positive \(>0\) integer\.",
        ):
            RetrievalPrecision(k=None, limit_k_to_size=True)
