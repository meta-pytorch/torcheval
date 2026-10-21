# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import torch
from torcheval.metrics import RetrievalRecall
from torcheval.metrics.functional import retrieval_recall


class TestRetrievalRecall(unittest.TestCase):
    def test_batched_recall(self) -> None:
        scores = torch.tensor([0.9, 0.8, 0.7, 0.6])
        target = torch.tensor([1, 0, 1, 1])
        for k in (2, 8, None):
            for batch_size in (1, 3, 4):
                with self.subTest(k=k, batch_size=batch_size):
                    metric = RetrievalRecall(k=k)
                    for x, y in zip(scores.split(batch_size), target.split(batch_size)):
                        metric.update(x, y)
                    expected = retrieval_recall(scores, target, k=k).reshape(1)
                    torch.testing.assert_close(metric.compute(), expected)
                    restored = RetrievalRecall(k=k)
                    restored.load_state_dict(metric.state_dict())
                    restored.to(torch.device("cpu"))
                    torch.testing.assert_close(restored.compute(), expected)
                    restored.update(torch.tensor([0.1]), torch.tensor([1]))
                    full_scores = torch.cat([scores, torch.tensor([0.1])])
                    full_target = torch.cat([target, torch.tensor([1])])
                    torch.testing.assert_close(
                        restored.compute(),
                        retrieval_recall(full_scores, full_target, k=k).reshape(1),
                    )
                    metric.reset()
                    self.assertTrue(torch.isnan(metric.compute()).all())
                    metric.update(scores, torch.ones_like(target))
                    expected = retrieval_recall(scores, torch.ones_like(target), k=k)
                    torch.testing.assert_close(metric.compute(), expected.reshape(1))

    def test_queries_and_merge(self) -> None:
        scores = torch.tensor([0.9, 0.8, 0.7, 0.6] * 2)
        target = torch.tensor([1, 0, 1, 1, 0, 0, 1, 1])
        indexes = torch.tensor([0] * 4 + [1] * 4)
        for avg in (None, "macro"):
            for as_generator in (False, True):
                with self.subTest(avg=avg, as_generator=as_generator):
                    metric = RetrievalRecall(k=2, num_queries=2, avg=avg)
                    donors = [
                        RetrievalRecall(k=2, num_queries=2).update(x, y, i)
                        for x, y, i in zip(
                            scores.split(3), target.split(3), indexes.split(3)
                        )
                    ]
                    before = [donor.compute().clone() for donor in donors]
                    metric.merge_state((d for d in donors) if as_generator else donors)
                    expected = torch.tensor([1 / 3, 0.0])
                    if avg == "macro":
                        expected = expected.mean()
                    torch.testing.assert_close(metric.compute(), expected)
                    metric.update(
                        torch.tensor([0.1, 0.2]),
                        torch.tensor([0, 0]),
                        torch.tensor([0, 1]),
                    )
                    torch.testing.assert_close(metric.compute(), expected)
                    for donor, result in zip(donors, before):
                        torch.testing.assert_close(
                            donor.compute(), result, equal_nan=True
                        )
                    self.assertTrue(all(len(values) <= 2 for values in metric.topk))

    def test_empty_target_action_uses_all_documents(self) -> None:
        scores = torch.tensor([0.9, 0.8, 0.7, 0.6])
        for action in ("neg", "pos", "skip", "err"):
            with self.subTest(action=action):
                metric = RetrievalRecall(k=2, empty_target_action=action)
                metric.update(scores, torch.tensor([0, 0, 1, 1]))
                torch.testing.assert_close(metric.compute(), torch.tensor([0.0]))
                metric.reset()
                metric.update(scores, torch.zeros(4))
                if action == "err":
                    with self.assertRaises(ValueError):
                        metric.compute()
                else:
                    expected = {"neg": 0.0, "pos": 1.0, "skip": torch.nan}[action]
                    torch.testing.assert_close(
                        metric.compute(), torch.tensor([expected]), equal_nan=True
                    )

    def test_large_relevant_count(self) -> None:
        metric = RetrievalRecall(k=1).update(torch.tensor([0.9]), torch.tensor([1]))
        state = metric.state_dict()
        state["total_relevant"] = torch.tensor([2**24], dtype=torch.long)
        metric.load_state_dict(state)
        metric.update(torch.tensor([0.1]), torch.tensor([1.0]))
        self.assertEqual(metric.total_relevant.item(), 2**24 + 1)
        self.assertEqual(metric.total_relevant.dtype, torch.long)
