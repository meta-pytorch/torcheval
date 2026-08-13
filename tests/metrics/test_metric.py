# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# pyre-ignore-all-errors[16]: Undefined attribute of metric states.
# pyre-ignore-all-errors[56]: Pyre was not able to infer the type of argument

import unittest
from collections.abc import Iterable
from typing import TypeVar

import torch
from torcheval.metrics.metric import Metric
from torcheval.utils.test_utils.dummy_metric import (
    DummySumDictStateMetric,
    DummySumListStateMetric,
    DummySumMetric,
)


class MetricBaseClassTest(unittest.TestCase):
    def test_add_state_tensor(self) -> None:
        metric = DummySumMetric()
        metric._add_state("x1", torch.tensor([0.0, 1.0]))
        torch.testing.assert_close(metric.x1, torch.tensor([0.0, 1.0]))

        default_tensor = torch.tensor([0.0, 1.0])
        metric._add_state("x2", default_tensor)
        torch.testing.assert_close(metric.x2, default_tensor)
        # state is still set to the value when _add_state() is called
        default_tensor = torch.tensor([1.0, 2.0])
        self.assertFalse(torch.allclose(metric.x2, default_tensor))
        torch.testing.assert_close(metric.x2, torch.tensor([0.0, 1.0]))

    def test_add_state_tensor_list(self) -> None:
        metric = DummySumListStateMetric()
        metric._add_state("x1", [])
        torch.testing.assert_close(metric.x1, [])

        metric._add_state("x2", [torch.tensor(0.0), torch.tensor(1.0)])
        torch.testing.assert_close(metric.x2, [torch.tensor(0.0), torch.tensor(1.0)])

        tensor_list = [torch.tensor(0.0)]
        metric._add_state("x3", tensor_list)
        torch.testing.assert_close(metric.x3, tensor_list)
        tensor_list.append(torch.tensor(1.0))
        self.assertNotEqual(metric.x3, tensor_list)
        torch.testing.assert_close(metric.x3, [torch.tensor(0.0)])

    def test_add_state_tensor_dict(self) -> None:
        metric = DummySumDictStateMetric()
        metric._add_state("x1", {})
        torch.testing.assert_close(metric.x1, {})

        metric._add_state("x2", {"doc1": torch.tensor(0.0)})
        torch.testing.assert_close(metric.x2, {"doc1": torch.tensor(0.0)})

        tensor_dict = {"doc1": torch.tensor(0.0)}
        metric._add_state("x3", tensor_dict)
        torch.testing.assert_close(metric.x3, tensor_dict)
        tensor_dict["doc1"] = torch.tensor(1.0)
        self.assertNotEqual(metric.x3, tensor_dict)
        torch.testing.assert_close(metric.x3, {"doc1": torch.tensor(0.0)})

        metric._add_state(
            "x4",
            {
                "doc1": torch.tensor(0.0),
                "doc2": torch.tensor(0.0),
                "doc3": torch.tensor(0.0),
                "doc4": torch.tensor(0.0),
            },
        )
        torch.testing.assert_close(
            metric.x4,
            {
                "doc1": torch.tensor(0.0),
                "doc2": torch.tensor(0.0),
                "doc3": torch.tensor(0.0),
                "doc4": torch.tensor(0.0),
            },
        )

    def test_add_state_invalid(self) -> None:
        metric = DummySumMetric()
        with self.assertRaisesRegex(
            TypeError,
            r"The value of state variable must be.*Got x2=\[\[tensor\(0.\)\]\] instead.",
        ):
            # pyre-ignore[6]: Incompatible parameter type
            metric._add_state("x2", [[torch.tensor(0.0)]])

    def test_reset_state_tensor(self) -> None:
        metric = DummySumMetric()
        torch.testing.assert_close(metric.sum, torch.tensor(0.0))
        metric.update(torch.tensor(1.0))
        metric.update(torch.tensor(2.0))
        torch.testing.assert_close(metric.sum, torch.tensor(3.0))

        # reset once
        metric.reset()
        torch.testing.assert_close(metric.sum, torch.tensor(0.0))
        metric.update(torch.tensor(1.0))
        metric.update(torch.tensor(2.0))
        torch.testing.assert_close(metric.sum, torch.tensor(3.0))

        # reset again
        metric.reset()
        torch.testing.assert_close(metric.sum, torch.tensor(0.0))

        # update and reset chained
        metric.update(torch.tensor(1.0)).reset().update(torch.tensor(2.0))
        torch.testing.assert_close(metric.sum, torch.tensor(2.0))

    def test_reset_state_tensor_list(self) -> None:
        metric = DummySumListStateMetric()
        torch.testing.assert_close(metric.x, [])
        metric.update(torch.tensor(1.0))
        metric.update(torch.tensor(2.0))
        torch.testing.assert_close(metric.x, [torch.tensor(1.0), torch.tensor(2.0)])

        # reset once
        metric.reset()
        torch.testing.assert_close(metric.x, [])
        metric.update(torch.tensor(1.0))
        metric.update(torch.tensor(2.0))
        torch.testing.assert_close(metric.x, [torch.tensor(1.0), torch.tensor(2.0)])

        # reset again
        metric.reset()
        torch.testing.assert_close(metric.x, [])

        # update and reset chained
        metric.update(torch.tensor(1.0)).reset().update(torch.tensor(2.0))
        torch.testing.assert_close(metric.x, [torch.tensor(2.0)])

    def test_reset_state_tensor_dict(self) -> None:
        metric = DummySumDictStateMetric()
        torch.testing.assert_close(metric.x, {})
        metric.update("doc1", torch.tensor(0.0))
        metric.update("doc1", torch.tensor(1.0))
        torch.testing.assert_close(metric.x, {"doc1": torch.tensor(1.0)})

        # reset once
        metric.reset()
        torch.testing.assert_close(metric.x, {})
        metric.update("doc1", torch.tensor(0.0))
        metric.update("doc1", torch.tensor(1.0))
        torch.testing.assert_close(metric.x, {"doc1": torch.tensor(1.0)})

        # reset again
        metric.reset()
        torch.testing.assert_close(metric.x, {})

        # update and reset chained
        metric.update("doc1", torch.tensor(2.0)).reset().update(
            "doc1", torch.tensor(1.0)
        )
        torch.testing.assert_close(metric.x, {"doc1": torch.tensor(1.0)})

        # reset again
        metric.reset()
        torch.testing.assert_close(metric.x, {})
        metric.update("doc1", torch.tensor(0.0))
        metric.update("doc2", torch.tensor(1.0))
        torch.testing.assert_close(
            metric.x, {"doc1": torch.tensor(0.0), "doc2": torch.tensor(1.0)}
        )

    def test_reset_primitive_type(self) -> None:
        TDummyMetric = TypeVar("TDummyMetric")

        class DummyMetric(Metric[tuple[float, int]]):
            def __init__(self: TDummyMetric) -> None:
                super().__init__()
                self._add_state("sum", 10.0)
                self._add_state("count", 20)

            def update(self: TDummyMetric) -> None:
                self.sum += 1.0
                self.count += 1

            def compute(self: TDummyMetric) -> tuple[float, int]:
                return (self.sum, self.count)

            def merge_state(self, metrics: Iterable[TDummyMetric]) -> TDummyMetric:
                for metric in metrics:
                    self.sum += metric.sum
                    self.count += metric.count
                return self

        metric = DummyMetric()
        metric.update()
        self.assertEqual(metric.sum, 11.0)
        self.assertEqual(metric.count, 21)
        metric.reset()
        self.assertEqual(metric.sum, 10.0)
        self.assertEqual(metric.count, 20)

    def test_save_load_state_dict_state_tensor(self) -> None:
        metric = DummySumMetric()
        self.assertDictEqual(metric.state_dict(), {"sum": torch.tensor(0.0)})
        metric.update(torch.tensor(1.0))
        metric.update(torch.tensor(2.0))
        self.assertDictEqual(metric.state_dict(), {"sum": torch.tensor(3.0)})

        state_dict = metric.state_dict()
        loaded_metric = DummySumMetric()
        loaded_metric.load_state_dict(state_dict)
        self.assertDictEqual(loaded_metric.state_dict(), {"sum": torch.tensor(3.0)})
        loaded_metric.update(torch.tensor(-1.0))
        self.assertDictEqual(loaded_metric.state_dict(), {"sum": torch.tensor(2.0)})

        another_loaded_metric = DummySumMetric()
        another_loaded_metric.load_state_dict(state_dict)
        self.assertDictEqual(
            another_loaded_metric.state_dict(), {"sum": torch.tensor(3.0)}
        )

    def test_save_load_state_dict_state_tensor_list(self) -> None:
        metric = DummySumListStateMetric()
        self.assertDictEqual(metric.state_dict(), {"x": []})
        metric.update(torch.tensor(1.0))
        metric.update(torch.tensor(2.0))
        self.assertDictEqual(
            metric.state_dict(), {"x": [torch.tensor(1.0), torch.tensor(2.0)]}
        )

        state_dict = metric.state_dict()
        loaded_metric = DummySumListStateMetric()
        loaded_metric.load_state_dict(state_dict)
        self.assertDictEqual(
            loaded_metric.state_dict(), {"x": [torch.tensor(1.0), torch.tensor(2.0)]}
        )
        torch.testing.assert_close(
            loaded_metric.update(torch.tensor(-1.0)).compute(), torch.tensor(2.0)
        )
        self.assertDictEqual(
            loaded_metric.state_dict(),
            {"x": [torch.tensor(1.0), torch.tensor(2.0), torch.tensor(-1.0)]},
        )

        another_loaded_metric = DummySumListStateMetric()
        another_loaded_metric.load_state_dict(state_dict)
        self.assertDictEqual(
            another_loaded_metric.state_dict(),
            {"x": [torch.tensor(1.0), torch.tensor(2.0)]},
        )

    def test_save_load_state_dict_state_tensor_dict(self) -> None:
        metric = DummySumDictStateMetric()
        self.assertDictEqual(metric.state_dict(), {"x": {}})
        metric.update("doc1", torch.tensor(1.0))
        metric.update("doc2", torch.tensor(2.0))
        self.assertDictEqual(
            metric.state_dict(),
            {"x": {"doc1": torch.tensor(1.0), "doc2": torch.tensor(2.0)}},
        )

        state_dict = metric.state_dict()
        loaded_metric = DummySumDictStateMetric()
        loaded_metric.load_state_dict(state_dict)
        self.assertDictEqual(
            loaded_metric.state_dict(),
            {"x": {"doc1": torch.tensor(1.0), "doc2": torch.tensor(2.0)}},
        )
        torch.testing.assert_close(
            loaded_metric.update("doc2", torch.tensor(-1.0)).compute(),
            {"doc1": torch.tensor(1.0), "doc2": torch.tensor(1.0)},
        )

        self.assertDictEqual(
            loaded_metric.state_dict(),
            {"x": {"doc1": torch.tensor(1.0), "doc2": torch.tensor(1.0)}},
        )

        another_loaded_metric = DummySumDictStateMetric()
        another_loaded_metric.load_state_dict(state_dict)
        self.assertDictEqual(
            another_loaded_metric.state_dict(),
            {"x": {"doc1": torch.tensor(1.0), "doc2": torch.tensor(2.0)}},
        )

    def test_state_dict_destination_prefix_wrong_state_value(self) -> None:
        metric = DummySumMetric()
        metric.sum = "1.0"

        with self.assertRaisesRegex(
            TypeError, r"The value of state variable must be.*Got sum=1.0 instead."
        ):
            metric.state_dict()

    def test_load_state_dict_strict(self) -> None:
        metric = DummySumMetric()
        metric.update(torch.tensor(3.0))

        my_state_dict = {}
        my_state_dict["nonexistent_state"] = torch.tensor(1.0)
        loaded_metric = DummySumMetric()
        with self.assertRaisesRegex(
            RuntimeError,
            r"Error\(s\) in loading state_dict for DummySumMetric. "
            r"Encountered missing keys: {'sum'} and unexpected keys: "
            r"{'nonexistent_state'}.",
        ):
            loaded_metric.load_state_dict(my_state_dict)

        loaded_metric.load_state_dict(my_state_dict, strict=False)
        torch.testing.assert_close(loaded_metric.sum, torch.tensor(0.0))

        state_dict = metric.state_dict()
        state_dict["nonexistent_state"] = torch.tensor(1.0)
        with self.assertRaisesRegex(
            RuntimeError,
            r"Error\(s\) in loading state_dict for DummySumMetric. "
            r"Encountered missing keys: set\(\) and unexpected keys: "
            r"{'nonexistent_state'}.",
        ):
            loaded_metric.load_state_dict(state_dict)

        loaded_metric.load_state_dict(state_dict, strict=False)
        torch.testing.assert_close(loaded_metric.sum, torch.tensor(3.0))

    def test_load_state_dict_invalid_value_type(self) -> None:
        metric = DummySumMetric()
        with self.assertRaisesRegex(
            TypeError, r"The value of state variable must be.*Got sum=1.0 instead."
        ):
            metric.load_state_dict({"sum": "1.0"})

    #  `torch.cuda.is_available()` to decorator factory `unittest.skipUnless`.
    @unittest.skipUnless(
        condition=torch.cuda.is_available(), reason="This test needs a GPU host to run."
    )
    def test_to_device_state_tensor(self) -> None:
        metric = DummySumMetric().to("cuda")
        self.assertEqual(metric.sum.device.type, "cuda")
        metric.update(torch.tensor(1.0).to("cuda")).compute()
        self.assertEqual(metric.sum.device.type, "cuda")

        metric.to("cpu")
        self.assertEqual(metric.sum.device.type, "cpu")

        metric.to("cuda")
        self.assertEqual(metric.sum.device.type, "cuda")
        metric.reset()
        self.assertEqual(metric.sum.device.type, "cuda")

    def test_to_device_state_tensor_list(self) -> None:
        metric = DummySumListStateMetric()
        metric.update(torch.tensor(1.0, dtype=torch.float32)).compute()
        torch.testing.assert_close(metric.x, [torch.tensor(1.0, dtype=torch.float32)])

        metric.to("cpu", dtype=torch.float64)
        torch.testing.assert_close(metric.x, [torch.tensor(1.0, dtype=torch.float64)])

    #  `torch.cuda.is_available()` to decorator factory `unittest.skipUnless`.
    @unittest.skipUnless(
        condition=torch.cuda.is_available(), reason="This test needs a GPU host to run."
    )
    def test_to_device_state_tensor_dict(self) -> None:
        metric = DummySumDictStateMetric().to("cuda")
        metric.update("doc1", torch.tensor(1.0).to("cuda")).compute()
        torch.testing.assert_close(metric.x, {"doc1": torch.tensor(1.0, device="cuda")})

        metric.to("cpu")
        torch.testing.assert_close(metric.x, {"doc1": torch.tensor(1.0, device="cpu")})
        metric.to("cuda")
        torch.testing.assert_close(metric.x, {"doc1": torch.tensor(1.0, device="cuda")})
