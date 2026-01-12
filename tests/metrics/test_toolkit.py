# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import os
import unittest
import uuid
from collections.abc import Callable

import pytest
import torch
import torch.distributed.launcher as pet
from torcheval.metrics import Metric
from torcheval.metrics.toolkit import (
    classwise_converter,
    clone_metric,
    clone_metrics,
    get_synced_metric,
    get_synced_metric_collection,
    get_synced_state_dict,
    get_synced_state_dict_collection,
    reset_metrics,
    sync_and_compute,
    sync_and_compute_collection,
    to_device,
)
from torcheval.utils.test_utils.dummy_metric import (
    DummySumListStateMetric,
    DummySumMetric,
)
from torchtnt.utils import init_from_env


class MetricToolkitTest(unittest.TestCase):
    def test_metric_sync(self) -> None:
        num_processes = torch.cuda.device_count() if torch.cuda.is_available() else 4
        input_tensor = torch.rand(num_processes * 2)

        self._launch_metric_sync_test(num_processes, input_tensor, DummySumMetric)
        self._launch_metric_sync_test(
            num_processes, input_tensor, DummySumListStateMetric
        )

    def test_metric_sync_world_size_1(self) -> None:
        metric = DummySumMetric()
        synced_metric = get_synced_metric(metric)
        self.assertIsNotNone(synced_metric)
        self.assertDictEqual(metric.state_dict(), synced_metric.state_dict())

        self.assertDictEqual(get_synced_state_dict(metric), metric.state_dict())

        torch.testing.assert_close(sync_and_compute(metric), metric.compute())

    # pyre-fixmeI[56]: Pyre was not able to infer the type of the decorator `pytest.mark.cpu_and_gpu`.
    @pytest.mark.cpu_and_gpu
    def test_clone_metric(self) -> None:
        metric = DummySumMetric()
        self.assertDictEqual(clone_metric(metric).state_dict(), metric.state_dict())
        metric.update(torch.tensor(2.0))
        self.assertDictEqual(clone_metric(metric).state_dict(), metric.state_dict())
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            metric.to(device)
            cloned_metric = clone_metric(metric)
            self.assertEqual(cloned_metric.device, device)
            # pyre-ignore[16]: Undefined attribute [16]: `Metric` has no attribute `sum`.
            self.assertEqual(cloned_metric.sum.device, device)

    # pyre-fixmeI[56]: Pyre was not able to infer the type of the decorator `pytest.mark.cpu_and_gpu`.
    @pytest.mark.cpu_and_gpu
    def test_clone_metrics(self) -> None:
        metrics = [DummySumMetric(), DummySumMetric()]
        cloned = clone_metrics(metrics)
        for original, clone in zip(metrics, cloned):
            self.assertDictEqual(original.state_dict(), clone.state_dict())

        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            metrics = to_device(metrics, device)
            cloned = clone_metrics(metrics)
            for original, clone in zip(metrics, cloned):
                self.assertEqual(original.device, clone.device)
                self.assertEqual(clone.device, device)
                # pyre-fixme[16]: `Metric` has no attribute `sum`.
                self.assertEqual(clone.sum.device, device)

    @staticmethod
    def _test_per_process_metric_sync(
        input_tensor: torch.Tensor,
        metric_class: Callable[[], Metric],
    ) -> None:
        device = init_from_env()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        tc = unittest.TestCase()

        metric = metric_class().to(device)
        input_tensor = input_tensor.to(device)
        num_total_updates = len(input_tensor)
        for i in range(rank, num_total_updates, world_size):
            metric.update(input_tensor[i])

        state_dict_before_sync = metric.state_dict()
        synced_metric = get_synced_metric(metric)
        compute_result = sync_and_compute(metric)
        synced_state_dict = get_synced_state_dict(metric)

        # input metric state unchanged
        tc.assertDictEqual(metric.state_dict(), state_dict_before_sync)

        metric_with_all_updates = metric_class().to(device)
        for i in range(num_total_updates):
            metric_with_all_updates.update(input_tensor[i])
        tc.assertIsNotNone(synced_metric)
        torch.testing.assert_close(
            synced_metric.compute(),
            metric_with_all_updates.compute(),
            check_device=False,
        )
        torch.testing.assert_close(
            compute_result,
            metric_with_all_updates.compute(),
            check_device=False,
        )
        tc.assertGreater(len(synced_state_dict), 0)

    def _launch_metric_sync_test(
        self,
        num_processes: int,
        input_tensor: torch.Tensor,
        metric_class: type[Metric],
    ) -> None:
        lc = pet.LaunchConfig(
            min_nodes=1,
            max_nodes=1,
            nproc_per_node=num_processes,
            run_id=str(uuid.uuid4()),
            rdzv_backend="c10d",
            rdzv_endpoint="localhost:0",
            max_restarts=0,
            monitor_interval=1,
        )
        pet.elastic_launch(lc, entrypoint=self._test_per_process_metric_sync)(
            input_tensor,
            metric_class,
        )

    def test_reset_metrics(self) -> None:
        metric1 = DummySumMetric()
        metric2 = DummySumMetric()
        metric1.update(torch.tensor(2.0))
        metric2.update(torch.tensor(3.0))
        metric1, metric2 = reset_metrics((metric1, metric2))
        torch.testing.assert_close(metric1.compute(), torch.tensor(0.0))
        torch.testing.assert_close(metric2.compute(), torch.tensor(0.0))

    # pyre-fixmeI[56]: Pyre was not able to infer the type of the decorator `pytest.mark.cpu_and_gpu`.
    @pytest.mark.cpu_and_gpu
    def test_to_device(self) -> None:
        metric1, metric2 = DummySumMetric(), DummySumMetric()
        self.assertEqual(metric1.device.type, "cpu")
        self.assertEqual(metric2.device.type, "cpu")
        if torch.cuda.is_available():
            cuda = torch.device("cuda")
            metric1, metric2 = to_device((metric1, metric2), cuda)
            self.assertTrue(metric1.device, torch.device("cuda"))
            self.assertTrue(metric2.device, torch.device("cuda"))
            metric1, metric2 = to_device((metric1, metric2), torch.device("cpu"))
            self.assertTrue(metric1.device, torch.device("cpu"))
            self.assertTrue(metric2.device, torch.device("cpu"))

    def test_classwise_converter(self) -> None:
        metrics = torch.rand(2, 1)
        name = "SomeMetrics"
        # No labels
        classwise = classwise_converter(metrics, name)
        expected = {f"{name}_{i}": val for i, val in enumerate(metrics)}
        self.assertEqual(classwise, expected)

        # With labels
        labels = ["class1", "class2"]
        classwise = classwise_converter(metrics, name, labels)
        expected = {f"{name}_{label}": val for label, val in zip(labels, metrics)}
        self.assertEqual(classwise, expected)

        # Incorrect number of labels
        labels = ["class1"]
        with self.assertRaisesRegex(
            ValueError,
            "Number of labels [0-9]+ must be equal to the number of classes [0-9]+",
        ):
            classwise_converter(metrics, name, labels)


class MetricCollectionToolkitTest(unittest.TestCase):
    @staticmethod
    def _test_per_process_metric_collection_sync(
        input_tensor: torch.Tensor,
        metric_constructors: list[Callable[[], Metric]],
    ) -> None:
        device = init_from_env()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        tc = unittest.TestCase()

        num_total_updates = len(input_tensor)
        num_total_metrics = len(metric_constructors)

        # ==========================
        # dictwise test:
        # ==========================
        metric_names = [f"metric_{i}" for i in range(num_total_metrics)]
        metrics = {
            metric_names[i]: metric_constructors[i]().to(device)
            for i in range(num_total_metrics)
        }
        for i in range(rank, num_total_updates, world_size):
            for j in range(num_total_metrics):
                metrics[metric_names[j]].update(input_tensor[i, j])

        state_dicts_before_sync = {
            metric_names[i]: metrics[metric_names[i]].state_dict()
            for i in range(num_total_metrics)
        }
        synced_metric_dict = get_synced_metric_collection(metrics)
        compute_result_dict = sync_and_compute_collection(metrics)
        synced_state_dict_dict = get_synced_state_dict_collection(metrics)

        # input metric state unchanged
        tc.assertDictEqual(
            {
                metric_names[i]: metrics[metric_names[i]].state_dict()
                for i in range(num_total_metrics)
            },
            state_dicts_before_sync,
        )

        # make sure we get real dicts on all ranks
        tc.assertIsNotNone(synced_metric_dict)
        tc.assertIsNotNone(compute_result_dict)
        tc.assertIsNotNone(synced_state_dict_dict)

        # construct new empty metrics
        metrics_with_all_updates = {
            metric_names[i]: metric_constructors[i]().to(device)
            for i in range(num_total_metrics)
        }

        # update all the metrics in the collection so that we have a copy of the
        # metrics with all the updates applied on one rank, bypassing syncing logic.
        for i in range(num_total_updates):
            for j in range(num_total_metrics):
                metrics_with_all_updates[metric_names[j]].update(input_tensor[i, j])

        # test that each of the metrics constructed on one rank without syncing have
        # the same compute() value as the metrics updated on different ranks and synced
        # on this rank.
        for metric_id in metric_names:
            # test through the get_synced_metric_collection route
            torch.testing.assert_close(
                synced_metric_dict[metric_id].compute(),
                metrics_with_all_updates[metric_id].compute(),
                check_device=False,
            )
            # test through the synced_and_compute_collection route
            torch.testing.assert_close(
                compute_result_dict[metric_id],
                metrics_with_all_updates[metric_id].compute(),
                check_device=False,
            )
            tc.assertGreater(len(synced_state_dict_dict[metric_id]), 0)

    def _launch_collection_sync_test(
        self,
        num_processes: int,
        input_tensor: torch.Tensor,
        metric_classes: list[type[Metric]],
    ) -> None:
        lc = pet.LaunchConfig(
            min_nodes=1,
            max_nodes=1,
            nproc_per_node=num_processes,
            run_id=str(uuid.uuid4()),
            rdzv_backend="c10d",
            rdzv_endpoint="localhost:0",
            max_restarts=0,
            monitor_interval=1,
        )
        pet.elastic_launch(
            lc, entrypoint=self._test_per_process_metric_collection_sync
        )(
            input_tensor,
            metric_classes,
        )

    def test_metric_collection_sync(self) -> None:
        num_processes = torch.cuda.device_count() if torch.cuda.is_available() else 4
        num_metrics = 6
        num_updates = 3 * num_processes

        input_tensor = torch.rand(size=(num_metrics, num_updates))

        self._launch_collection_sync_test(
            num_processes,
            input_tensor,
            [DummySumMetric for i in range(num_metrics)],
        )
        self._launch_collection_sync_test(
            num_processes,
            input_tensor,
            [DummySumListStateMetric for i in range(num_metrics)],
        )

    def test_metric_collection_sync_world_size_1(self) -> None:
        metric_collection: dict[str, Metric] = {
            "m1": DummySumMetric(),
            "m2": DummySumMetric(),
        }
        synced_metric_collection = get_synced_metric_collection(metric_collection)
        self.assertIsNotNone(synced_metric_collection)
        self.assertDictEqual(synced_metric_collection, metric_collection)

        state_dict_collection = {
            "m1": metric_collection["m1"].state_dict(),
            "m2": metric_collection["m2"].state_dict(),
        }

        self.assertDictEqual(
            get_synced_state_dict_collection(metric_collection),
            state_dict_collection,
        )
        self.assertDictEqual(
            get_synced_state_dict_collection(metric_collection),
            state_dict_collection,
        )
