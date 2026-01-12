# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
import uuid

import torch
import torch.distributed as dist
import torch.distributed.launcher as pet
from pyre_extensions import none_throws
from torcheval.metrics.synclib import (
    _sync_dtype_and_shape,
    _sync_list_length,
    metrics_traversal_order,
    send_tensors,
    sync_states,
)
from torchtnt.utils.device import get_device_from_env
from torchtnt.utils.env import init_from_env

_METRIC_NAME = "tmp"


def _get_launch_config(num_processes: int) -> pet.LaunchConfig:
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
    return lc


class SynclibTest(unittest.TestCase):
    def test_sync_list_length(self) -> None:
        lc = _get_launch_config(num_processes=4)
        pet.elastic_launch(lc, entrypoint=_test_sync_list_length)()

    def test_sync_dtype_and_shape(self) -> None:
        lc = _get_launch_config(num_processes=3)
        pet.elastic_launch(lc, entrypoint=_test_sync_dtype_and_shape)()

    def test_tensor_sync_states(self) -> None:
        lc = _get_launch_config(num_processes=3)
        pet.elastic_launch(lc, entrypoint=_test_tensor_sync_state)()
        pet.elastic_launch(lc, entrypoint=_test_tensor_sync_state)(0)

    def test_tensor_list_sync_states(self) -> None:
        lc = _get_launch_config(num_processes=3)
        pet.elastic_launch(lc, entrypoint=_test_tensor_list_sync_state)()
        pet.elastic_launch(lc, entrypoint=_test_tensor_list_sync_state)(0)

    def test_tensor_dict_sync_states(self) -> None:
        lc = _get_launch_config(num_processes=2)
        pet.elastic_launch(lc, entrypoint=_test_tensor_dict_sync_state)()
        pet.elastic_launch(lc, entrypoint=_test_tensor_dict_sync_state)(0)

    def test_complex_mixed_state_sync(self) -> None:
        lc = _get_launch_config(num_processes=2)
        pet.elastic_launch(lc, entrypoint=_test_complex_mixed_state)()
        pet.elastic_launch(lc, entrypoint=_test_complex_mixed_state)(0)

    def test_empty_tensor_list_sync_state(self) -> None:
        lc = _get_launch_config(num_processes=2)
        pet.elastic_launch(lc, entrypoint=_test_empty_tensor_list_sync_state)()
        pet.elastic_launch(lc, entrypoint=_test_empty_tensor_list_sync_state)(0)

    def test_numeric_sync_state(self) -> None:
        lc = _get_launch_config(num_processes=3)
        pet.elastic_launch(lc, entrypoint=_test_numeric_sync_state)()
        pet.elastic_launch(lc, entrypoint=_test_numeric_sync_state)(0)

    # pyre-ignore[56]
    @unittest.skipUnless(
        torch.distributed.is_available(), reason="Torch distributed is needed to run"
    )
    def test_gather_uneven(self, world_size: int | None = 4) -> None:
        config = _get_launch_config(2)
        pet.elastic_launch(config, entrypoint=self._test_ddp_gather_uneven_tensors)()
        # rank 0 gather test
        pet.elastic_launch(config, entrypoint=self._test_ddp_gather_uneven_tensors)(0)

    @staticmethod
    def _test_ddp_gather_uneven_tensors(dst_rank: int | None = None) -> None:
        dist.init_process_group("gloo")
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        tensor = torch.ones(rank)
        result = send_tensors(tensor, rank=dst_rank)
        if dst_rank is None or rank == dst_rank:
            assert result is not None
            assert len(result) == world_size
            for idx in range(world_size):
                assert len(result[idx]) == idx
                assert (result[idx] == torch.ones_like(result[idx])).all()
        else:
            assert dst_rank == 0
            assert result is None

    # pyre-ignore[56]
    @unittest.skipUnless(
        torch.distributed.is_available(), reason="Torch distributed is needed to run"
    )
    def test_gather_uneven_multidim(self) -> None:
        config = _get_launch_config(2)
        pet.elastic_launch(
            config, entrypoint=self._test_ddp_gather_uneven_tensors_multidim
        )()
        pet.elastic_launch(
            config, entrypoint=self._test_ddp_gather_uneven_tensors_multidim
        )(1)

    @staticmethod
    def _test_ddp_gather_uneven_tensors_multidim(
        dst_rank: int | None = None,
    ) -> None:
        dist.init_process_group("gloo")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        tensor = torch.ones(rank + 1, 4 - rank)
        result = send_tensors(tensor, rank=dst_rank)

        if dst_rank is None or rank == dst_rank:
            assert result is not None
            assert len(result) == world_size
            for idx in range(world_size):
                val = result[idx]
                assert val.shape == (idx + 1, 4 - idx)
                assert (val == torch.ones_like(val)).all()
        else:
            assert result is None

    # pyre-ignore[56]
    @unittest.skipUnless(
        condition=torch.cuda.is_available(),
        reason="This test should only run on a GPU host.",
    )
    def test_gather_uneven_multidim_nccl(self) -> None:
        config = _get_launch_config(2)
        pet.elastic_launch(
            config, entrypoint=self._test_ddp_gather_uneven_tensors_multidim_nccl
        )()
        pet.elastic_launch(
            config, entrypoint=self._test_ddp_gather_uneven_tensors_multidim_nccl
        )(0)

    @staticmethod
    def _test_ddp_gather_uneven_tensors_multidim_nccl(
        dst_rank: int | None = None,
    ) -> None:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        tensor = torch.ones(rank + 1, 4 - rank, device=get_device_from_env())
        result = send_tensors(tensor, rank=dst_rank)

        if dst_rank is None or rank == dst_rank:
            assert result is not None
            assert len(result) == world_size
            for idx in range(world_size):
                val = result[idx]
                assert val.shape == (idx + 1, 4 - idx)
                assert (val == 1).all()
        else:
            assert result is None


def _test_sync_list_length() -> None:
    device = init_from_env()

    if dist.get_rank() == 0:
        tensor_list = [torch.tensor(1).to(device) for _ in range(1)]
    elif dist.get_rank() == 1:
        tensor_list = []
    elif dist.get_rank() == 2:
        tensor_list = [torch.tensor(1).to(device) for _ in range(4)]
    else:
        tensor_list = [torch.tensor(1).to(device) for _ in range(3)]

    length_list = _sync_list_length(tensor_list, process_group=None)
    tc = unittest.TestCase()
    tc.assertEqual(length_list, [1, 0, 4, 3])


def _test_sync_dtype_and_shape() -> None:
    device = init_from_env()

    dtype = torch.float64
    shape = torch.Size([])
    if dist.get_rank() == 0:
        data = [torch.tensor(val, dtype=dtype).to(device) for val in (3,)]
    elif dist.get_rank() == 1:
        data = [torch.tensor(val, dtype=dtype).to(device) for val in (1, 2)]
    else:
        data = []

    if len(data) > 0:
        res_dtype, res_shape = none_throws(
            _sync_dtype_and_shape(data[0], process_group=None)
        )
    else:
        res_dtype, res_shape = none_throws(
            _sync_dtype_and_shape(None, process_group=None)
        )

    tc = unittest.TestCase()
    tc.assertEqual(dtype, res_dtype)
    tc.assertEqual(shape, res_shape)

    synced_dtype = _sync_dtype_and_shape(None, process_group=None)
    tc.assertIsNone(synced_dtype)


def _test_tensor_sync_state(dst_rank: int | None = None) -> None:
    device = init_from_env()

    if dist.get_rank() == 0:
        state_data = {
            _METRIC_NAME: {
                "num_correct": torch.tensor(11.0, device=device),
                "num_total": torch.tensor(80.0, device=device),
            }
        }
    elif dist.get_rank() == 1:
        state_data = {
            _METRIC_NAME: {
                "num_correct": torch.tensor(43.0, device=device),
                "num_total": torch.tensor(50.0, device=device),
            }
        }
    else:
        state_data = {
            _METRIC_NAME: {
                "num_correct": torch.tensor(51.0, device=device),
                "num_total": torch.tensor(60.0, device=device),
            }
        }

    # pyre-ignore: Incompatible parameter type [6]:
    dict_items = metrics_traversal_order(state_data)
    synced_states = sync_states(
        state_data, {_METRIC_NAME: device}, dict_items, rank=dst_rank
    )

    tc = unittest.TestCase()

    if dst_rank is None or dist.get_rank() == dst_rank:
        tc.assertIsNotNone(synced_states)
        tc.assertEqual(len(synced_states), 3)
        tc.assertTrue(all(len(synced_states[i]) == 1 for i in range(3)))
        tc.assertTrue(all(_METRIC_NAME in synced_states[i] for i in range(3)))
        tc.assertTrue(all(len(synced_states[i][_METRIC_NAME]) == 2 for i in range(3)))
        tc.assertTrue(
            all("num_correct" in synced_states[i][_METRIC_NAME] for i in range(3))
        )
        tc.assertTrue(
            all("num_total" in synced_states[i][_METRIC_NAME] for i in range(3))
        )

        torch.testing.assert_close(
            synced_states[0][_METRIC_NAME]["num_correct"],
            torch.tensor(11.0, device=device),
        )
        torch.testing.assert_close(
            synced_states[0][_METRIC_NAME]["num_total"],
            torch.tensor(80.0, device=device),
        )
        torch.testing.assert_close(
            synced_states[1][_METRIC_NAME]["num_correct"],
            torch.tensor(43.0, device=device),
        )
        torch.testing.assert_close(
            synced_states[1][_METRIC_NAME]["num_total"],
            torch.tensor(50.0, device=device),
        )
        torch.testing.assert_close(
            synced_states[2][_METRIC_NAME]["num_correct"],
            torch.tensor(51.0, device=device),
        )
        torch.testing.assert_close(
            synced_states[2][_METRIC_NAME]["num_total"],
            torch.tensor(60.0, device=device),
        )
    else:
        tc.assertIsNone(synced_states)


def _test_tensor_list_sync_state(dst_rank: int | None = None) -> None:
    device = init_from_env()

    if dist.get_rank() == 0:
        state_data = {
            _METRIC_NAME: {
                "seen": [
                    torch.tensor(1, device=device),
                    torch.tensor(3, device=device),
                ],
                "total": [torch.tensor(1, device=device)],
            }
        }
    elif dist.get_rank() == 1:
        state_data = {
            _METRIC_NAME: {
                "seen": [torch.tensor(1, device=device)],
                "total": [torch.tensor(1, device=device)],
            }
        }
    else:
        state_data = {
            _METRIC_NAME: {
                "seen": [torch.tensor(1, device=device)],
                "total": [torch.tensor(1, device=device)],
            }
        }

    # pyre-ignore: Incompatible parameter type [6]:
    dict_items = metrics_traversal_order(state_data)
    synced_states = sync_states(
        state_data, {_METRIC_NAME: device}, dict_items, rank=dst_rank
    )

    tc = unittest.TestCase()

    if dst_rank is None or dist.get_rank() == dst_rank:
        tc.assertIsNotNone(synced_states)
        tc.assertEqual(len(synced_states), 3)
        tc.assertTrue(all(len(synced_states[i]) == 1 for i in range(3)))
        tc.assertTrue(all(_METRIC_NAME in synced_states[i] for i in range(3)))
        tc.assertTrue(all(len(synced_states[i][_METRIC_NAME]) == 2 for i in range(3)))
        tc.assertTrue(all("seen" in synced_states[i][_METRIC_NAME] for i in range(3)))
        tc.assertTrue(all("total" in synced_states[i][_METRIC_NAME] for i in range(3)))

        torch.testing.assert_close(
            synced_states[0][_METRIC_NAME]["seen"],
            [torch.tensor(1, device=device), torch.tensor(3, device=device)],
        )
        torch.testing.assert_close(
            synced_states[0][_METRIC_NAME]["total"], [torch.tensor(1, device=device)]
        )
        torch.testing.assert_close(
            synced_states[1][_METRIC_NAME]["seen"], [torch.tensor(1, device=device)]
        )
        torch.testing.assert_close(
            synced_states[1][_METRIC_NAME]["total"], [torch.tensor(1, device=device)]
        )
        torch.testing.assert_close(
            synced_states[2][_METRIC_NAME]["seen"], [torch.tensor(1, device=device)]
        )
        torch.testing.assert_close(
            synced_states[2][_METRIC_NAME]["total"], [torch.tensor(1, device=device)]
        )
    else:
        tc.assertIsNone(synced_states)


def _test_tensor_dict_sync_state(dst_rank: int | None = None) -> None:
    device = init_from_env()

    if dist.get_rank() == 0:
        state_data = {
            _METRIC_NAME: {
                "mapping": {
                    "a": torch.tensor(1, device=device),
                    "b": torch.tensor(10, device=device),
                },
            }
        }
    else:
        state_data = {
            _METRIC_NAME: {
                "mapping": {
                    "a": torch.tensor(2, device=device),
                    "b": torch.tensor(20, device=device),
                },
            }
        }

    # pyre-ignore: Incompatible parameter type [6]:
    dict_items = metrics_traversal_order(state_data)
    synced_states = sync_states(
        state_data, {_METRIC_NAME: device}, dict_items, rank=dst_rank
    )

    tc = unittest.TestCase()
    if dst_rank is None or dist.get_rank() == dst_rank:
        tc.assertIsNotNone(synced_states)
        tc.assertEqual(len(synced_states), 2)

        torch.testing.assert_close(
            synced_states[0][_METRIC_NAME]["mapping"]["a"],
            torch.tensor(1, device=device),
        )
        torch.testing.assert_close(
            synced_states[1][_METRIC_NAME]["mapping"]["a"],
            torch.tensor(2, device=device),
        )

        torch.testing.assert_close(
            synced_states[0][_METRIC_NAME]["mapping"]["b"],
            torch.tensor(10, device=device),
        )
        torch.testing.assert_close(
            synced_states[1][_METRIC_NAME]["mapping"]["b"],
            torch.tensor(20, device=device),
        )
    else:
        tc.assertIsNone(synced_states)


def _test_complex_mixed_state(dst_rank: int | None = None) -> None:
    device = init_from_env()

    if dist.get_rank() == 0:
        state_data = {
            _METRIC_NAME: {
                "seen": [
                    torch.randn((2, 3), device=device),
                    torch.randn((2, 3), device=device),
                ],
                "total": torch.tensor(1, device=device),
            }
        }
    else:
        state_data = {
            _METRIC_NAME: {
                "seen": [
                    torch.randn((2, 3), device=device),
                    torch.randn((2, 3), device=device),
                    torch.randn((2, 3), device=device),
                ],
                "total": torch.tensor(2, device=device),
            }
        }

    # pyre-ignore: Incompatible parameter type [6]:
    dict_items = metrics_traversal_order(state_data)
    synced_states = sync_states(
        state_data, {_METRIC_NAME: device}, dict_items, rank=dst_rank
    )
    tc = unittest.TestCase()

    if dst_rank is None or dist.get_rank() == dst_rank:
        tc.assertIsNotNone(synced_states)
        tc.assertEqual(len(synced_states), 2)
        tc.assertTrue(all(len(synced_states[i]) == 1 for i in range(2)))
        tc.assertTrue(all(_METRIC_NAME in synced_states[i] for i in range(2)))
        tc.assertTrue(all(len(synced_states[i][_METRIC_NAME]) == 2 for i in range(2)))
        tc.assertTrue(all("seen" in synced_states[i][_METRIC_NAME] for i in range(2)))
        tc.assertTrue(all("total" in synced_states[i][_METRIC_NAME] for i in range(2)))

        tc.assertEquals(len(synced_states[0][_METRIC_NAME]["seen"]), 2)
        tc.assertEquals(len(synced_states[1][_METRIC_NAME]["seen"]), 3)

        torch.testing.assert_close(
            synced_states[0][_METRIC_NAME]["total"], torch.tensor(1, device=device)
        )
        torch.testing.assert_close(
            synced_states[1][_METRIC_NAME]["total"], torch.tensor(2, device=device)
        )
    else:
        tc.assertIsNone(synced_states)


def _test_empty_tensor_list_sync_state(dst_rank: int | None = None) -> None:
    device = init_from_env()

    if dist.get_rank() == 0:
        state_data = {
            _METRIC_NAME: {
                "seen": [
                    torch.randn((2, 3), device=device),
                    torch.randn((2, 3), device=device),
                ],
                "total": torch.tensor(1, device=device),
            }
        }
    else:
        state_data = {
            _METRIC_NAME: {
                "seen": [],
                "total": torch.tensor(1, device=device),
            }
        }

    # pyre-ignore: Incompatible parameter type [6]:
    dict_items = metrics_traversal_order(state_data)
    synced_states = sync_states(
        state_data, {_METRIC_NAME: device}, dict_items, rank=dst_rank
    )
    tc = unittest.TestCase()
    if dst_rank is None or dist.get_rank() == dst_rank:
        tc.assertIsNotNone(synced_states)
        tc.assertEqual(len(synced_states), 2)
        tc.assertTrue(all(len(synced_states[i]) == 1 for i in range(2)))
        tc.assertTrue(all(_METRIC_NAME in synced_states[i] for i in range(2)))
        tc.assertTrue(all(len(synced_states[i][_METRIC_NAME]) == 2 for i in range(2)))
        tc.assertTrue(all("seen" in synced_states[i][_METRIC_NAME] for i in range(2)))
        tc.assertTrue(all("total" in synced_states[i][_METRIC_NAME] for i in range(2)))

        tc.assertEquals(len(synced_states[0][_METRIC_NAME]["seen"]), 2)
        tc.assertEquals(len(synced_states[1][_METRIC_NAME]["seen"]), 0)
    else:
        tc.assertIsNone(synced_states)


def _test_numeric_sync_state(dst_rank: int | None = None) -> None:
    device = init_from_env()

    if dist.get_rank() == 0:
        state_data = {
            _METRIC_NAME: {
                "num_correct": 11,
                "num_total": 80,
            }
        }
    elif dist.get_rank() == 1:
        state_data = {
            _METRIC_NAME: {
                "num_correct": 43,
                "num_total": 50,
            }
        }
    else:
        state_data = {
            _METRIC_NAME: {
                "num_correct": 51.0,
                "num_total": 60.0,
            }
        }

    # pyre-ignore: Incompatible parameter type [6]:
    dict_items = metrics_traversal_order(state_data)
    synced_states = sync_states(
        state_data, {_METRIC_NAME: device}, dict_items, rank=dst_rank
    )

    tc = unittest.TestCase()
    if dst_rank is None or dist.get_rank() == dst_rank:
        tc.assertIsNotNone(synced_states)
        tc.assertEqual(len(synced_states), 3)
        tc.assertTrue(all(len(synced_states[i]) == 1 for i in range(3)))
        tc.assertTrue(all(_METRIC_NAME in synced_states[i] for i in range(3)))
        tc.assertTrue(all(len(synced_states[i][_METRIC_NAME]) == 2 for i in range(3)))
        tc.assertTrue(
            all("num_correct" in synced_states[i][_METRIC_NAME] for i in range(3))
        )
        tc.assertTrue(
            all("num_total" in synced_states[i][_METRIC_NAME] for i in range(3))
        )

        torch.testing.assert_close(synced_states[0][_METRIC_NAME]["num_correct"], 11)
        torch.testing.assert_close(synced_states[0][_METRIC_NAME]["num_total"], 80)
        torch.testing.assert_close(synced_states[1][_METRIC_NAME]["num_correct"], 43)
        torch.testing.assert_close(synced_states[1][_METRIC_NAME]["num_total"], 50)
        torch.testing.assert_close(synced_states[2][_METRIC_NAME]["num_correct"], 51.0)
        torch.testing.assert_close(synced_states[2][_METRIC_NAME]["num_total"], 60.0)
    else:
        tc.assertIsNone(synced_states)
