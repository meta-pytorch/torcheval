# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest

import torch
from torcheval.metrics.functional.classification.auprc import (
    binary_auprc,
    multiclass_auprc,
    multilabel_auprc,
)
from torcheval.metrics.functional.classification.binned_auprc import (
    binary_binned_auprc,
    multiclass_binned_auprc,
    multilabel_binned_auprc,
)
from torcheval.utils import random_data as rd


class TestBinaryBinnedAUPRC(unittest.TestCase):
    def _test_binary_binned_auprc_with_input(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        num_tasks: int,
        threshold: torch.Tensor | int,
        compute_result: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        my_compute_result = binary_binned_auprc(
            input,
            target,
            num_tasks=num_tasks,
            threshold=threshold,
        )
        torch.testing.assert_close(
            my_compute_result,
            compute_result,
            equal_nan=True,
            atol=1e-8,
            rtol=1e-5,
        )

        # Also test for cuda
        if torch.cuda.is_available():
            threshold_cuda = (
                threshold.to("cuda")
                if isinstance(threshold, torch.Tensor)
                else threshold
            )
            compute_result_cuda = tuple(c.to("cuda") for c in compute_result)
            my_compute_result = binary_binned_auprc(
                input.to("cuda"),
                target.to("cuda"),
                threshold=threshold_cuda,
            )
            my_compute_result_cuda = tuple(c.to("cuda") for c in my_compute_result)
            torch.testing.assert_close(
                my_compute_result_cuda,
                compute_result_cuda,
                equal_nan=True,
                atol=1e-8,
                rtol=1e-5,
            )

    def test_with_randomized_data_getter(self) -> None:
        num_bins = 5
        num_tasks = 2
        batch_size = 4
        num_updates = 1

        for _ in range(100):
            input, target, threshold = rd.get_rand_data_binned_binary(
                num_updates, num_tasks, batch_size, num_bins
            )
            input = input.reshape(shape=(num_tasks, batch_size))
            target = target.reshape(shape=(num_tasks, batch_size))

            input_positions = torch.searchsorted(
                threshold, input, right=False
            )  # get thresholds not larger than each element
            inputs_quantized = threshold[input_positions]

            compute_result = (
                binary_auprc(inputs_quantized, target, num_tasks=num_tasks),
                threshold,
            )
            self._test_binary_binned_auprc_with_input(
                input, target, num_tasks, threshold, compute_result
            )

    def test_single_task_threshold_specified_as_tensor(self) -> None:
        input = torch.tensor([0.2, 0.3, 0.4, 0.5])
        target = torch.tensor([0, 0, 1, 1])
        threshold = torch.tensor([0.0000, 0.2500, 0.7500, 1.0000])
        num_tasks = 1
        compute_result = (
            torch.tensor(2 / 3),
            torch.tensor([0.0000, 0.2500, 0.7500, 1.0000]),
        )
        self._test_binary_binned_auprc_with_input(
            input, target, num_tasks, threshold, compute_result
        )

    def test_single_task_threshold_specified_as_int(self) -> None:
        input = torch.tensor([0.2, 0.8, 0.5, 0.9])
        target = torch.tensor([0, 1, 0, 1])
        threshold = 5
        num_tasks = 1
        compute_result = (
            torch.tensor(1.0),
            torch.tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000]),
        )
        self._test_binary_binned_auprc_with_input(
            input, target, num_tasks, threshold, compute_result
        )

    def test_single_task_target_all_zero(self) -> None:
        # See N3224103 for this example
        input = torch.tensor([0.2539, 0.4058, 0.9785, 0.6885])
        target = torch.tensor([0, 0, 0, 0])
        threshold = torch.tensor([0.0000, 0.1183, 0.1195, 0.3587, 1.0000])
        num_tasks = 1
        compute_result = (torch.tensor(0.0), threshold)
        self._test_binary_binned_auprc_with_input(
            input, target, num_tasks, threshold, compute_result
        )

    def test_two_tasks_threshold_specified_as_tensor(self) -> None:
        input = torch.tensor([[0.2, 0.3, 0.4, 0.5], [0, 1, 2, 3]])
        target = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]])
        threshold = torch.tensor([0.0000, 0.2500, 0.7500, 1.0000])

        num_tasks = 2
        compute_result = (
            torch.tensor([2 / 3, 1.0000]),
            torch.tensor([0.0000, 0.2500, 0.7500, 1.0000]),
        )
        self._test_binary_binned_auprc_with_input(
            input, target, num_tasks, threshold, compute_result
        )

    def test_binary_binned_auprc_invalid_input(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "`num_tasks` has to be at least 1.",
        ):
            binary_binned_auprc(torch.rand(3, 2), torch.rand(3, 2), num_tasks=-1)

        with self.assertRaisesRegex(
            ValueError,
            "The `input` and `target` should have the same shape, "
            r"got shapes torch.Size\(\[4\]\) and torch.Size\(\[3\]\).",
        ):
            binary_binned_auprc(torch.rand(4), torch.rand(3))

        with self.assertRaisesRegex(
            ValueError,
            "`num_tasks = 1`, `input` is expected to be 1D or 2D tensor, but got shape "
            r"torch.Size\(\[4, 5, 5\]\).",
        ):
            binary_binned_auprc(torch.rand(4, 5, 5), torch.rand(4, 5, 5))

        with self.assertRaisesRegex(
            ValueError,
            "`num_tasks = 2`, `input` is expected to be 2D tensor, but got shape "
            r"torch.Size\(\[4, 5, 5\]\).",
        ):
            binary_binned_auprc(torch.rand(4, 5, 5), torch.rand(4, 5, 5), num_tasks=2)

        with self.assertRaisesRegex(
            ValueError,
            "`num_tasks = 2`, `input` is expected to be 2D tensor, but got shape "
            r"torch.Size\(\[5\]\).",
        ):
            binary_binned_auprc(torch.rand(5), torch.rand(5), num_tasks=2)

        with self.assertRaisesRegex(
            ValueError,
            r"`num_tasks = 2`, `input`'s shape is expected to be \(2, num_samples\), but got shape torch.Size\(\[4, 5\]\).",
        ):
            binary_binned_auprc(torch.rand(4, 5), torch.rand(4, 5), num_tasks=2)

        # threshold checks
        with self.assertRaisesRegex(
            ValueError, "The `threshold` should be a sorted tensor."
        ):
            binary_binned_auprc(
                torch.rand(4),
                torch.rand(4),
                threshold=torch.tensor([0.1, 0.2, 0.5, 0.7, 0.6]),
            )

        with self.assertRaisesRegex(
            ValueError,
            r"The values in `threshold` should be in the range of \[0, 1\].",
        ):
            binary_binned_auprc(
                torch.rand(4),
                torch.rand(4),
                threshold=torch.tensor([-0.1, 0.2, 0.5, 0.7]),
            )

        with self.assertRaisesRegex(
            ValueError,
            r"`threshold` should be 1-dimensional, but got 2D tensor.",
        ):
            binary_binned_auprc(
                torch.rand(4),
                torch.rand(4),
                threshold=torch.tensor([[-0.1, 0.2, 0.5, 0.7], [0.0, 0.4, 0.6, 1.0]]),
            )

        with self.assertRaisesRegex(
            ValueError,
            r"First value in `threshold` should be 0.",
        ):
            binary_binned_auprc(
                torch.rand(4),
                torch.rand(4),
                threshold=torch.tensor([0.1, 0.2, 0.5, 1.0]),
            )

        with self.assertRaisesRegex(
            ValueError,
            r"Last value in `threshold` should be 1.",
        ):
            binary_binned_auprc(
                torch.rand(4),
                torch.rand(4),
                threshold=torch.tensor([0.0, 0.2, 0.5, 0.9]),
            )


class TestMulticlassBinnedAUPRC(unittest.TestCase):
    def _test_multiclass_binned_auprc_with_input(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        num_classes: int,
        threshold: int | list[float] | torch.Tensor,
        average: str | None,
        compute_result: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        for optimization in ("vectorized", "memory"):
            my_compute_result = multiclass_binned_auprc(
                input,
                target,
                num_classes=num_classes,
                threshold=threshold,
                average=average,
                optimization=optimization,
            )
            torch.testing.assert_close(
                my_compute_result,
                compute_result,
                equal_nan=True,
                atol=1e-8,
                rtol=1e-5,
            )

            # Also test for cuda
            if torch.cuda.is_available():
                threshold_cuda = (
                    threshold.to("cuda")
                    if isinstance(threshold, torch.Tensor)
                    else threshold
                )
                compute_result_cuda = tuple(c.to("cuda") for c in compute_result)
                my_compute_result = multiclass_binned_auprc(
                    input.to("cuda"),
                    target.to("cuda"),
                    num_classes=num_classes,
                    threshold=threshold_cuda,
                    average=average,
                    optimization=optimization,
                )
                my_compute_result_cuda = tuple(c.to("cuda") for c in my_compute_result)
                torch.testing.assert_close(
                    my_compute_result_cuda,
                    compute_result_cuda,
                    equal_nan=True,
                    atol=1e-8,
                    rtol=1e-5,
                )

    def test_with_randomized_data_getter(self) -> None:
        num_classes = 4
        batch_size = 5
        num_bins = 5

        for _ in range(100):
            input, target = rd.get_rand_data_multiclass(1, num_classes, batch_size)
            input, target = input.squeeze(0), target.squeeze(0)
            threshold = torch.cat([torch.tensor([0, 1]), torch.rand(num_bins - 2)])
            threshold, _ = torch.sort(threshold)
            threshold = torch.unique(threshold)

            input_positions = torch.searchsorted(
                threshold, input, right=False
            )  # get thresholds not larger than each element
            inputs_quantized = threshold[input_positions]

            for average in (None, "macro"):
                compute_result = (
                    multiclass_auprc(
                        inputs_quantized,
                        target,
                        num_classes=num_classes,
                        average=average,
                    ),
                    threshold,
                )
                self._test_multiclass_binned_auprc_with_input(
                    input, target, num_classes, threshold, average, compute_result
                )

    def test_multiclass_binned_auprc_threshold_specified_as_tensor(self) -> None:
        input = torch.tensor(
            [
                [0.1, 0.2, 0.1, 0.4],
                [0.4, 0.2, 0.1, 0.7],
                [0.6, 0.1, 0.2, 0.4],
                [0.4, 0.2, 0.3, 0.2],
                [0.6, 0.2, 0.4, 0.5],
            ]
        )
        target = torch.tensor([0, 1, 3, 2, 0])
        num_classes = 4
        threshold = torch.tensor([0.0, 0.1, 0.4, 0.7, 0.8, 1.0])

        compute_result = (
            torch.tensor([0.3250, 0.2000, 0.2000, 0.2500]),
            torch.tensor([0.0, 0.1, 0.4, 0.7, 0.8, 1.0]),
        )
        self._test_multiclass_binned_auprc_with_input(
            input, target, num_classes, threshold, "none", compute_result
        )
        self._test_multiclass_binned_auprc_with_input(
            input, target, num_classes, threshold, None, compute_result
        )
        compute_result = (
            torch.tensor(0.24375),
            torch.tensor([0.0, 0.1, 0.4, 0.7, 0.8, 1.0]),
        )
        self._test_multiclass_binned_auprc_with_input(
            input, target, num_classes, threshold, "macro", compute_result
        )

    def test_multiclass_binned_auprc_threshold_specified_as_int(self) -> None:
        input = torch.tensor(
            [
                [0.1, 0.2, 0.1],
                [0.4, 0.2, 0.1],
                [0.6, 0.1, 0.2],
                [0.4, 0.2, 0.3],
                [0.6, 0.2, 0.4],
            ]
        )
        target = torch.tensor([0, 1, 2, 1, 0])
        num_classes = 3
        threshold = 5

        compute_result = (
            torch.tensor([0.45, 0.40, 0.20]),
            torch.tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000]),
        )
        self._test_multiclass_binned_auprc_with_input(
            input, target, num_classes, threshold, "none", compute_result
        )
        self._test_multiclass_binned_auprc_with_input(
            input, target, num_classes, threshold, None, compute_result
        )

        compute_result = (
            torch.tensor(0.35),
            torch.tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000]),
        )
        self._test_multiclass_binned_auprc_with_input(
            input, target, num_classes, threshold, "macro", compute_result
        )

    def test_multiclass_binned_auprc_invalid_input(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "`average` was not in the allowed value of .*, got micro."
        ):
            multiclass_binned_auprc(
                torch.randint(high=4, size=(4,)),
                torch.randint(high=4, size=(4,)),
                num_classes=4,
                average="micro",
            )

        with self.assertRaisesRegex(ValueError, "`num_classes` has to be at least 2."):
            multiclass_binned_auprc(torch.rand(4, 2), torch.rand(2), num_classes=1)

        with self.assertRaisesRegex(
            ValueError,
            "The `input` and `target` should have the same first dimension, "
            r"got shapes torch.Size\(\[4, 2\]\) and torch.Size\(\[3\]\).",
        ):
            multiclass_binned_auprc(torch.rand(4, 2), torch.rand(3), num_classes=2)

        with self.assertRaisesRegex(
            ValueError,
            "target should be a one-dimensional tensor, "
            r"got shape torch.Size\(\[3, 2\]\).",
        ):
            multiclass_binned_auprc(torch.rand(3, 2), torch.rand(3, 2), num_classes=2)

        with self.assertRaisesRegex(
            ValueError,
            r"input should have shape of \(num_sample, num_classes\), "
            r"got torch.Size\(\[3, 4\]\) and num_classes=2.",
        ):
            multiclass_binned_auprc(torch.rand(3, 4), torch.rand(3), num_classes=2)

        with self.assertRaisesRegex(
            ValueError, "The `threshold` should be a sorted tensor."
        ):
            multiclass_binned_auprc(
                torch.randint(high=4, size=(4,)),
                torch.randint(high=4, size=(4,)),
                num_classes=4,
                threshold=torch.tensor([0.1, 0.2, 0.5, 0.7, 0.6]),
            )

        with self.assertRaisesRegex(
            ValueError,
            r"The values in `threshold` should be in the range of \[0, 1\].",
        ):
            multiclass_binned_auprc(
                torch.randint(high=4, size=(4,)),
                torch.randint(high=4, size=(4,)),
                num_classes=4,
                threshold=torch.tensor([-0.1, 0.2, 0.5, 0.7]),
            )

        with self.assertRaisesRegex(
            ValueError,
            r"`threshold` should be 1-dimensional, but got 2D tensor.",
        ):
            multiclass_binned_auprc(
                torch.rand(4),
                torch.rand(4),
                num_classes=4,
                threshold=torch.tensor([[-0.1, 0.2, 0.5, 0.7], [0.0, 0.4, 0.6, 1.0]]),
            )

        with self.assertRaisesRegex(
            ValueError,
            r"First value in `threshold` should be 0.",
        ):
            multiclass_binned_auprc(
                torch.rand(4),
                torch.rand(4),
                num_classes=4,
                threshold=torch.tensor([0.1, 0.2, 0.5, 1.0]),
            )

        with self.assertRaisesRegex(
            ValueError,
            r"Last value in `threshold` should be 1.",
        ):
            multiclass_binned_auprc(
                torch.rand(4),
                torch.rand(4),
                num_classes=4,
                threshold=torch.tensor([0.0, 0.2, 0.5, 0.9]),
            )

        with self.assertRaisesRegex(
            ValueError,
            r"Unknown memory approach: expected 'vectorized' or 'memory', but got foo.",
        ):
            multiclass_binned_auprc(
                torch.rand(4),
                torch.rand(4),
                num_classes=4,
                threshold=5,
                optimization="foo",
            )


class TestMultilabelBinnedAUPRC(unittest.TestCase):
    def _test_multilabel_binned_auprc_with_input(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        num_labels: int,
        threshold: int | list[float] | torch.Tensor,
        average: str | None,
        compute_result: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        for optimization in ["vectorized", "memory"]:
            my_compute_result = multilabel_binned_auprc(
                input,
                target,
                num_labels=num_labels,
                threshold=threshold,
                average=average,
                optimization=optimization,
            )

            torch.testing.assert_close(
                my_compute_result,
                compute_result,
                equal_nan=True,
                atol=1e-4,
                rtol=1e-4,
            )

            # Also test for cuda
            if torch.cuda.is_available():
                threshold_cuda = (
                    threshold.to("cuda")
                    if isinstance(threshold, torch.Tensor)
                    else threshold
                )
                compute_result_cuda = tuple(c.to("cuda") for c in compute_result)
                my_compute_result = multilabel_binned_auprc(
                    input.to("cuda"),
                    target.to("cuda"),
                    num_labels=num_labels,
                    threshold=threshold_cuda,
                    average=average,
                    optimization=optimization,
                )
                my_compute_result_cuda = tuple(c.to("cuda") for c in my_compute_result)
                torch.testing.assert_close(
                    my_compute_result_cuda,
                    compute_result_cuda,
                    equal_nan=True,
                    atol=1e-4,
                    rtol=1e-4,
                )

    def test_with_randomized_data_getter(self) -> None:
        num_labels = 3
        batch_size = 5
        num_bins = 5

        for _ in range(100):
            input, target = rd.get_rand_data_multilabel(1, num_labels, batch_size)
            threshold = torch.cat([torch.tensor([0, 1]), torch.rand(num_bins - 2)])

            input, target = input.squeeze(0), target.squeeze(0)
            threshold, _ = torch.sort(threshold)
            threshold = torch.unique(threshold)

            input_positions = torch.searchsorted(
                threshold, input, right=False
            )  # get thresholds not larger than each element
            inputs_quantized = threshold[input_positions]

            for average in (None, "macro"):
                compute_result = (
                    multilabel_auprc(
                        inputs_quantized,
                        target,
                        num_labels=num_labels,
                        average=average,
                    ),
                    threshold,
                )
                self._test_multilabel_binned_auprc_with_input(
                    input, target, num_labels, threshold, average, compute_result
                )

    def test_multilabel_binned_auprc_threshold_specified_as_tensor(self) -> None:
        input = torch.tensor(
            [
                [0.75, 0.05, 0.35],
                [0.45, 0.75, 0.05],
                [0.05, 0.55, 0.75],
                [0.05, 0.65, 0.05],
            ]
        )
        target = torch.tensor([[1, 0, 1], [0, 0, 0], [0, 1, 1], [1, 1, 1]])
        num_labels = 3
        threshold = torch.tensor([0.0, 0.1, 0.4, 0.7, 0.8, 0.9, 1.0])

        compute_result = (
            torch.tensor([0.7500, 0.6667, 0.9167]),
            torch.tensor([0.0, 0.1, 0.4, 0.7, 0.8, 0.9, 1.0]),
        )
        self._test_multilabel_binned_auprc_with_input(
            input, target, num_labels, threshold, "none", compute_result
        )
        self._test_multilabel_binned_auprc_with_input(
            input, target, num_labels, threshold, None, compute_result
        )
        compute_result = (
            torch.tensor(7 / 9),
            torch.tensor([0.0, 0.1, 0.4, 0.7, 0.8, 0.9, 1.0]),
        )
        self._test_multilabel_binned_auprc_with_input(
            input, target, num_labels, threshold, "macro", compute_result
        )

    def test_multilabel_binned_auprc_threshold_specified_as_int(self) -> None:
        input = torch.tensor(
            [
                [0.75, 0.05, 0.35],
                [0.45, 0.75, 0.05],
                [0.05, 0.55, 0.75],
                [0.05, 0.65, 0.05],
            ]
        )
        target = torch.tensor([[1, 0, 1], [0, 0, 0], [0, 1, 1], [1, 1, 1]])
        num_labels = 3
        threshold = 5

        compute_result = (
            torch.tensor([0.7500, 0.6667, 0.9167]),
            torch.tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000]),
        )
        self._test_multilabel_binned_auprc_with_input(
            input, target, num_labels, threshold, "none", compute_result
        )
        self._test_multilabel_binned_auprc_with_input(
            input, target, num_labels, threshold, None, compute_result
        )

        compute_result = (
            torch.tensor(7 / 9),
            torch.tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000]),
        )
        self._test_multilabel_binned_auprc_with_input(
            input, target, num_labels, threshold, "macro", compute_result
        )

        # with threshold as 100, should reach the same result to non-binned version, since there are many small bins here.
        threshold = 100
        threshold_tensor = torch.tensor(list(range(100))) / 99
        compute_result = (
            torch.tensor([0.7500, 0.5833, 0.9167]),
            threshold_tensor,
        )
        self._test_multilabel_binned_auprc_with_input(
            input, target, num_labels, threshold, "none", compute_result
        )
        self._test_multilabel_binned_auprc_with_input(
            input, target, num_labels, threshold, None, compute_result
        )
        compute_result = (
            torch.tensor(0.75),
            threshold_tensor,
        )
        self._test_multilabel_binned_auprc_with_input(
            input, target, num_labels, threshold, "macro", compute_result
        )

    def test_multilabel_binned_auprc_invalid_input(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "Expected both input.shape and target.shape to have the same shape"
            r" but got torch.Size\(\[4, 2\]\) and torch.Size\(\[3\]\).",
        ):
            multilabel_binned_auprc(
                torch.rand(4, 2), torch.randint(low=0, high=2, size=(3,)), num_labels=3
            )

        with self.assertRaisesRegex(
            ValueError,
            "input should be a two-dimensional tensor, got shape "
            r"torch.Size\(\[3\]\).",
        ):
            multilabel_binned_auprc(
                torch.rand(3), torch.randint(low=0, high=2, size=(3,)), num_labels=3
            )

        with self.assertRaisesRegex(
            ValueError,
            "input should have shape of "
            r"\(num_sample, num_labels\), got torch.Size\(\[4, 2\]\) and num_labels=3.",
        ):
            multilabel_binned_auprc(
                torch.rand(4, 2),
                torch.randint(low=0, high=2, size=(4, 2)),
                num_labels=3,
            )
        with self.assertRaisesRegex(
            ValueError, "The `threshold` should be a sorted tensor."
        ):
            multilabel_binned_auprc(
                torch.rand(4),
                torch.randint(low=0, high=2, size=(4,)),
                num_labels=4,
                threshold=torch.tensor([0.1, 0.2, 0.5, 0.7, 0.6]),
            )
        with self.assertRaisesRegex(
            ValueError,
            r"The values in `threshold` should be in the range of \[0, 1\].",
        ):
            multilabel_binned_auprc(
                torch.rand(4),
                torch.randint(low=0, high=2, size=(4,)),
                num_labels=4,
                threshold=torch.tensor([-0.1, 0.2, 0.5, 0.7]),
            )
        with self.assertRaisesRegex(
            ValueError,
            r"The values in `threshold` should be in the range of \[0, 1\].",
        ):
            multilabel_binned_auprc(
                torch.rand(4),
                torch.randint(low=0, high=2, size=(4,)),
                num_labels=4,
                threshold=torch.tensor([0.1, 0.2, 0.5, 1.7]),
            )

        with self.assertRaisesRegex(
            ValueError,
            r"Unknown memory approach: expected 'vectorized' or 'memory', but got cpu.",
        ):
            multilabel_binned_auprc(
                torch.rand(4, 3),
                torch.randint(high=3, size=(4, 3)),
                threshold=5,
                optimization="cpu",
            )
