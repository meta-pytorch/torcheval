# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import torch
from torcheval.metrics.classification.binned_auprc import (
    BinaryBinnedAUPRC,
    MulticlassBinnedAUPRC,
    MultilabelBinnedAUPRC,
)
from torcheval.metrics.functional.classification import (
    binary_auprc,
    multiclass_auprc,
    multilabel_auprc,
)
from torcheval.utils import random_data as rd
from torcheval.utils.test_utils.metric_class_tester import (
    BATCH_SIZE,
    MetricClassTester,
    NUM_TOTAL_UPDATES,
)


class TestBinaryBinnedAUPRC(MetricClassTester):
    def _test_binned_auprc_class_with_input(
        self,
        update_input: torch.Tensor,
        update_target: torch.Tensor,
        num_tasks: int,
        threshold: int | list[float] | torch.Tensor,
        compute_result: torch.Tensor,
    ) -> None:
        self.run_class_implementation_tests(
            metric=BinaryBinnedAUPRC(num_tasks=num_tasks, threshold=threshold),
            state_names={"num_tp", "num_fp", "num_fn"},
            update_kwargs={
                "input": update_input,
                "target": update_target,
            },
            compute_result=compute_result,
        )

    def test_binned_auprc_class_valid_input(self) -> None:
        torch.manual_seed(123)
        # test case with num_tasks=1
        input = torch.randint(high=2, size=(NUM_TOTAL_UPDATES, BATCH_SIZE))
        target = torch.randint(high=2, size=(NUM_TOTAL_UPDATES, BATCH_SIZE))
        threshold = 5
        self._test_binned_auprc_class_with_input(
            input,
            target,
            num_tasks=1,
            threshold=threshold,
            compute_result=torch.tensor(0.5117788314819336),
        )

        # test case with num_tasks=2
        torch.manual_seed(123)
        num_tasks = 2
        input = torch.rand(NUM_TOTAL_UPDATES, num_tasks, BATCH_SIZE)
        target = torch.randint(high=2, size=(NUM_TOTAL_UPDATES, num_tasks, BATCH_SIZE))
        threshold = 5
        self._test_binned_auprc_class_with_input(
            input,
            target,
            num_tasks=num_tasks,
            threshold=threshold,
            compute_result=torch.tensor([0.5810506343841553, 0.5106710195541382]),
        )

        # test case with different update shape
        num_classes = 2
        threshold = 5
        torch.manual_seed(123)
        update_input = [
            torch.rand(5),
            torch.rand(8),
            torch.rand(2),
            torch.rand(5),
        ]

        update_target = [
            torch.randint(high=num_classes, size=(5,)),
            torch.randint(high=num_classes, size=(8,)),
            torch.randint(high=num_classes, size=(2,)),
            torch.randint(high=num_classes, size=(5,)),
        ]
        compute_result = torch.tensor(0.42704516649246216)

        self.run_class_implementation_tests(
            metric=BinaryBinnedAUPRC(threshold=threshold),
            state_names={"num_tp", "num_fp", "num_fn"},
            update_kwargs={
                "input": update_input,
                "target": update_target,
            },
            compute_result=compute_result,
            num_total_updates=4,
            num_processes=2,
        )

    def test_with_randomized_binary_data_getter_single_task(self) -> None:
        batch_size = 4
        num_bins = 5

        for _ in range(10):
            update_input, update_target = rd.get_rand_data_binary(
                NUM_TOTAL_UPDATES, 1, batch_size
            )
            threshold = torch.cat([torch.tensor([0, 1]), torch.rand(num_bins - 2)])
            threshold, _ = torch.sort(threshold)
            threshold = torch.unique(threshold)

            input_positions = (
                torch.searchsorted(threshold, update_input, right=True) - 1
            )  # get thresholds not larger than each element

            # update_input, update_target original shape: [num_updates, batch_size]
            # simply reshape to a 1D tensor: [num_updates * batch_size, ]
            inputs_quantized = threshold[input_positions].reshape((-1,))
            full_target = update_target.reshape((-1,))

            compute_result = binary_auprc(
                inputs_quantized,
                full_target,
                num_tasks=1,
            )

            self._test_binned_auprc_class_with_input(
                update_input,
                update_target,
                num_tasks=1,
                threshold=threshold,
                compute_result=compute_result,
            )

    def test_with_randomized_binary_data_getter_multiple_tasks(self) -> None:
        batch_size = 4
        num_bins = 5
        num_tasks = 3

        for _ in range(10):
            update_input, update_target = rd.get_rand_data_binary(
                NUM_TOTAL_UPDATES, num_tasks, batch_size
            )
            threshold = torch.cat([torch.tensor([0, 1]), torch.rand(num_bins - 2)])
            threshold, _ = torch.sort(threshold)
            threshold = torch.unique(threshold)
            input_positions = (
                torch.searchsorted(threshold, update_input, right=True) - 1
            )  # get thresholds not larger than each element

            # update_target original shape: [num_updates, num_tasks, batch_size]
            # transpose 0, 1: [num_tasks, num_updates, batch_size]
            # then, flatten to get full_target shape: [num_tasks, num_updates * batch_size]
            inputs_quantized = threshold[input_positions].transpose(0, 1).flatten(1, 2)
            full_target = update_target.transpose(0, 1).flatten(1, 2)

            compute_result = binary_auprc(
                inputs_quantized,
                full_target,
                num_tasks=num_tasks,
            )

            self._test_binned_auprc_class_with_input(
                update_input,
                update_target,
                num_tasks=num_tasks,
                threshold=threshold,
                compute_result=compute_result,
            )

    def test_binary_binned_auprc_class_invalid_input(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "`num_tasks` has to be at least 1.",
        ):
            BinaryBinnedAUPRC(num_tasks=-1)

        with self.assertRaisesRegex(
            ValueError,
            "The `input` and `target` should have the same shape, "
            r"got shapes torch.Size\(\[4\]\) and torch.Size\(\[3\]\).",
        ):
            metric = BinaryBinnedAUPRC()
            metric.update(torch.rand(4), torch.rand(3))

        with self.assertRaisesRegex(
            ValueError,
            "`num_tasks = 1`, `input` is expected to be 1D or 2D tensor, but got shape "
            r"torch.Size\(\[\]\).",
        ):
            metric = BinaryBinnedAUPRC()
            metric.update(torch.rand(size=()), torch.rand(size=()))

        with self.assertRaisesRegex(
            ValueError,
            "`num_tasks = 1`, `input` is expected to be 1D or 2D tensor, but got shape "
            r"torch.Size\(\[4, 5, 5\]\).",
        ):
            metric = BinaryBinnedAUPRC()
            metric.update(torch.rand(4, 5, 5), torch.rand(4, 5, 5))

        with self.assertRaisesRegex(
            ValueError,
            "`num_tasks = 2`, `input` is expected to be 2D tensor, but got shape "
            r"torch.Size\(\[4, 5, 5\]\).",
        ):
            metric = BinaryBinnedAUPRC(num_tasks=2)
            metric.update(torch.rand(4, 5, 5), torch.rand(4, 5, 5))

        with self.assertRaisesRegex(
            ValueError,
            "`num_tasks = 2`, `input`'s shape is expected to be "
            r"\(2, num_samples\), but got shape torch.Size\(\[4, 5\]\).",
        ):
            metric = BinaryBinnedAUPRC(num_tasks=2)
            metric.update(torch.rand(4, 5), torch.rand(4, 5))

        with self.assertRaisesRegex(
            ValueError, "The `threshold` should be a sorted tensor."
        ):
            metric = BinaryBinnedAUPRC(
                threshold=torch.tensor([0.1, 0.2, 0.5, 0.7, 0.6]),
            )

        with self.assertRaisesRegex(
            ValueError,
            r"The values in `threshold` should be in the range of \[0, 1\].",
        ):
            metric = BinaryBinnedAUPRC(
                threshold=torch.tensor([-0.1, 0.2, 0.5, 0.7]),
            )

        with self.assertRaisesRegex(
            ValueError,
            r"`threshold` should be 1-dimensional, but got 2D tensor.",
        ):
            metric = BinaryBinnedAUPRC(
                threshold=torch.tensor([[-0.1, 0.2, 0.5, 0.7], [0.0, 0.4, 0.6, 1.0]]),
            )

        with self.assertRaisesRegex(
            ValueError,
            r"First value in `threshold` should be 0.",
        ):
            metric = BinaryBinnedAUPRC(
                threshold=torch.tensor([0.1, 0.2, 0.5, 1.0]),
            )

        with self.assertRaisesRegex(
            ValueError,
            r"Last value in `threshold` should be 1.",
        ):
            metric = BinaryBinnedAUPRC(
                threshold=torch.tensor([0.0, 0.2, 0.5, 0.9]),
            )


class TestMulticlassBinnedAUPRC(MetricClassTester):
    def _test_multiclass_binned_auprc_class_with_input(
        self,
        update_input: torch.Tensor | list[torch.Tensor],
        update_target: torch.Tensor | list[torch.Tensor],
        compute_result: torch.Tensor,
        num_classes: int,
        threshold: int | list[float] | torch.Tensor,
        average: str | None,
    ) -> None:
        for optimization in ("vectorized", "memory"):
            self.run_class_implementation_tests(
                metric=MulticlassBinnedAUPRC(
                    num_classes=num_classes,
                    threshold=threshold,
                    average=average,
                    optimization=optimization,
                ),
                state_names={"num_tp", "num_fp", "num_fn"},
                update_kwargs={
                    "input": update_input,
                    "target": update_target,
                },
                compute_result=compute_result,
                num_total_updates=len(update_input),
                num_processes=2,
            )

    def test_binned_auprc_class_base(self) -> None:
        num_classes = 4
        threshold = 5
        torch.manual_seed(123)
        input = 10 * torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE, num_classes)
        input = input.abs() / input.abs().sum(dim=-1, keepdim=True)
        target = torch.randint(high=num_classes, size=(NUM_TOTAL_UPDATES, BATCH_SIZE))

        compute_result = torch.tensor(0.2522818148136139)

        self._test_multiclass_binned_auprc_class_with_input(
            input, target, compute_result, num_classes, threshold, average="macro"
        )

    def test_binned_auprc_average_options(self) -> None:
        input = torch.tensor(
            [
                [[0.16, 0.04, 0.8]],
                [[0.1, 0.7, 0.2]],
                [[0.16, 0.8, 0.04]],
                [[0.16, 0.04, 0.8]],
            ]
        )
        target = torch.tensor([[0], [0], [1], [2]])
        num_classes = 3
        threshold = 5

        compute_result = torch.tensor(2 / 3)
        self._test_multiclass_binned_auprc_class_with_input(
            input, target, compute_result, num_classes, threshold, average="macro"
        )

        compute_result = torch.tensor([0.5000, 1.0000, 0.5000])
        self._test_multiclass_binned_auprc_class_with_input(
            input, target, compute_result, num_classes, threshold, average=None
        )

    def test_with_randomized_data_getter(self) -> None:
        num_classes = 3
        batch_size = 4
        num_bins = 5

        for _ in range(4):
            input, target = rd.get_rand_data_multiclass(1, num_classes, batch_size)
            threshold = torch.cat([torch.tensor([0, 1]), torch.rand(num_bins - 2)])

            threshold, _ = torch.sort(threshold)
            threshold = torch.unique(threshold)

            input_positions = torch.searchsorted(
                threshold, input, right=False
            )  # get thresholds not larger than each element
            inputs_quantized = threshold[input_positions]

            for average in (None, "macro"):
                compute_result = multiclass_auprc(
                    inputs_quantized,
                    target,
                    num_classes=num_classes,
                    average=average,
                )
                self._test_multiclass_binned_auprc_class_with_input(
                    input.unsqueeze(1),
                    target.unsqueeze(1),
                    compute_result,
                    num_classes,
                    threshold,
                    average,
                )

    def test_binned_auprc_class_update_input_shape_different(self) -> None:
        torch.manual_seed(123)
        num_classes = 3
        update_input = [
            torch.rand(5, num_classes),
            torch.rand(8, num_classes),
            torch.rand(2, num_classes),
            torch.rand(5, num_classes),
        ]
        update_input = [
            input.abs() / input.abs().sum(dim=-1, keepdim=True)
            for input in update_input
        ]
        update_target = [
            torch.randint(high=num_classes, size=(5,)),
            torch.randint(high=num_classes, size=(8,)),
            torch.randint(high=num_classes, size=(2,)),
            torch.randint(high=num_classes, size=(5,)),
        ]
        threshold = 5
        compute_result = torch.tensor(0.372433333333333)

        self._test_multiclass_binned_auprc_class_with_input(
            update_input,
            update_target,
            compute_result,
            num_classes,
            threshold,
            average="macro",
        )

    def test_binned_auprc_class_invalid_input(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "`average` was not in the allowed value of .*, got micro."
        ):
            metric = MulticlassBinnedAUPRC(num_classes=4, average="micro")

        with self.assertRaisesRegex(ValueError, "`num_classes` has to be at least 2."):
            metric = MulticlassBinnedAUPRC(num_classes=1)

        with self.assertRaisesRegex(
            ValueError,
            "The `input` and `target` should have the same first dimension, "
            r"got shapes torch.Size\(\[4, 2\]\) and torch.Size\(\[3\]\).",
        ):
            metric = MulticlassBinnedAUPRC(num_classes=3)
            metric.update(torch.rand(4, 2), torch.rand(3))

        with self.assertRaisesRegex(
            ValueError,
            "target should be a one-dimensional tensor, "
            r"got shape torch.Size\(\[3, 2\]\).",
        ):
            metric = MulticlassBinnedAUPRC(num_classes=2)
            metric.update(torch.rand(3, 2), torch.rand(3, 2))

        with self.assertRaisesRegex(
            ValueError,
            r"input should have shape of \(num_sample, num_classes\), "
            r"got torch.Size\(\[3, 4\]\) and num_classes=2.",
        ):
            metric = MulticlassBinnedAUPRC(num_classes=2)
            metric.update(torch.rand(3, 4), torch.rand(3))

        with self.assertRaisesRegex(
            ValueError, "The `threshold` should be a sorted tensor."
        ):
            metric = MulticlassBinnedAUPRC(
                num_classes=4, threshold=torch.tensor([0.1, 0.2, 0.5, 0.7, 0.6])
            )

        with self.assertRaisesRegex(
            ValueError,
            r"The values in `threshold` should be in the range of \[0, 1\].",
        ):
            metric = MulticlassBinnedAUPRC(
                num_classes=4, threshold=torch.tensor([-0.1, 0.2, 0.5, 0.7])
            )

        with self.assertRaisesRegex(
            ValueError,
            r"`threshold` should be 1-dimensional, but got 2D tensor.",
        ):
            metric = MulticlassBinnedAUPRC(
                num_classes=4,
                threshold=torch.tensor([[-0.1, 0.2, 0.5, 0.7], [0.0, 0.4, 0.6, 1.0]]),
            )

        with self.assertRaisesRegex(
            ValueError,
            r"First value in `threshold` should be 0.",
        ):
            metric = MulticlassBinnedAUPRC(
                num_classes=4,
                threshold=torch.tensor([0.1, 0.2, 0.5, 1.0]),
            )

        with self.assertRaisesRegex(
            ValueError,
            r"Last value in `threshold` should be 1.",
        ):
            metric = MulticlassBinnedAUPRC(
                num_classes=4,
                threshold=torch.tensor([0.0, 0.2, 0.5, 0.9]),
            )

        with self.assertRaisesRegex(
            ValueError,
            r"Unknown memory approach: expected 'vectorized' or 'memory', but got cpu.",
        ):
            metric = (
                MulticlassBinnedAUPRC(
                    num_classes=3,
                    threshold=5,
                    optimization="cpu",
                ),
            )


class TestMultilabelBinnedAUPRC(MetricClassTester):
    def _test_multilabel_binned_auprc_class_with_input(
        self,
        update_input: torch.Tensor | list[torch.Tensor],
        update_target: torch.Tensor | list[torch.Tensor],
        compute_result: torch.Tensor,
        num_labels: int,
        threshold: int | list[float] | torch.Tensor,
        average: str | None,
    ) -> None:
        for optimization in ["vectorized", "memory"]:
            self.run_class_implementation_tests(
                metric=MultilabelBinnedAUPRC(
                    num_labels=num_labels,
                    threshold=threshold,
                    average=average,
                    optimization=optimization,
                ),
                state_names={"num_tp", "num_fp", "num_fn"},
                update_kwargs={
                    "input": update_input,
                    "target": update_target,
                },
                compute_result=compute_result,
                num_total_updates=len(update_input),
                num_processes=2,
            )

    def test_multilabel_binned_auprc_class_threshold_specified_as_int(
        self,
    ) -> None:
        num_labels = 3
        input = torch.tensor(
            [
                [[0.75, 0.05, 0.35]],
                [[0.45, 0.75, 0.05]],
                [[0.05, 0.55, 0.75]],
                [[0.05, 0.65, 0.05]],
            ]
        )
        target = torch.tensor([[[1, 0, 1]], [[0, 0, 0]], [[0, 1, 1]], [[1, 1, 1]]])
        threshold = 5
        compute_result = torch.tensor([0.7500, 2 / 3, 11 / 12])
        self._test_multilabel_binned_auprc_class_with_input(
            input, target, compute_result, num_labels, threshold, None
        )

        compute_result = torch.tensor(7 / 9)
        self._test_multilabel_binned_auprc_class_with_input(
            input, target, compute_result, num_labels, threshold, "macro"
        )

        # Result should match non-binned result if there are enough thresholds
        threshold = 100
        compute_result = torch.tensor([0.7500, 7 / 12, 11 / 12])
        self._test_multilabel_binned_auprc_class_with_input(
            input, target, compute_result, num_labels, threshold, None
        )

    def test_multilabel_binned_auprc_class_threshold_specified_as_tensor(
        self,
    ) -> None:
        num_labels = 3
        input = torch.tensor(
            [
                [[0.75, 0.05, 0.35]],
                [[0.45, 0.75, 0.05]],
                [[0.05, 0.55, 0.75]],
                [[0.05, 0.65, 0.05]],
            ]
        )
        target = torch.tensor([[[1, 0, 1]], [[0, 0, 0]], [[0, 1, 1]], [[1, 1, 1]]])
        threshold = torch.tensor([0.0, 0.1, 0.4, 0.7, 0.8, 0.9, 1.0])
        compute_result = torch.tensor([0.7500, 2 / 3, 11 / 12])

        self._test_multilabel_binned_auprc_class_with_input(
            input, target, compute_result, num_labels, threshold, None
        )

        compute_result = torch.tensor(7 / 9)
        self._test_multilabel_binned_auprc_class_with_input(
            input, target, compute_result, num_labels, threshold, "macro"
        )

    def test_with_randomized_data_getter(self) -> None:
        num_labels = 3
        batch_size = 4
        num_bins = 5
        torch.manual_seed(123)
        input, target = rd.get_rand_data_multilabel(1, num_labels, batch_size)
        threshold = torch.cat([torch.tensor([0, 1]), torch.rand(num_bins - 2)])

        threshold, _ = torch.sort(threshold)
        threshold = torch.unique(threshold)

        input_positions = torch.searchsorted(
            threshold, input, right=False
        )  # get thresholds not larger than each element
        inputs_quantized = threshold[input_positions]

        for average in (None, "macro"):
            compute_result = multilabel_auprc(
                inputs_quantized,
                target,
                num_labels=num_labels,
                average=average,
            )
            self._test_multilabel_binned_auprc_class_with_input(
                input.unsqueeze(1),
                target.unsqueeze(1),
                compute_result,
                num_labels,
                threshold,
                average,
            )

    def test_multilabel_binned_auprc_class_update_input_shape_different(
        self,
    ) -> None:
        # Generated with torch.manual_seed(123)
        num_labels = 10
        update_input = [
            torch.tensor(
                [
                    [
                        0.2961,
                        0.5166,
                        0.2517,
                        0.6886,
                        0.0740,
                        0.8665,
                        0.1366,
                        0.1025,
                        0.1841,
                        0.7264,
                    ],
                    [
                        0.3153,
                        0.6871,
                        0.0756,
                        0.1966,
                        0.3164,
                        0.4017,
                        0.1186,
                        0.8274,
                        0.3821,
                        0.6605,
                    ],
                    [
                        0.8536,
                        0.5932,
                        0.6367,
                        0.9826,
                        0.2745,
                        0.6584,
                        0.2775,
                        0.8573,
                        0.8993,
                        0.0390,
                    ],
                    [
                        0.9268,
                        0.7388,
                        0.7179,
                        0.7058,
                        0.9156,
                        0.4340,
                        0.0772,
                        0.3565,
                        0.1479,
                        0.5331,
                    ],
                    [
                        0.4066,
                        0.2318,
                        0.4545,
                        0.9737,
                        0.4606,
                        0.5159,
                        0.4220,
                        0.5786,
                        0.9455,
                        0.8057,
                    ],
                ]
            ),
            torch.tensor(
                [
                    [
                        0.6775,
                        0.6087,
                        0.6179,
                        0.6932,
                        0.4354,
                        0.0353,
                        0.1908,
                        0.9268,
                        0.5299,
                        0.0950,
                    ],
                    [
                        0.5789,
                        0.9131,
                        0.0275,
                        0.1634,
                        0.3009,
                        0.5201,
                        0.3834,
                        0.4451,
                        0.0126,
                        0.7341,
                    ],
                    [
                        0.9389,
                        0.8056,
                        0.1459,
                        0.0969,
                        0.7076,
                        0.5112,
                        0.7050,
                        0.0114,
                        0.4702,
                        0.8526,
                    ],
                    [
                        0.7320,
                        0.5183,
                        0.5983,
                        0.4527,
                        0.2251,
                        0.3111,
                        0.1955,
                        0.9153,
                        0.7751,
                        0.6749,
                    ],
                    [
                        0.1166,
                        0.8858,
                        0.6568,
                        0.8459,
                        0.3033,
                        0.6060,
                        0.9882,
                        0.8363,
                        0.9010,
                        0.3950,
                    ],
                    [
                        0.8809,
                        0.1084,
                        0.5432,
                        0.2185,
                        0.3834,
                        0.3720,
                        0.5374,
                        0.9551,
                        0.7475,
                        0.4979,
                    ],
                    [
                        0.8549,
                        0.2438,
                        0.7577,
                        0.4536,
                        0.4130,
                        0.5585,
                        0.1170,
                        0.5578,
                        0.6681,
                        0.9275,
                    ],
                    [
                        0.3443,
                        0.6800,
                        0.9998,
                        0.2855,
                        0.9753,
                        0.2518,
                        0.7204,
                        0.6959,
                        0.6397,
                        0.8954,
                    ],
                ]
            ),
            torch.tensor(
                [
                    [
                        0.2979,
                        0.6314,
                        0.5028,
                        0.1239,
                        0.3786,
                        0.1661,
                        0.7211,
                        0.5449,
                        0.5490,
                        0.3483,
                    ],
                    [
                        0.5024,
                        0.3445,
                        0.6437,
                        0.9856,
                        0.5757,
                        0.2785,
                        0.1946,
                        0.5382,
                        0.1291,
                        0.1242,
                    ],
                ]
            ),
            torch.tensor(
                [
                    [
                        0.1746,
                        0.3302,
                        0.5370,
                        0.8443,
                        0.6937,
                        0.8831,
                        0.1861,
                        0.5422,
                        0.0556,
                        0.7868,
                    ],
                    [
                        0.6042,
                        0.9836,
                        0.1444,
                        0.9010,
                        0.9221,
                        0.9043,
                        0.5713,
                        0.9546,
                        0.8339,
                        0.8730,
                    ],
                    [
                        0.4675,
                        0.1163,
                        0.4938,
                        0.5938,
                        0.1594,
                        0.2132,
                        0.0206,
                        0.3247,
                        0.9355,
                        0.5855,
                    ],
                    [
                        0.4695,
                        0.5201,
                        0.8118,
                        0.0585,
                        0.1142,
                        0.3338,
                        0.2122,
                        0.7579,
                        0.8533,
                        0.0149,
                    ],
                    [
                        0.0757,
                        0.0131,
                        0.6886,
                        0.9024,
                        0.1123,
                        0.2685,
                        0.6591,
                        0.1735,
                        0.9247,
                        0.6166,
                    ],
                ]
            ),
        ]
        update_target = [
            torch.tensor(
                [
                    [1, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                    [0, 0, 0, 1, 0, 1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
                    [1, 0, 0, 1, 1, 1, 0, 1, 0, 0],
                    [0, 1, 1, 1, 0, 1, 0, 0, 1, 1],
                ]
            ),
            torch.tensor(
                [
                    [1, 0, 1, 1, 0, 0, 0, 1, 0, 1],
                    [1, 0, 0, 1, 1, 1, 1, 1, 0, 1],
                    [1, 1, 0, 1, 1, 0, 0, 0, 1, 1],
                    [0, 0, 1, 0, 0, 0, 1, 0, 1, 1],
                    [1, 0, 1, 1, 1, 0, 0, 0, 0, 1],
                    [0, 1, 1, 1, 1, 0, 1, 1, 1, 1],
                    [1, 1, 0, 0, 1, 0, 0, 1, 0, 1],
                    [1, 1, 1, 1, 1, 0, 1, 1, 1, 0],
                ]
            ),
            torch.tensor(
                [[0, 0, 1, 1, 0, 1, 0, 0, 1, 0], [1, 1, 0, 0, 0, 0, 1, 1, 1, 1]]
            ),
            torch.tensor(
                [
                    [1, 0, 1, 0, 1, 1, 1, 0, 0, 1],
                    [0, 0, 1, 0, 0, 0, 0, 0, 1, 1],
                    [0, 1, 0, 0, 0, 1, 1, 0, 1, 0],
                    [1, 0, 1, 1, 0, 0, 0, 1, 0, 1],
                    [1, 0, 0, 1, 0, 0, 1, 1, 0, 0],
                ]
            ),
        ]

        threshold = 5

        compute_result = torch.tensor(0.5810989141464233)
        self._test_multilabel_binned_auprc_class_with_input(
            update_input, update_target, compute_result, num_labels, threshold, "macro"
        )

    def test_multilabel_binned_auprc_invalid_input(self) -> None:
        metric = MultilabelBinnedAUPRC(num_labels=3)
        with self.assertRaisesRegex(
            ValueError,
            "Expected both input.shape and target.shape to have the same shape"
            r" but got torch.Size\(\[4, 2\]\) and torch.Size\(\[3\]\).",
        ):
            metric.update(torch.rand(4, 2), torch.rand(3))

        with self.assertRaisesRegex(
            ValueError,
            "input should be a two-dimensional tensor, got shape "
            r"torch.Size\(\[3\]\).",
        ):
            metric.update(torch.rand(3), torch.rand(3))

        with self.assertRaisesRegex(
            ValueError,
            "input should have shape of "
            r"\(num_sample, num_labels\), got torch.Size\(\[4, 2\]\) and num_labels=3.",
        ):
            metric.update(torch.rand(4, 2), torch.rand(4, 2))

        with self.assertRaisesRegex(
            ValueError, "The `threshold` should be a sorted tensor."
        ):
            MultilabelBinnedAUPRC(
                num_labels=3, threshold=torch.tensor([0.1, 0.2, 0.5, 0.7, 0.6])
            )

        with self.assertRaisesRegex(
            ValueError,
            r"The values in `threshold` should be in the range of \[0, 1\].",
        ):
            MultilabelBinnedAUPRC(
                num_labels=3, threshold=torch.tensor([-0.1, 0.2, 0.5, 0.7])
            )

        with self.assertRaisesRegex(
            ValueError,
            r"The values in `threshold` should be in the range of \[0, 1\].",
        ):
            MultilabelBinnedAUPRC(
                num_labels=3, threshold=torch.tensor([0.1, 0.2, 0.5, 1.7])
            )

        with self.assertRaisesRegex(
            ValueError,
            r"Unknown memory approach: expected 'vectorized' or 'memory', but got cpu.",
        ):
            MultilabelBinnedAUPRC(num_labels=3, threshold=5, optimization="cpu")
