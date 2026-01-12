# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import torch
from torcheval.metrics import (
    BinaryBinnedPrecisionRecallCurve,
    MulticlassBinnedPrecisionRecallCurve,
    MultilabelBinnedPrecisionRecallCurve,
)
from torcheval.metrics.functional import (
    binary_binned_precision_recall_curve,
    multiclass_binned_precision_recall_curve,
)
from torcheval.metrics.functional.classification.binned_precision_recall_curve import (
    multilabel_binned_precision_recall_curve,
)
from torcheval.utils.test_utils.metric_class_tester import (
    BATCH_SIZE,
    MetricClassTester,
    NUM_TOTAL_UPDATES,
)


class TestBinaryBinnedPrecisionRecallCurve(MetricClassTester):
    def _test_binary_binned_precision_recall_curve_class_with_input(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        threshold: int | list[float] | torch.Tensor,
    ) -> None:
        compute_result = binary_binned_precision_recall_curve(
            input.reshape(-1), target.reshape(-1), threshold=threshold
        )

        self.run_class_implementation_tests(
            metric=BinaryBinnedPrecisionRecallCurve(threshold=threshold),
            state_names={"num_tp", "num_fp", "num_fn"},
            update_kwargs={
                "input": input,
                "target": target,
            },
            compute_result=compute_result,
        )

    def test_binary_binned_precision_recall_curve_class_base(self) -> None:
        input = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE)
        target = torch.randint(high=2, size=(NUM_TOTAL_UPDATES, BATCH_SIZE))
        threshold = 200
        self._test_binary_binned_precision_recall_curve_class_with_input(
            input, target, threshold
        )

        input = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE)
        target = torch.randint(high=2, size=(NUM_TOTAL_UPDATES, BATCH_SIZE))
        threshold = torch.tensor(
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        )
        self._test_binary_binned_precision_recall_curve_class_with_input(
            input, target, threshold
        )

        input = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE)
        target = torch.randint(high=2, size=(NUM_TOTAL_UPDATES, BATCH_SIZE))
        threshold = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        self._test_binary_binned_precision_recall_curve_class_with_input(
            input, target, threshold
        )

    def test_binary_binned_precision_recall_curve_class_update_input_shape_different(
        self,
    ) -> None:
        num_classes = 2
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

        compute_result = binary_binned_precision_recall_curve(
            torch.cat(update_input, dim=0),
            torch.cat(update_target, dim=0),
        )

        self.run_class_implementation_tests(
            metric=BinaryBinnedPrecisionRecallCurve(),
            state_names={"num_tp", "num_fp", "num_fn"},
            update_kwargs={
                "input": update_input,
                "target": update_target,
            },
            compute_result=compute_result,
            num_total_updates=4,
            num_processes=2,
        )

    def test_binary_binned_precision_recall_curve_class_invalid_input(self) -> None:
        metric = BinaryBinnedPrecisionRecallCurve(
            threshold=torch.tensor([0.1, 0.5, 0.9])
        )
        with self.assertRaisesRegex(
            ValueError,
            "input should be a one-dimensional tensor, "
            r"got shape torch.Size\(\[3, 2\]\).",
        ):
            metric.update(torch.rand(3, 2), torch.rand(3))

        with self.assertRaisesRegex(
            ValueError,
            "target should be a one-dimensional tensor, "
            r"got shape torch.Size\(\[3, 2\]\).",
        ):
            metric.update(torch.rand(3), torch.rand(3, 2))

        with self.assertRaisesRegex(
            ValueError,
            "The `input` and `target` should have the same shape, "
            r"got shapes torch.Size\(\[4\]\) and torch.Size\(\[3\]\).",
        ):
            metric.update(torch.rand(4), torch.rand(3))

        with self.assertRaisesRegex(
            ValueError, "The `threshold` should be a sorted tensor."
        ):
            BinaryBinnedPrecisionRecallCurve(threshold=torch.tensor([0.1, 0.9, 0.5]))

        with self.assertRaisesRegex(
            ValueError,
            r"The values in `threshold` should be in the range of \[0, 1\].",
        ):
            BinaryBinnedPrecisionRecallCurve(threshold=torch.tensor([-0.1, 0.5, 0.9]))

        with self.assertRaisesRegex(
            ValueError,
            r"The values in `threshold` should be in the range of \[0, 1\].",
        ):
            BinaryBinnedPrecisionRecallCurve(
                threshold=torch.tensor([0.1, 0.2, 0.5, 1.7])
            )


class TestMulticlassBinnedPrecisionRecallCurve(MetricClassTester):
    def _test_multiclass_binned_precision_recall_curve_class_with_input(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        compute_result: tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor],
        num_classes: int,
        threshold: int | list[float] | torch.Tensor,
    ) -> None:
        for optimization in ("vectorized", "memory"):
            self.run_class_implementation_tests(
                metric=MulticlassBinnedPrecisionRecallCurve(
                    num_classes=num_classes,
                    threshold=threshold,
                    optimization=optimization,
                ),
                state_names={"num_tp", "num_fp", "num_fn"},
                update_kwargs={
                    "input": input,
                    "target": target,
                },
                compute_result=compute_result,
            )

    def test_multiclass_binned_precision_recall_curve_class_base(self) -> None:
        num_classes = 3
        input = torch.tensor(
            [
                [[0.1, 0.2, 0.1]],
                [[0.4, 0.2, 0.1]],
                [[0.6, 0.1, 0.2]],
                [[0.4, 0.2, 0.3]],
            ]
        )
        target = torch.tensor([[0], [1], [2], [1]])
        threshold = 3
        compute_result = (
            [
                torch.tensor([0.25, 0.0, 1.0, 1.0]),
                torch.tensor([0.5, 1.0, 1.0, 1.0]),
                torch.tensor([0.25, 1.0, 1.0, 1.0]),
            ],
            [
                torch.tensor([1.0, 0.0, 0.0, 0.0]),
                torch.tensor([1.0, 0.0, 0.0, 0.0]),
                torch.tensor([1.0, 0.0, 0.0, 0.0]),
            ],
            torch.tensor([0.0000, 0.5000, 1.0000]),
        )

        self.run_class_implementation_tests(
            metric=MulticlassBinnedPrecisionRecallCurve(
                num_classes=num_classes, threshold=threshold
            ),
            state_names={"num_tp", "num_fp", "num_fn"},
            update_kwargs={
                "input": input,
                "target": target,
            },
            compute_result=compute_result,
            num_total_updates=4,
            num_processes=2,
        )

        num_classes = 3
        input = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE, num_classes)
        target = torch.randint(high=num_classes, size=(NUM_TOTAL_UPDATES, BATCH_SIZE))
        threshold = 10

        precision, recall, thresholds = multiclass_binned_precision_recall_curve(
            input.reshape(-1, num_classes),
            target.reshape(-1),
            num_classes=num_classes,
            threshold=threshold,
        )

        compute_result = (
            [t.detach().clone() for t in precision],
            [t.detach().clone() for t in recall],
            thresholds,
        )

        self._test_multiclass_binned_precision_recall_curve_class_with_input(
            input,
            target,
            compute_result=compute_result,
            num_classes=num_classes,
            threshold=threshold,
        )

        num_classes = 5
        input = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE, num_classes)
        target = torch.randint(high=num_classes, size=(NUM_TOTAL_UPDATES, BATCH_SIZE))
        threshold = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

        precision, recall, thresholds = multiclass_binned_precision_recall_curve(
            input.reshape(-1, num_classes),
            target.reshape(-1),
            num_classes=num_classes,
            threshold=threshold,
        )

        compute_result = (
            [t.detach().clone() for t in precision],
            [t.detach().clone() for t in recall],
            thresholds,
        )

        self._test_multiclass_binned_precision_recall_curve_class_with_input(
            input,
            target,
            compute_result=compute_result,
            num_classes=num_classes,
            threshold=threshold,
        )

        num_classes = 5
        input = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE, num_classes)
        target = torch.randint(high=num_classes, size=(NUM_TOTAL_UPDATES, BATCH_SIZE))
        threshold = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        precision, recall, thresholds = multiclass_binned_precision_recall_curve(
            input.reshape(-1, num_classes),
            target.reshape(-1),
            num_classes=num_classes,
            threshold=threshold,
        )

        compute_result = (
            [t.detach().clone() for t in precision],
            [t.detach().clone() for t in recall],
            thresholds,
        )

        self._test_multiclass_binned_precision_recall_curve_class_with_input(
            input,
            target,
            compute_result=compute_result,
            num_classes=num_classes,
            threshold=threshold,
        )

    def test_multiclass_binned_precision_recall_curve_class_update_input_shape_different(
        self,
    ) -> None:
        num_classes = 10
        update_input = [
            torch.rand(5, num_classes),
            torch.rand(8, num_classes),
            torch.rand(2, num_classes),
            torch.rand(5, num_classes),
        ]

        update_target = [
            torch.randint(high=num_classes, size=(5,)),
            torch.randint(high=num_classes, size=(8,)),
            torch.randint(high=num_classes, size=(2,)),
            torch.randint(high=num_classes, size=(5,)),
        ]

        threshold = 10

        precision, recall, thresholds = multiclass_binned_precision_recall_curve(
            torch.cat(update_input, dim=0),
            torch.cat(update_target, dim=0),
            num_classes=num_classes,
            threshold=threshold,
        )

        compute_result = (
            [t.detach().clone() for t in precision],
            [t.detach().clone() for t in recall],
            thresholds,
        )

        self.run_class_implementation_tests(
            metric=MulticlassBinnedPrecisionRecallCurve(
                num_classes=num_classes, threshold=threshold
            ),
            state_names={"num_tp", "num_fp", "num_fn"},
            update_kwargs={
                "input": update_input,
                "target": update_target,
            },
            compute_result=compute_result,
            num_total_updates=4,
            num_processes=2,
        )

    def test_multiclass_binned_precision_recall_curve_class_invalid_input(self) -> None:
        metric = MulticlassBinnedPrecisionRecallCurve(num_classes=4)
        with self.assertRaisesRegex(
            ValueError,
            "The `input` and `target` should have the same first dimension, "
            r"got shapes torch.Size\(\[4, 2\]\) and torch.Size\(\[3\]\).",
        ):
            metric.update(torch.rand(4, 2), torch.rand(3))

        with self.assertRaisesRegex(
            ValueError,
            "target should be a one-dimensional tensor, "
            r"got shape torch.Size\(\[3, 2\]\).",
        ):
            metric.update(torch.rand(3, 2), torch.rand(3, 2))

        with self.assertRaisesRegex(
            ValueError,
            r"input should have shape of \(num_sample, num_classes\), "
            r"got torch.Size\(\[3, 4\]\) and num_classes=2.",
        ):
            metric = MulticlassBinnedPrecisionRecallCurve(num_classes=2)
            metric.update(torch.rand(3, 4), torch.rand(3))

        with self.assertRaisesRegex(
            ValueError, "The `threshold` should be a sorted tensor."
        ):
            MulticlassBinnedPrecisionRecallCurve(
                num_classes=5, threshold=torch.tensor([0.1, 0.9, 0.5])
            )

        with self.assertRaisesRegex(
            ValueError,
            r"The values in `threshold` should be in the range of \[0, 1\].",
        ):
            MulticlassBinnedPrecisionRecallCurve(
                num_classes=5, threshold=torch.tensor([-0.1, 0.5, 0.9])
            )

        with self.assertRaisesRegex(
            ValueError,
            r"The values in `threshold` should be in the range of \[0, 1\].",
        ):
            MulticlassBinnedPrecisionRecallCurve(
                num_classes=5, threshold=torch.tensor([0.1, 0.2, 0.5, 1.7])
            )

        with self.assertRaisesRegex(
            ValueError,
            r"Unknown memory approach: expected 'vectorized' or 'memory', but got cpu.",
        ):
            metric = MulticlassBinnedPrecisionRecallCurve(
                num_classes=3,
                threshold=5,
                optimization="cpu",
            )


class TestMultilabelBinnedPrecisionRecallCurve(MetricClassTester):
    def _test_multilabel_binned_precision_recall_curve_class_with_input(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        compute_result: tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor],
        num_labels: int,
        threshold: int | torch.Tensor,
    ) -> None:
        for optimization in ["vectorized", "memory"]:
            self.run_class_implementation_tests(
                metric=MultilabelBinnedPrecisionRecallCurve(
                    num_labels=num_labels,
                    threshold=threshold,
                    optimization=optimization,
                ),
                state_names={"num_tp", "num_fp", "num_fn"},
                update_kwargs={
                    "input": input,
                    "target": target,
                },
                compute_result=compute_result,
            )

    def test_multilabel_binned_precision_recall_curve_like_multiclass(self) -> None:
        # Same test as multiclass, except that target is specified as one-hot.
        num_labels = 3
        input = torch.tensor(
            [
                [[0.1, 0.2, 0.1]],
                [[0.4, 0.2, 0.1]],
                [[0.6, 0.1, 0.2]],
                [[0.4, 0.2, 0.3]],
            ]
        )
        target = torch.tensor([[[1, 0, 0]], [[0, 1, 0]], [[0, 0, 1]], [[0, 1, 0]]])
        threshold = 3
        compute_result = (
            [
                torch.tensor([0.25, 0.0, 1.0, 1.0]),
                torch.tensor([0.5, 1.0, 1.0, 1.0]),
                torch.tensor([0.25, 1.0, 1.0, 1.0]),
            ],
            [
                torch.tensor([1.0, 0.0, 0.0, 0.0]),
                torch.tensor([1.0, 0.0, 0.0, 0.0]),
                torch.tensor([1.0, 0.0, 0.0, 0.0]),
            ],
            torch.tensor([0.0000, 0.5000, 1.0000]),
        )

        self.run_class_implementation_tests(
            metric=MultilabelBinnedPrecisionRecallCurve(
                num_labels=num_labels, threshold=threshold
            ),
            state_names={"num_tp", "num_fp", "num_fn"},
            update_kwargs={
                "input": input,
                "target": target,
            },
            compute_result=compute_result,
            num_total_updates=4,
            num_processes=2,
        )

    def test_multilabel_binned_precision_recall_curve_threshold_specified_as_int(
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
        compute_result = (
            [
                torch.tensor([0.5000, 0.5000, 1.0000, 1.0000, 1.0000, 1.0000]),
                torch.tensor([0.5000, 2 / 3, 2 / 3, 0.0000, 1.0000, 1.0000]),
                torch.tensor([0.7500, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000]),
            ],
            [
                torch.tensor([1.0000, 0.5000, 0.5000, 0.5000, 0.0000, 0.0000]),
                torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]),
                torch.tensor([1.0000, 2 / 3, 1 / 3, 1 / 3, 0.0000, 0.0000]),
            ],
            torch.tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000]),
        )
        self.run_class_implementation_tests(
            metric=MultilabelBinnedPrecisionRecallCurve(
                num_labels=num_labels, threshold=threshold
            ),
            state_names={"num_tp", "num_fp", "num_fn"},
            update_kwargs={
                "input": input,
                "target": target,
            },
            compute_result=compute_result,
            num_total_updates=4,
            num_processes=2,
        )

    def test_multilabel_binned_precision_recall_curve_threshold_specified_as_tensor(
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
        threshold = torch.tensor([0.0, 0.2, 0.5, 0.8, 1.0])
        compute_result = (
            [
                torch.tensor([0.5000, 0.5000, 1.0000, 1.0000, 1.0000, 1.0000]),
                torch.tensor([0.5000, 2 / 3, 2 / 3, 1.0000, 1.0000, 1.0000]),
                torch.tensor([0.7500, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000]),
            ],
            [
                torch.tensor([1.0000, 0.5000, 0.5000, 0.0000, 0.0000, 0.0000]),
                torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]),
                torch.tensor([1.0000, 2 / 3, 1 / 3, 0.0000, 0.0000, 0.0000]),
            ],
            torch.tensor([0.0000, 0.2000, 0.5000, 0.8000, 1.0000]),
        )
        self.run_class_implementation_tests(
            metric=MultilabelBinnedPrecisionRecallCurve(
                num_labels=num_labels, threshold=threshold
            ),
            state_names={"num_tp", "num_fp", "num_fn"},
            update_kwargs={
                "input": input,
                "target": target,
            },
            compute_result=compute_result,
            num_total_updates=4,
            num_processes=2,
        )

    def test_multilabel_binned_precision_recall_curve_random_data(self) -> None:
        num_labels = 3
        input = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE, num_labels)
        # Note: these are BINARY labels.
        target = torch.randint(high=2, size=(NUM_TOTAL_UPDATES, BATCH_SIZE, num_labels))
        threshold = 10

        precision, recall, thresholds = multilabel_binned_precision_recall_curve(
            input.reshape(-1, num_labels),
            target.reshape(-1, num_labels),
            num_labels=num_labels,
            threshold=threshold,
        )

        compute_result = (
            [t.detach().clone() for t in precision],
            [t.detach().clone() for t in recall],
            thresholds,
        )

        self._test_multilabel_binned_precision_recall_curve_class_with_input(
            input,
            target,
            compute_result=compute_result,
            num_labels=num_labels,
            threshold=threshold,
        )

        num_labels = 5
        input = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE, num_labels)
        target = torch.randint(high=2, size=(NUM_TOTAL_UPDATES, BATCH_SIZE, num_labels))
        threshold = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

        precision, recall, thresholds = multilabel_binned_precision_recall_curve(
            input.reshape(-1, num_labels),
            target.reshape(-1, num_labels),
            num_labels=num_labels,
            threshold=threshold,
        )

        compute_result = (
            [t.detach().clone() for t in precision],
            [t.detach().clone() for t in recall],
            thresholds,
        )

        self._test_multilabel_binned_precision_recall_curve_class_with_input(
            input,
            target,
            compute_result=compute_result,
            num_labels=num_labels,
            threshold=threshold,
        )

    def test_multilabel_binned_precision_recall_curve_class_update_input_shape_different(
        self,
    ) -> None:
        num_labels = 10
        update_input = [
            torch.rand(5, num_labels),
            torch.rand(8, num_labels),
            torch.rand(2, num_labels),
            torch.rand(5, num_labels),
        ]

        update_target = [
            torch.randint(high=num_labels, size=(5, num_labels)),
            torch.randint(high=num_labels, size=(8, num_labels)),
            torch.randint(high=num_labels, size=(2, num_labels)),
            torch.randint(high=num_labels, size=(5, num_labels)),
        ]

        threshold = 10

        precision, recall, thresholds = multilabel_binned_precision_recall_curve(
            torch.cat(update_input, dim=0),
            torch.cat(update_target, dim=0),
            num_labels=num_labels,
            threshold=threshold,
        )

        compute_result = (
            [t.detach().clone() for t in precision],
            [t.detach().clone() for t in recall],
            thresholds,
        )

        self.run_class_implementation_tests(
            metric=MultilabelBinnedPrecisionRecallCurve(
                num_labels=num_labels, threshold=threshold
            ),
            state_names={"num_tp", "num_fp", "num_fn"},
            update_kwargs={
                "input": update_input,
                "target": update_target,
            },
            compute_result=compute_result,
            num_total_updates=4,
            num_processes=2,
        )

    def test_multilabel_binned_precision_recall_curve_invalid_input(self) -> None:
        metric = MultilabelBinnedPrecisionRecallCurve(num_labels=3)
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
            MultilabelBinnedPrecisionRecallCurve(
                num_labels=3, threshold=torch.tensor([0.1, 0.2, 0.5, 0.7, 0.6])
            )

        with self.assertRaisesRegex(
            ValueError,
            r"The values in `threshold` should be in the range of \[0, 1\].",
        ):
            MultilabelBinnedPrecisionRecallCurve(
                num_labels=3, threshold=torch.tensor([-0.1, 0.2, 0.5, 0.7])
            )

        with self.assertRaisesRegex(
            ValueError,
            r"The values in `threshold` should be in the range of \[0, 1\].",
        ):
            MultilabelBinnedPrecisionRecallCurve(
                num_labels=3, threshold=torch.tensor([0.1, 0.2, 0.5, 1.7])
            )

        with self.assertRaisesRegex(
            ValueError,
            r"Unknown memory approach: expected 'vectorized' or 'memory', but got cpu.",
        ):
            MultilabelBinnedPrecisionRecallCurve(
                num_labels=3, threshold=5, optimization="cpu"
            )
