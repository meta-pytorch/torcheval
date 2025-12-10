# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import torch

from torcheval.metrics import (
    BinaryRecallAtFixedPrecision,
    MultilabelRecallAtFixedPrecision,
)

from torcheval.metrics.functional.classification.recall_at_fixed_precision import (
    binary_recall_at_fixed_precision,
)
from torcheval.utils.test_utils.metric_class_tester import (
    BATCH_SIZE,
    MetricClassTester,
    NUM_TOTAL_UPDATES,
)


class TestBinaryRecallAtFixedPrecision(MetricClassTester):
    def _test_binary_recall_at_fixed_precision_class_with_input(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        min_precision: float,
    ) -> None:
        input_tensors = input.reshape(-1)
        target_tensors = target.reshape(-1)

        recall, thresholds = binary_recall_at_fixed_precision(
            input_tensors, target_tensors, min_precision=min_precision
        )

        compute_result = (recall, thresholds)
        self.run_class_implementation_tests(
            metric=BinaryRecallAtFixedPrecision(min_precision=min_precision),
            state_names={"inputs", "targets"},
            update_kwargs={
                "input": input,
                "target": target,
            },
            compute_result=compute_result,
        )

    def test_binary_recall_at_fixed_precision_class_base(self) -> None:
        input = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE)
        target = torch.randint(high=2, size=(NUM_TOTAL_UPDATES, BATCH_SIZE))
        self._test_binary_recall_at_fixed_precision_class_with_input(
            input, target, min_precision=0.5
        )

        input = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE)
        target = torch.randint(high=2, size=(NUM_TOTAL_UPDATES, BATCH_SIZE))
        self._test_binary_recall_at_fixed_precision_class_with_input(
            input, target, min_precision=0.5
        )

    def test_binary_recall_at_fixed_precision_class_update_input_shape_different(
        self,
    ) -> None:
        num_classes = 2
        min_precision = 0.5
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

        compute_result = binary_recall_at_fixed_precision(
            torch.cat(update_input, dim=0),
            torch.cat(update_target, dim=0),
            min_precision=min_precision,
        )

        self.run_class_implementation_tests(
            metric=BinaryRecallAtFixedPrecision(min_precision=min_precision),
            state_names={"inputs", "targets"},
            update_kwargs={
                "input": update_input,
                "target": update_target,
            },
            compute_result=tuple(compute_result),
            num_total_updates=4,
            num_processes=2,
        )

    def test_binary_recall_at_fixed_precision_class_invalid_input(self) -> None:
        metric = BinaryRecallAtFixedPrecision(min_precision=0.5)
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
            ValueError,
            r"Expected min_precision to be a float in the \[0, 1\] range"
            r" but got 1.1.",
        ):
            metric = BinaryRecallAtFixedPrecision(min_precision=1.1)
            metric.update(torch.rand(4), torch.rand(4))

    def test_binary_recall_at_fixed_precision_with_weights_basic(self) -> None:
        # Setup: Create input, target, and weight tensors
        input = torch.tensor([0.1, 0.4, 0.6, 0.6, 0.6, 0.35, 0.8])
        target = torch.tensor([0, 0, 1, 1, 1, 1, 1])
        weight = torch.tensor([1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0])
        min_precision = 0.5

        # Execute: Create metric with use_weights=True and update
        metric = BinaryRecallAtFixedPrecision(
            min_precision=min_precision, use_weights=True
        )
        metric.update(input, target, weight)

        # Assert: Compute should return recall and threshold tensors
        recall, threshold = metric.compute()
        self.assertIsInstance(recall, torch.Tensor)
        self.assertIsInstance(threshold, torch.Tensor)
        # Verify recall is a valid value between 0 and 1
        self.assertGreaterEqual(recall.item(), 0.0)
        self.assertLessEqual(recall.item(), 1.0)

    def test_binary_recall_at_fixed_precision_with_weights_state_names(
        self,
    ) -> None:
        # Setup: Create metric with use_weights=True
        metric = BinaryRecallAtFixedPrecision(min_precision=0.5, use_weights=True)

        # Execute: Check that weights state is present
        state_dict = metric.state_dict()

        # Assert: Verify that weights are included in state names
        self.assertIn("weights", state_dict)
        self.assertIn("inputs", state_dict)
        self.assertIn("targets", state_dict)

    def test_binary_recall_at_fixed_precision_with_weights_multiple_updates(
        self,
    ) -> None:
        # Setup: Create multiple batches of input, target, and weight tensors
        min_precision = 0.5
        metric = BinaryRecallAtFixedPrecision(
            min_precision=min_precision, use_weights=True
        )

        # Execute: Update metric with multiple batches
        input1 = torch.tensor([0.1, 0.4, 0.6])
        target1 = torch.tensor([0, 0, 1])
        weight1 = torch.tensor([1.0, 1.0, 2.0])
        metric.update(input1, target1, weight1)

        input2 = torch.tensor([0.7, 0.8, 0.9])
        target2 = torch.tensor([1, 1, 1])
        weight2 = torch.tensor([1.5, 1.0, 1.0])
        metric.update(input2, target2, weight2)

        # Assert: Compute should successfully process accumulated data
        recall, threshold = metric.compute()
        self.assertIsInstance(recall, torch.Tensor)
        self.assertIsInstance(threshold, torch.Tensor)
        self.assertGreaterEqual(recall.item(), 0.0)
        self.assertLessEqual(recall.item(), 1.0)

    def test_binary_recall_at_fixed_precision_missing_weights_error(self) -> None:
        # Setup: Create metric with use_weights=True
        metric = BinaryRecallAtFixedPrecision(min_precision=0.5, use_weights=True)
        input = torch.tensor([0.1, 0.4, 0.6, 0.6])
        target = torch.tensor([0, 0, 1, 1])

        # Execute & Assert: Verify error is raised when weights are not provided
        with self.assertRaisesRegex(
            ValueError,
            "Weights must be provided when use_weights=True",
        ):
            metric.update(input, target)


class TestMultilabelRecallAtFixedPrecision(MetricClassTester):
    def _test_multilabel_recall_at_fixed_precision_class_with_input(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        num_labels: int,
        min_precision: float,
    ) -> None:
        assert isinstance(num_labels, int)
        assert isinstance(min_precision, float)
        input_tensors = input.reshape(-1, num_labels)
        target_tensors = target.reshape(-1, num_labels)
        recall, thresholds = [], []
        for idx in range(num_labels):
            r, t = binary_recall_at_fixed_precision(
                input_tensors[:, idx],
                target_tensors[:, idx],
                min_precision=min_precision,
            )
            recall.append(r)
            thresholds.append(t)

        compute_result = (recall, thresholds)
        self.run_class_implementation_tests(
            metric=MultilabelRecallAtFixedPrecision(
                num_labels=num_labels, min_precision=min_precision
            ),
            state_names={"inputs", "targets"},
            update_kwargs={
                "input": input,
                "target": target,
            },
            compute_result=compute_result,
        )

    def test_multilabel_recall_at_fixed_precision_class_base(self) -> None:
        num_labels = 4
        min_precision = 0.5
        input = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE, num_labels)
        target = torch.randint(high=2, size=(NUM_TOTAL_UPDATES, BATCH_SIZE, num_labels))
        self._test_multilabel_recall_at_fixed_precision_class_with_input(
            input, target, num_labels=num_labels, min_precision=min_precision
        )

    def test_multilabel_recall_at_fixed_precision_class_label_not_exist(self) -> None:
        num_labels = 4
        min_precision = 0.5
        num_labels = 4
        input = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE, num_labels)
        target = torch.randint(high=2, size=(NUM_TOTAL_UPDATES, BATCH_SIZE, num_labels))
        # change last entries in target to 0
        target[:, :, -1] = 0
        self._test_multilabel_recall_at_fixed_precision_class_with_input(
            input, target, num_labels=num_labels, min_precision=min_precision
        )

        num_labels = 8
        input = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE, num_labels)
        target = torch.randint(high=2, size=(NUM_TOTAL_UPDATES, BATCH_SIZE, num_labels))
        target[:, :, 0] = 0
        self._test_multilabel_recall_at_fixed_precision_class_with_input(
            input, target, num_labels=num_labels, min_precision=min_precision
        )

    def test_multilabel_recall_at_fixed_precision_class_update_input_shape_different(
        self,
    ) -> None:
        num_labels = 3
        min_precision = 0.5
        update_input = [
            torch.rand(5, num_labels),
            torch.rand(8, num_labels),
            torch.rand(2, num_labels),
            torch.rand(5, num_labels),
        ]
        update_target = [
            torch.randint(high=2, size=(5, num_labels)),
            torch.randint(high=2, size=(8, num_labels)),
            torch.randint(high=2, size=(2, num_labels)),
            torch.randint(high=2, size=(5, num_labels)),
        ]
        input_tensors = torch.cat(update_input, dim=0)
        target_tensors = torch.cat(update_target, dim=0)

        recall, thresholds = [], []
        for idx in range(num_labels):
            r, t = binary_recall_at_fixed_precision(
                input_tensors[:, idx],
                target_tensors[:, idx],
                min_precision=min_precision,
            )
            recall.append(r)
            thresholds.append(t)

        compute_result = (recall, thresholds)

        self.run_class_implementation_tests(
            metric=MultilabelRecallAtFixedPrecision(
                num_labels=num_labels, min_precision=min_precision
            ),
            state_names={"inputs", "targets"},
            update_kwargs={
                "input": update_input,
                "target": update_target,
            },
            compute_result=tuple(compute_result),
            num_total_updates=4,
            num_processes=2,
        )

    def test_multilabel_recall_at_fixed_precision_class_invalid_input(self) -> None:
        metric = MultilabelRecallAtFixedPrecision(num_labels=4, min_precision=0.5)
        with self.assertRaisesRegex(
            ValueError,
            "Expected both input.shape and target.shape to have the same shape"
            r" but got torch.Size\(\[4, 2\]\) and torch.Size\(\[4, 3\]\).",
        ):
            metric.update(torch.rand(4, 2), torch.randint(high=2, size=(4, 3)))

        with self.assertRaisesRegex(
            ValueError,
            r"input should have shape of \(num_sample, num_labels\), "
            r"got torch.Size\(\[4, 2\]\) and num_labels=3.",
        ):
            metric = MultilabelRecallAtFixedPrecision(num_labels=3, min_precision=0.5)
            metric.update(torch.rand(4, 2), torch.randint(high=2, size=(4, 2)))

        with self.assertRaisesRegex(
            ValueError,
            r"Expected min_precision to be a float in the \[0, 1\] range"
            r" but got 1.1.",
        ):
            metric = MultilabelRecallAtFixedPrecision(num_labels=3, min_precision=1.1)
            metric.update(torch.rand(4, 3), torch.randint(high=2, size=(4, 3)))
