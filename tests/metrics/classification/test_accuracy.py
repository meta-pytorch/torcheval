# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import numpy as np

import torch

from sklearn.metrics import accuracy_score
from torcheval.metrics import (
    BinaryAccuracy,
    MulticlassAccuracy,
    MultilabelAccuracy,
    TopKMultilabelAccuracy,
)
from torcheval.utils.test_utils.metric_class_tester import (
    BATCH_SIZE,
    MetricClassTester,
    NUM_TOTAL_UPDATES,
)


class TestBinaryAccuracy(MetricClassTester):
    def _test_binary_accuracy_with_input(
        self, input: torch.Tensor, target: torch.Tensor
    ) -> None:
        input_np = input.squeeze().numpy().round()
        target_np = target.squeeze().numpy()

        sklearn_result = torch.tensor(
            np.stack([accuracy_score(t, i) for t, i in zip(input_np, target_np)])
        )
        self.run_class_implementation_tests(
            metric=BinaryAccuracy(),
            state_names={"num_correct", "num_total"},
            update_kwargs={"input": input, "target": target},
            compute_result=sklearn_result.to(torch.float32).mean(),
        )

        self.run_class_implementation_tests(
            metric=BinaryAccuracy(),
            state_names={"num_correct", "num_total"},
            update_kwargs={"input": input, "target": target},
            compute_result=sklearn_result.to(torch.float32).mean(),
        )

    def test_binary_accuracy(self) -> None:
        num_classes = 2
        input = torch.randint(high=num_classes, size=(NUM_TOTAL_UPDATES, BATCH_SIZE))
        target = torch.randint(high=num_classes, size=(NUM_TOTAL_UPDATES, BATCH_SIZE))

        self._test_binary_accuracy_with_input(input, target)

    def test_binary_accuracy_with_rounding(self) -> None:
        num_classes = 2
        input = torch.rand(size=(NUM_TOTAL_UPDATES, BATCH_SIZE))
        target = torch.randint(high=num_classes, size=(NUM_TOTAL_UPDATES, BATCH_SIZE))

        self._test_binary_accuracy_with_input(input, target)

    def test_binary_accuracy_class_invalid_input(self) -> None:
        metric = BinaryAccuracy()
        with self.assertRaisesRegex(
            ValueError,
            "The `input` and `target` should have the same dimensions, "
            r"got shapes torch.Size\(\[4, 2\]\) and torch.Size\(\[3\]\).",
        ):
            metric.update(torch.rand(4, 2), torch.rand(3))


class TestMulticlassAccuracy(MetricClassTester):
    def _test_accuracy_class_with_input(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        num_classes: int,
    ) -> None:
        input_tensors = [t.squeeze() for t in input]
        input_tensors = [
            torch.argmax(t, dim=1) if t.ndim == 2 else t for t in input_tensors
        ]
        target_tensors = [t.squeeze() for t in target]

        target_np = torch.stack(target_tensors).flatten().numpy()
        input_np = torch.stack(input_tensors).flatten().numpy()
        compute_result = torch.tensor(accuracy_score(target_np, input_np)).to(
            torch.float32
        )
        self.run_class_implementation_tests(
            metric=MulticlassAccuracy(),
            state_names={"num_correct", "num_total"},
            update_kwargs={"input": input, "target": target},
            compute_result=compute_result,
        )

    def test_accuracy_class_base(self) -> None:
        num_classes = 4
        input = torch.randint(high=num_classes, size=(NUM_TOTAL_UPDATES, BATCH_SIZE))
        target = torch.randint(high=num_classes, size=(NUM_TOTAL_UPDATES, BATCH_SIZE))
        self._test_accuracy_class_with_input(input, target, num_classes)

        input = torch.rand(NUM_TOTAL_UPDATES, BATCH_SIZE, num_classes)
        self._test_accuracy_class_with_input(input, target, num_classes)

    def test_accuracy_class_average(self) -> None:
        num_classes = 4
        # high=num_classes-1 gives us NaN value for the last class
        input = torch.randint(high=num_classes, size=(NUM_TOTAL_UPDATES, BATCH_SIZE))
        target = torch.randint(
            high=num_classes - 1, size=(NUM_TOTAL_UPDATES, BATCH_SIZE)
        )

        input_flattened = input.flatten()
        target_flattened = target.flatten()
        accuracy_per_class = np.empty(num_classes)
        for i in range(num_classes):
            mask = target_flattened == i
            if mask.sum().item() == 0:
                accuracy_per_class[i] = np.nan
            else:
                accuracy_per_class[i] = accuracy_score(
                    target_flattened[mask].numpy(),
                    input_flattened[mask].numpy(),
                )


        self.run_class_implementation_tests(
            metric=MulticlassAccuracy(num_classes=num_classes, average="macro"),
            state_names={"num_correct", "num_total"},
            update_kwargs={"input": input, "target": target},
            compute_result=torch.tensor(
                np.mean(accuracy_per_class[~np.isnan(accuracy_per_class)])
            ).to(torch.float32),
        )

        self.run_class_implementation_tests(
            metric=MulticlassAccuracy(num_classes=num_classes, average=None),
            state_names={"num_correct", "num_total"},
            update_kwargs={"input": input, "target": target},
            compute_result=torch.tensor(accuracy_per_class).view(-1).to(torch.float32),
        )

    def test_accuracy_class_invalid_intialization(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "`average` was not in the allowed value of .*, got weighted."
        ):
            MulticlassAccuracy(num_classes=4, average="weighted")

        with self.assertRaisesRegex(
            ValueError,
            "num_classes should be a positive number when average=None. Got num_classes=None",
        ):
            MulticlassAccuracy(average=None)

    def test_accuracy_class_invalid_input(self) -> None:
        metric = MulticlassAccuracy()
        with self.assertRaisesRegex(
            ValueError,
            "The `input` and `target` should have the same first dimension, "
            r"got shapes torch.Size\(\[4, 2\]\) and torch.Size\(\[3\]\).",
        ):
            metric.update(torch.rand(4, 2), torch.rand(3))

        with self.assertRaisesRegex(
            ValueError, "target should be a one-dimensional tensor, got shape ."
        ):
            metric.update(torch.rand(BATCH_SIZE, 1), torch.rand(BATCH_SIZE, 1))

        with self.assertRaisesRegex(ValueError, "input should have shape"):
            metric.update(torch.rand(BATCH_SIZE, 2, 1), torch.rand(BATCH_SIZE))


class TestMultilabelAccuracy(MetricClassTester):
    def _test_exact_match_with_input(
        self, input: torch.Tensor, target: torch.Tensor
    ) -> None:
        input_np = input.squeeze().numpy().round()
        target_np = target.squeeze().numpy()

        sklearn_result = torch.tensor(
            np.stack([accuracy_score(t, i) for t, i in zip(input_np, target_np)])
        )
        self.run_class_implementation_tests(
            metric=MultilabelAccuracy(),
            state_names={"num_correct", "num_total"},
            update_kwargs={"input": input, "target": target},
            compute_result=sklearn_result.to(torch.float32).mean(),
        )

    def _test_hamming_with_input(
        self, input: torch.Tensor, target: torch.Tensor
    ) -> None:
        input_np = input.numpy().round()
        target_np = target.numpy()
        sklearn_result = torch.tensor(
            np.stack(
                [
                    accuracy_score(t.flatten(), i.flatten())
                    for t, i in zip(input_np, target_np)
                ]
            )
        )

        self.run_class_implementation_tests(
            metric=MultilabelAccuracy(criteria="hamming"),
            state_names={"num_correct", "num_total"},
            update_kwargs={"input": input, "target": target},
            compute_result=sklearn_result.to(torch.float32).mean(),
        )

    def test_multilabel_accuracy_exact_match(self) -> None:
        num_classes = 2
        input = torch.randint(0, 2, size=(NUM_TOTAL_UPDATES, BATCH_SIZE, num_classes))
        target = torch.randint(0, 2, size=(NUM_TOTAL_UPDATES, BATCH_SIZE, num_classes))

        self._test_exact_match_with_input(input, target)

    def test_multilabel_accuracy_exact_match_with_rounding(self) -> None:
        num_classes = 2
        input = torch.rand(size=(NUM_TOTAL_UPDATES, BATCH_SIZE, num_classes))
        target = torch.randint(0, 2, size=(NUM_TOTAL_UPDATES, BATCH_SIZE, num_classes))

        self._test_exact_match_with_input(input, target)

    def test_multilabel_accuracy_hamming(self) -> None:
        num_classes = 2
        input = torch.randint(0, 2, size=(NUM_TOTAL_UPDATES, BATCH_SIZE, num_classes))
        target = torch.randint(0, 2, size=(NUM_TOTAL_UPDATES, BATCH_SIZE, num_classes))

        self._test_hamming_with_input(input, target)

    def test_multilabel_accuracy_hamming_with_rounding(self) -> None:
        num_classes = 2
        input = torch.rand(size=(NUM_TOTAL_UPDATES, BATCH_SIZE, num_classes))
        target = torch.randint(0, 2, size=(NUM_TOTAL_UPDATES, BATCH_SIZE, num_classes))

        self._test_hamming_with_input(input, target)

    def test_multilabel_accuracy_criteria(self) -> None:
        input = torch.tensor([[0, 1], [1, 1], [0, 0], [0, 1]])
        target = torch.tensor([[0, 1], [1, 0], [0, 0], [1, 1]])
        # test overlap criteria
        overlap_accuracy = MultilabelAccuracy(criteria="overlap")
        overlap_accuracy.update(input, target)
        result = overlap_accuracy.compute()
        self.assertEqual(result, torch.tensor(1))
        # test overlap criteria with no positive inputs: input[2]
        input_overlap_no_pos = torch.tensor(
            [[0, 0, 1], [0, 0, 0], [0, 0, 0], [1, 0, 0]]
        )
        target_overlap_no_pos = torch.tensor(
            [[0, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 1]]
        )
        overlap_accuracy = MultilabelAccuracy(criteria="overlap")
        overlap_accuracy.update(input_overlap_no_pos, target_overlap_no_pos)
        result = overlap_accuracy.compute()
        self.assertEqual(result, torch.tensor(0.25))
        # test contain criteria
        contain_accuracy = MultilabelAccuracy(criteria="contain")
        contain_accuracy.update(input, target)
        result = contain_accuracy.compute()
        self.assertEqual(result, torch.tensor(0.75))
        # test belong criteria
        belong_accuracy = MultilabelAccuracy(criteria="belong")
        belong_accuracy.update(input, target)
        result = belong_accuracy.compute()
        self.assertEqual(result, torch.tensor(0.75))

    def test_accuracy_class_invalid_intialization(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "`criteria` was not in the allowed value of .*, got weighted."
        ):
            MultilabelAccuracy(criteria="weighted")

    def test_accuracy_class_invalid_input(self) -> None:
        metric = MultilabelAccuracy()
        with self.assertRaisesRegex(
            ValueError,
            "The `input` and `target` should have the same dimensions, "
            r"got shapes torch.Size\(\[4, 2\]\) and torch.Size\(\[3\]\).",
        ):
            metric.update(torch.rand(4, 2), torch.rand(3))


class TestTopKAccuracy(MetricClassTester):
    def test_accuracy_class_base(self) -> None:
        input = torch.tensor(
            [
                [
                    [0.1, 0.2, 0.3, 0.4],
                    [0.1, 0.2, 0.3, 0.4],
                ],
                [
                    [0.1, 0.2, 0.3, 0.4],
                    [0.1, 0.2, 0.3, 0.4],
                ],
                [
                    [0.1, 0.2, 0.3, 0.4],
                    [0.1, 0.2, 0.3, 0.4],
                ],
                [
                    [0.1, 0.2, 0.3, 0.4],
                    [0.1, 0.2, 0.3, 0.4],
                ],
            ]
        )
        target = torch.tensor([[0, 1], [2, 3], [0, 1], [2, 3]])
        compute_result = torch.tensor(0.5)
        self.run_class_implementation_tests(
            metric=MulticlassAccuracy(k=2),
            state_names={"num_correct", "num_total"},
            update_kwargs={"input": input, "target": target},
            compute_result=compute_result,
            num_total_updates=4,
        )

    def test_accuracy_class_macro(self) -> None:
        input = torch.tensor(
            [
                [
                    [0.9, 0.1, 0, 0],
                    [0.9, 0.1, 0, 0],
                ],
                [
                    [0.3, 0.2, 0.4, 0.1],
                    [0.3, 0.2, 0.4, 0.1],
                ],
                [
                    [0.3, 0.2, 0.4, 0.1],
                    [0.3, 0.2, 0.4, 0.1],
                ],
                [
                    [0.4, 0.4, 0.1, 0.1],
                    [0.3, 0.5, 0.1, 0.1],
                ],
            ]
        )
        target = torch.tensor([[0, 0], [0, 0], [0, 0], [2, 2]])
        compute_result = torch.tensor(0.5)
        self.run_class_implementation_tests(
            metric=MulticlassAccuracy(k=2, average="macro", num_classes=4),
            state_names={"num_correct", "num_total"},
            update_kwargs={"input": input, "target": target},
            compute_result=compute_result,
            num_total_updates=4,
        )

    def test_accuracy_class_no_average(self) -> None:
        input = torch.tensor(
            [
                [
                    [0.9, 0.1, 0, 0],
                    [0.9, 0.1, 0, 0],
                ],
                [
                    [0.3, 0.2, 0.4, 0.1],
                    [0.3, 0.2, 0.4, 0.1],
                ],
                [
                    [0.3, 0.2, 0.4, 0.1],
                    [0.3, 0.2, 0.4, 0.1],
                ],
                [
                    [0.4, 0.4, 0.1, 0.1],
                    [0.3, 0.5, 0.1, 0.1],
                ],
            ]
        )
        target = torch.tensor([[0, 0], [0, 0], [0, 0], [2, 2]])
        compute_result = torch.tensor([1.0, np.NAN, 0, np.NAN])
        self.run_class_implementation_tests(
            metric=MulticlassAccuracy(k=2, average=None, num_classes=4),
            state_names={"num_correct", "num_total"},
            update_kwargs={"input": input, "target": target},
            compute_result=compute_result,
            num_total_updates=4,
        )

    def test_topk_accuracy_invalid_params(self) -> None:
        k = -2
        with self.assertRaisesRegex(
            ValueError,
            f"Expected `k` to be an integer greater than 0, but {k} was provided.",
        ):
            MulticlassAccuracy(k=k)

        input = torch.rand(2)
        metric = MulticlassAccuracy(k=2)
        with self.assertRaisesRegex(
            ValueError,
            r"input should have shape \(num_sample, num_classes\) for k > 1, "
            r"got shape torch.Size\(\[2\]\).",
        ):
            metric.update(input, torch.rand(2))


class TestTopKMultilabelAccuracy(MetricClassTester):
    def test_accuracy_label_base(self) -> None:
        input = torch.tensor(
            [[0.1, 0.5, 0.2], [0.3, 0.2, 0.1], [0.2, 0.4, 0.5], [0, 0.1, 0.9]]
        )
        target = torch.tensor([[1, 1, 0], [0, 1, 0], [1, 1, 1], [0, 1, 0]])
        accuracy = TopKMultilabelAccuracy(k=2)
        accuracy.update(input, target)
        result = accuracy.compute()
        self.assertEqual(result, torch.tensor(0.0))

    def test_accuracy_label_hamming(self) -> None:
        input = torch.tensor(
            [[0.1, 0.5, 0.2], [0.3, 0.2, 0.1], [0.2, 0.4, 0.5], [0, 0.1, 0.9]]
        )
        target = torch.tensor([[1, 1, 0], [0, 1, 0], [1, 1, 1], [0, 1, 0]])
        hamming_accuracy = TopKMultilabelAccuracy(k=2, criteria="hamming")
        hamming_accuracy.update(input, target)
        result = hamming_accuracy.compute()
        self.assertEqual(result, torch.tensor(7 / 12))

    def test_accuracy_label_overlap(self) -> None:
        input = torch.tensor(
            [[0.1, 0.5, 0.2], [0.3, 0.2, 0.1], [0.2, 0.4, 0.5], [0, 0.1, 0.9]]
        )
        target = torch.tensor([[1, 1, 0], [0, 1, 0], [1, 1, 1], [0, 1, 0]])
        overlap_accuracy = TopKMultilabelAccuracy(k=2, criteria="overlap")
        overlap_accuracy.update(input, target)
        result = overlap_accuracy.compute()
        self.assertEqual(result, torch.tensor(4 / 4))

    def test_accuracy_label_contain(self) -> None:
        input = torch.tensor(
            [[0.1, 0.5, 0.2], [0.3, 0.2, 0.1], [0.2, 0.4, 0.5], [0, 0.1, 0.9]]
        )
        target = torch.tensor([[1, 1, 0], [0, 1, 0], [1, 1, 1], [0, 1, 0]])
        contain_accuracy = TopKMultilabelAccuracy(k=2, criteria="contain")
        contain_accuracy.update(input, target)
        result = contain_accuracy.compute()
        self.assertEqual(result, torch.tensor(2 / 4))

    def test_accuracy_label_belong(self) -> None:
        input = torch.tensor(
            [[0.1, 0.5, 0.2], [0.3, 0.2, 0.1], [0.2, 0.4, 0.5], [0, 0.1, 0.9]]
        )
        target = torch.tensor([[1, 1, 0], [0, 1, 0], [1, 1, 1], [0, 1, 0]])
        belong_accuracy = TopKMultilabelAccuracy(k=2, criteria="belong")
        belong_accuracy.update(input, target)
        result = belong_accuracy.compute()
        self.assertEqual(result, torch.tensor(1 / 4))

    def test_topk_accuracy_invalid_params(self) -> None:
        k = -2
        with self.assertRaisesRegex(
            ValueError,
            f"Expected `k` to be an integer greater than 1, but {k} was provided.",
        ):
            TopKMultilabelAccuracy(k=k)

        input = torch.rand(2)
        metric = TopKMultilabelAccuracy(k=2)
        with self.assertRaisesRegex(
            ValueError,
            r"input should have shape \(num_sample, num_classes\) for k > 1, "
            r"got shape torch.Size\(\[2\]\).",
        ):
            metric.update(input, torch.rand(2))
