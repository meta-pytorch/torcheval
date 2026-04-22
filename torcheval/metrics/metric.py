# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable
from copy import deepcopy
from typing import Any, Generic, TypeVar, Union

import torch

# pyre-fixme[24]: Generic type `Metric` expects 1 type parameter.
TSelf = TypeVar("TSelf", bound="Metric")
TComputeReturn = TypeVar("TComputeReturn")
TState = Union[torch.Tensor, list[torch.Tensor], dict[Any, torch.Tensor], int, float]


class Metric(Generic[TComputeReturn], ABC):
    """
    Base class for all metrics present in the Metrics API.

    Implement __init__(), update(), compute(), merge_state() functions
    to implement your own metric.
    """

    def __init__(
        self: TSelf,
        *,
        device: torch.device | None = None,
    ) -> None:
        """
        Initialize a metric object and its internal states.

        Use ``self._add_state()`` to initialize state variables of your metric class.
        The state variables should be either ``torch.Tensor``, a list of
        ``torch.Tensor``, or a dictionary with ``torch.Tensor`` as values
        """
        torch._C._log_api_usage_once(f"torcheval.metrics.{self.__class__.__name__}")

        # limit state variable type to tensor/[tensor] to avoid working with nested
        # data structures when move/detach/clone tensors. Can open more types up
        # upon user requests in the future.
        self._state_name_to_default: dict[str, TState] = {}
        self._device: torch.device = torch.device("cpu") if device is None else device

    def _add_state(self: TSelf, name: str, default: TState) -> None:
        """
        Used in subclass ``__init__()`` to add a metric state variable.

        Args:
            name: The name of the state variable. The variable can be accessed
                with ``self.name``.
            default: Default value of the state. It should be a type of TState.
                The state will be reset to this value when ``self.reset()`` is called.
        Raises:
            TypeError: If ``default`` is not a type of TState.
        """
        _check_state_variable_type(name, default)
        # deepcopy makes sure the input/initial value/default value of the state
        # variable are independent.
        setattr(self, name, deepcopy(default))
        self._state_name_to_default[name] = deepcopy(default)

    @abstractmethod
    @torch.inference_mode()
    def update(self: TSelf, *_: Any, **__: Any) -> TSelf:
        """
        Implement this method to update the state variables of your metric class.

        Decorate update() with @torch.inference_mode() which gives better
        performance by disabling view tracking.
        """

    @abstractmethod
    @torch.inference_mode()
    def compute(self: TSelf) -> TComputeReturn:
        """
        Implement this method to compute and return the final metric value
        from state variables.

        Decorate compute() with @torch.inference_mode() which gives better
        performance by disabling view tracking.
        """

    @abstractmethod
    @torch.inference_mode()
    def merge_state(self: TSelf, metrics: Iterable[TSelf]) -> TSelf:
        """
        Implement this method to update the current metric's state variables
        to be the merged states of the current metric and input metrics. The state
        variables of input metrics should stay unchanged.

        Decorate merge_state() with @torch.inference_mode() which gives better
        performance by disabling view tracking.

        ``self.merge_state`` might change the size/shape of state variables.
        Make sure ``self.update`` and ``self.compute`` can still be called
        without exceptions when state variables are merged.

        This method can be used as a building block for syncing metric states
        in distributed training. For example, ``sync_and_compute`` in the metric
        toolkit will use this method to merge metric objects gathered from the
        process group.
        """

    @torch.inference_mode()
    def _prepare_for_merge_state(self: TSelf) -> None:
        """
        Called before syncing metrics in ``toolkit._sync_metric_object()``.

        It can be utilized to adjust metric states to accelerate syncing.
        For example, concatenated metric state from a list of tensors to
        one tensor. See ``torcheval.metrics.BinaryAUROC`` as an example.
        """
        pass

    def reset(self: TSelf) -> TSelf:
        """
        Reset the metric state variables to their default value.
        The tensors in the default values are also moved to the device of
        the last ``self.to(device)`` call.
        """
        for state_name, default in self._state_name_to_default.items():
            if isinstance(default, torch.Tensor):
                setattr(self, state_name, default.clone().to(self.device))
            elif isinstance(default, list):
                setattr(
                    self,
                    state_name,
                    [tensor.clone().to(self.device) for tensor in default],
                )
            elif isinstance(default, dict):
                setattr(
                    self,
                    state_name,
                    defaultdict(
                        lambda: torch.tensor(0.0, device=self.device),
                        {
                            key: tensor.clone().to(self.device)
                            for key, tensor in default.items()
                        },
                    ),
                )
            elif isinstance(default, (int, float)):
                setattr(self, state_name, default)
            else:
                raise TypeError(
                    f"Invalid type for default value for {state_name}. Received {type(default)}, but expected ``torch.Tensor``, a list of ``torch.Tensor``,"
                    f"a dictionary with ``torch.Tensor``, int, or float."
                )
        return self

    def state_dict(self: TSelf) -> dict[str, TState]:
        """
        Save metric state variables in state_dict.

        Raises:
            TypeError: If ``default`` is not a type of TState.
        """
        state_dict = {}
        for state_name in self._state_name_to_default:
            value = getattr(self, state_name)
            _check_state_variable_type(state_name, value)

            if isinstance(value, torch.Tensor):
                state_dict[state_name] = value.detach().clone()
            elif isinstance(value, list):
                # pyrefly: ignore [unsupported-operation]
                state_dict[state_name] = [tensor.detach().clone() for tensor in value]
            elif isinstance(value, dict):
                # pyrefly: ignore [unsupported-operation]
                state_dict[state_name] = {
                    key: tensor.detach().clone() for key, tensor in value.items()
                }
            elif isinstance(value, int):
                # pyrefly: ignore [unsupported-operation]
                state_dict[state_name] = value
            elif isinstance(value, float):
                # pyrefly: ignore [unsupported-operation]
                state_dict[state_name] = value
        # pyrefly: ignore [bad-return]
        return state_dict

    def load_state_dict(
        self: TSelf,
        state_dict: dict[str, Any],
        strict: bool = True,
    ) -> None:
        """
        Loads metric state variables from state_dict.

        Args:
            state_dict (Dict[str, Any]): A dict containing metric state variables.
            strict (bool, Optional): Whether to strictly enforce that the keys in ``state_dict`` matches
                all names of the metric states.

        Raises:
            RuntimeError: If ``strict`` is ``True`` and keys in state_dict does not match
                all names of the metric states.
            TypeError: If ``default`` is not a type of TState.
        """
        state_dict = deepcopy(state_dict)
        metric_state_names = set(self._state_name_to_default.keys())
        for state_name in metric_state_names:
            if state_name in state_dict:
                value = state_dict[state_name]
                _check_state_variable_type(state_name, value)
                setattr(self, state_name, value)

        if strict:
            state_dict_keys = set(state_dict.keys())
            unexpected_keys = state_dict_keys.difference(metric_state_names)
            missing_keys = metric_state_names.difference(state_dict_keys)
            if missing_keys or unexpected_keys:
                raise RuntimeError(
                    f"Error(s) in loading state_dict for {self.__class__.__name__}. "
                    f"Encountered missing keys: {missing_keys} and unexpected "
                    f"keys: {unexpected_keys}."
                )

    def to(self: TSelf, device: str | torch.device, *args: Any, **kwargs: Any) -> TSelf:
        """
        Move tensors in metric state variables to device.

        Args:
            device: The destination device.
        Raises:
            TypeError: If ``default`` is not a type of TState.
        """
        device = torch.device(device) if isinstance(device, str) else device
        for state_name in self._state_name_to_default:
            value = getattr(self, state_name)
            _check_state_variable_type(state_name, value)
            if isinstance(value, torch.Tensor):
                setattr(self, state_name, value.to(device))
            elif isinstance(value, list):
                setattr(
                    self,
                    state_name,
                    [tensor.to(device, *args, **kwargs) for tensor in value],
                )
            elif isinstance(value, dict):
                setattr(
                    self,
                    state_name,
                    defaultdict(
                        lambda: torch.tensor(0.0, device=device),
                        {
                            key: tensor.to(device, *args, **kwargs)
                            for key, tensor in value.items()
                        },
                    ),
                )
        self._device = device
        return self

    @property
    def device(self: TSelf) -> torch.device:
        """
        The last input device of ``Metric.to()``.
        Default to ``torch.device("cpu")`` if ``Metric.to()`` is not called.
        """
        return self._device


def _check_state_variable_type(name: str, value: Any) -> None:
    """
    Check the type of a state variable value.
    It should be a type of TState.
    """
    if (
        not isinstance(value, torch.Tensor)
        and not (
            isinstance(value, list) and all(isinstance(x, torch.Tensor) for x in value)
        )
        and not (
            isinstance(value, dict)
            and all(isinstance(x, torch.Tensor) for x in value.values())
        )
        and not isinstance(value, int)
        and not isinstance(value, float)
    ):
        raise TypeError(
            "The value of state variable must be a ``torch.Tensor``, a list of ``torch.Tensor``, "
            f"a dictionary with ``torch.Tensor``, int, or float as values."
            f"Got {name}={value} instead."
        )
