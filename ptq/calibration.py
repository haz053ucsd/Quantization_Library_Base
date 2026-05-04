from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Union

import torch
import torch.nn as nn


@dataclass(frozen=True)
class LayerInputs:
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


_SINGLE_CAPTURE_KEY = "__single_capture__"


def detach_tree(value: Any, device: Optional[Union[torch.device, str]] = None) -> Any:
    if torch.is_tensor(value):
        target_device = "cpu" if device is None else device
        return value.detach().to(target_device, copy=True)
    if isinstance(value, tuple):
        return tuple(detach_tree(item, device=device) for item in value)
    if isinstance(value, list):
        return [detach_tree(item, device=device) for item in value]
    if isinstance(value, dict):
        return {key: detach_tree(item, device=device) for key, item in value.items()}
    return value


def move_batch_to_device(batch: Any, device: Optional[torch.device]) -> Any:
    if device is None:
        return batch
    if torch.is_tensor(batch):
        return batch.to(device)
    if isinstance(batch, tuple):
        return tuple(move_batch_to_device(item, device) for item in batch)
    if isinstance(batch, list):
        return [move_batch_to_device(item, device) for item in batch]
    if isinstance(batch, dict):
        return {key: move_batch_to_device(item, device) for key, item in batch.items()}
    return batch


def forward_batch(model: nn.Module, batch: Any) -> Any:
    if isinstance(batch, dict):
        return model(**batch)
    if isinstance(batch, tuple):
        return model(*batch)
    return model(batch)


def forward_layer(layer: nn.Module, layer_input: LayerInputs) -> Any:
    return layer(*layer_input.args, **layer_input.kwargs)


def first_hidden_state(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, tuple) and output and torch.is_tensor(output[0]):
        return output[0]
    if isinstance(output, dict):
        hidden_states = output.get("hidden_states")
        if torch.is_tensor(hidden_states):
            return hidden_states
        last_hidden_state = output.get("last_hidden_state")
        if torch.is_tensor(last_hidden_state):
            return last_hidden_state
    if hasattr(output, "last_hidden_state") and torch.is_tensor(output.last_hidden_state):
        return output.last_hidden_state
    raise ValueError(f"Could not extract hidden states from layer output type {type(output).__name__}.")


def replace_first_hidden_state(layer_input: LayerInputs, hidden_state: torch.Tensor) -> LayerInputs:
    if layer_input.args:
        return LayerInputs(
            args=(hidden_state, *layer_input.args[1:]),
            kwargs=dict(layer_input.kwargs),
        )
    if "hidden_states" in layer_input.kwargs:
        kwargs = dict(layer_input.kwargs)
        kwargs["hidden_states"] = hidden_state
        return LayerInputs(args=(), kwargs=kwargs)
    raise ValueError("Expected decoder layer inputs to carry hidden states as args[0] or kwargs['hidden_states'].")


class CalibrationInputCache:
    """Captures layer pre-hook inputs for data-dependent layerwise PTQ."""

    def __init__(
        self,
        max_batches: Optional[int] = None,
        storage_device: Optional[Union[torch.device, str]] = "cpu",
    ) -> None:
        self._max_batches = max_batches
        self._storage_device = storage_device

    @torch.no_grad()
    def collect(
        self,
        model: nn.Module,
        layer: nn.Module,
        calibration_batches: Iterable[Any],
        device: Optional[torch.device] = None,
    ) -> list[LayerInputs]:
        return self.collect_many(
            model=model,
            modules={_SINGLE_CAPTURE_KEY: layer},
            calibration_batches=calibration_batches,
            device=device,
        )[_SINGLE_CAPTURE_KEY]

    @torch.no_grad()
    def collect_many(
        self,
        model: nn.Module,
        modules: Mapping[str, nn.Module],
        calibration_batches: Iterable[Any],
        device: Optional[torch.device] = None,
    ) -> dict[str, list[LayerInputs]]:
        model.eval()
        captured, handles = self._register_hooks(modules)

        try:
            for batch_index, batch in enumerate(calibration_batches):
                if self._max_batches is not None and batch_index >= self._max_batches:
                    break
                forward_batch(model, move_batch_to_device(batch, device))
        finally:
            self._remove_handles(handles)

        return captured

    @torch.no_grad()
    def collect_from_layer(
        self,
        layer: nn.Module,
        target_module: nn.Module,
        layer_inputs: Iterable[LayerInputs],
        device: Optional[torch.device] = None,
    ) -> list[LayerInputs]:
        return self.collect_many_from_layer(
            layer=layer,
            modules={_SINGLE_CAPTURE_KEY: target_module},
            layer_inputs=layer_inputs,
            device=device,
        )[_SINGLE_CAPTURE_KEY]

    @torch.no_grad()
    def collect_many_from_layer(
        self,
        layer: nn.Module,
        modules: Mapping[str, nn.Module],
        layer_inputs: Iterable[LayerInputs],
        device: Optional[torch.device] = None,
    ) -> dict[str, list[LayerInputs]]:
        layer.eval()
        captured, handles = self._register_hooks(modules)

        try:
            for batch_index, layer_input in enumerate(layer_inputs):
                if self._max_batches is not None and batch_index >= self._max_batches:
                    break
                forward_layer(
                    layer,
                    LayerInputs(
                        args=move_batch_to_device(layer_input.args, device),
                        kwargs=move_batch_to_device(layer_input.kwargs, device),
                    ),
                )
        finally:
            self._remove_handles(handles)

        return captured

    @torch.no_grad()
    def advance_layer_inputs(
        self,
        layer: nn.Module,
        layer_inputs: Iterable[LayerInputs],
        device: Optional[torch.device] = None,
    ) -> list[LayerInputs]:
        layer.eval()
        advanced: list[LayerInputs] = []

        for batch_index, layer_input in enumerate(layer_inputs):
            if self._max_batches is not None and batch_index >= self._max_batches:
                break
            moved_input = LayerInputs(
                args=move_batch_to_device(layer_input.args, device),
                kwargs=move_batch_to_device(layer_input.kwargs, device),
            )
            output = forward_layer(layer, moved_input)
            hidden_state = detach_tree(first_hidden_state(output), device=self._storage_device)
            advanced.append(replace_first_hidden_state(layer_input, hidden_state))

        return advanced

    def _register_hooks(
        self,
        modules: Mapping[str, nn.Module],
    ) -> tuple[dict[str, list[LayerInputs]], list[torch.utils.hooks.RemovableHandle]]:
        captured: dict[str, list[LayerInputs]] = {name: [] for name in modules}
        handles: list[torch.utils.hooks.RemovableHandle] = []

        def make_hook(name: str):
            def hook(_module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
                captured[name].append(
                    LayerInputs(
                        args=detach_tree(args, device=self._storage_device),
                        kwargs=detach_tree(kwargs, device=self._storage_device),
                    )
                )

            return hook

        for name, module in modules.items():
            handles.append(module.register_forward_pre_hook(make_hook(name), with_kwargs=True))

        return captured, handles

    def _remove_handles(self, handles: Iterable[torch.utils.hooks.RemovableHandle]) -> None:
        for handle in handles:
            handle.remove()
