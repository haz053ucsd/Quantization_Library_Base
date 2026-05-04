from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator, Optional, Sequence

import torch.nn as nn

from .activation import ActivationQuantizedLinear


@dataclass(frozen=True)
class LayerSpec:
    name: str
    index: int
    reference_layer: nn.Module
    quant_layer: nn.Module


class LayerNavigator:
    """Finds decoder blocks and quantizable modules in Qwen-like LLMs."""

    DEFAULT_LAYER_PATHS: Sequence[str] = (
        "model.layers",
        "transformer.h",
        "gpt_neox.layers",
        "layers",
    )

    def __init__(
        self,
        decoder_layers_path: Optional[str] = None,
        linear_filter: Optional[Callable[[str, nn.Module], bool]] = None,
    ) -> None:
        self._decoder_layers_path = decoder_layers_path
        self._linear_filter = linear_filter

    def iter_layer_specs(
        self,
        reference_model: nn.Module,
        quant_model: nn.Module,
    ) -> Iterator[LayerSpec]:
        layer_path = self._decoder_layers_path or self._infer_decoder_layers_path(reference_model)
        reference_layers = reference_model.get_submodule(layer_path)
        quant_layers = quant_model.get_submodule(layer_path)

        if len(reference_layers) != len(quant_layers):
            raise ValueError(
                f"Reference and quant models have different layer counts at {layer_path}: "
                f"{len(reference_layers)} != {len(quant_layers)}"
            )

        for index, (reference_layer, quant_layer) in enumerate(zip(reference_layers, quant_layers)):
            yield LayerSpec(
                name=f"{layer_path}.{index}",
                index=index,
                reference_layer=reference_layer,
                quant_layer=quant_layer,
            )

    def iter_quantizable_linears(self, layer: nn.Module) -> Iterator[tuple[str, nn.Module]]:
        yield from self._iter_quantizable_linears(layer, prefix="")

    def _iter_quantizable_linears(self, module: nn.Module, prefix: str) -> Iterator[tuple[str, nn.Module]]:
        if prefix and isinstance(module, (nn.Linear, ActivationQuantizedLinear)):
            if self._linear_filter is None or self._linear_filter(prefix, module):
                yield prefix, module
            return

        for child_name, child in module.named_children():
            child_prefix = child_name if not prefix else f"{prefix}.{child_name}"
            yield from self._iter_quantizable_linears(child, child_prefix)

    def _infer_decoder_layers_path(self, model: nn.Module) -> str:
        for candidate in self.DEFAULT_LAYER_PATHS:
            try:
                layers = model.get_submodule(candidate)
            except AttributeError:
                continue
            if isinstance(layers, (nn.ModuleList, list, tuple)) and len(layers) > 0:
                return candidate
        raise ValueError(
            "Could not infer decoder layer path. Pass decoder_layers_path, "
            "for example 'model.layers' for Qwen-like Hugging Face models."
        )
