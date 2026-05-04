from __future__ import annotations
from typing import Any, Dict, Optional
from .base_model import BaseModelWrapper
import copy
import torch
import torch.nn as nn


class QuantModel(BaseModelWrapper):
    def __init__(
        self,
        model: nn.Module,
        name: str = "quant_model",
        device: Optional[torch.device] = None,
        quant_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(model=model, name=name, device=device)
        self._quant_config = quant_config or {}
        self._is_quantized = False
        self._quantized_layers: Dict[str, nn.Module] = {}

    @property
    def quant_config(self) -> Dict[str, Any]:
        return self._quant_config

    @property
    def is_quantized(self) -> bool:
        return self._is_quantized

    @property
    def quantized_layers(self) -> Dict[str, nn.Module]:
        return self._quantized_layers

    def mark_quantized(self) -> None:
        self._is_quantized = True

    def register_quantized_layer(self, layer_name: str, layer: nn.Module) -> None:
        self._quantized_layers[layer_name] = layer

    def clone(self) -> "QuantModel":
        cloned_model = copy.deepcopy(self._model)
        new_obj = QuantModel(
            model=cloned_model,
            name=self._name,
            device=self._device,
            quant_config=copy.deepcopy(self._quant_config),
        )
        new_obj._is_quantized = self._is_quantized
        for layer_name in self._quantized_layers:
            new_obj.register_quantized_layer(
                layer_name,
                cloned_model.get_submodule(layer_name),
            )
        return new_obj
