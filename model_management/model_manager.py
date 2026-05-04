from __future__ import annotations
from typing import Any, Dict, Optional
from .base_model import BaseModelWrapper
from .reference_model import ReferenceModel
from .quantized_model import QuantModel
import copy
import torch


class ModelManager:
    def __init__(
        self,
        reference_model: ReferenceModel,
        quant_model: Optional[QuantModel] = None,
    ) -> None:
        self._reference_model = reference_model
        self._quant_model = quant_model

    @property
    def reference_model(self) -> ReferenceModel:
        return self._reference_model

    @property
    def quant_model(self) -> Optional[QuantModel]:
        return self._quant_model

    def has_quant_model(self) -> bool:
        return self._quant_model is not None

    def build_quant_model_from_reference(
        self,
        name: str = "quant_model",
        quant_config: Optional[Dict[str, Any]] = None,
    ) -> QuantModel:
        quantized_copy = copy.deepcopy(self._reference_model.model)
        self._quant_model = QuantModel(
            model=quantized_copy,
            name=name,
            device=self._reference_model.device,
            quant_config=quant_config,
        )
        return self._quant_model

    def get_model(self, kind: str = "reference") -> BaseModelWrapper:
        if kind == "reference":
            return self._reference_model
        if kind == "quant":
            if self._quant_model is None:
                raise ValueError("Quant model has not been built yet.")
            return self._quant_model
        raise ValueError(f"Unknown model kind: {kind}")

    def to(self, device: torch.device) -> None:
        self._reference_model.to(device)
        if self._quant_model is not None:
            self._quant_model.to(device)

    def eval(self) -> None:
        self._reference_model.eval()
        if self._quant_model is not None:
            self._quant_model.eval()

    def train(self) -> None:
        self._reference_model.train()
        if self._quant_model is not None:
            self._quant_model.train()

    def summary(self) -> Dict[str, Any]:
        return {
            "reference_name": self._reference_model.name,
            "reference_device": str(self._reference_model.device),
            "has_quant_model": self._quant_model is not None,
            "quant_name": None if self._quant_model is None else self._quant_model.name,
            "quant_device": None if self._quant_model is None else str(self._quant_model.device),
            "is_quantized": None if self._quant_model is None else self._quant_model.is_quantized,
        }
