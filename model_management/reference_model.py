from __future__ import annotations
from typing import Optional
from .base_model import BaseModelWrapper
import copy
import torch
import torch.nn as nn

class ReferenceModel(BaseModelWrapper):
    def __init__(
        self,
        model: nn.Module,
        name: str = "reference_model",
        device: Optional[torch.device] = None,
        freeze: bool = True
    ) -> None:
        super().__init__(model=model, name=name, device=device)
        if freeze:
            self.freeze()
        self.eval()

    def clone(self) -> "ReferenceModel":
        cloned_model = copy.deepcopy(self._model)
        return ReferenceModel(
            model=cloned_model,
            name=self._name,
            device=self._device,
            freeze=True
        )
