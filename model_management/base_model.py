from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, Optional, Tuple
import copy
import torch
import torch.nn as nn

class BaseModelWrapper(ABC):
    def __init__(self, model: nn.Module, name: str, device: Optional[torch.device] = None):
        self._model = model.to(device) if device else model
        self._name = name
        self._device = device

    @property
    def model(self) -> nn.Module:
        return self._model

    @property
    def name(self) -> str:
        return self._name

    @property
    def device(self) -> Optional[torch.device]:
        return self._device

    def to(self, device: torch.device) -> "BaseModelWrapper":
        self._model.to(device)
        self._device = device
        return self

    def eval(self) -> "BaseModelWrapper":
        self._model.eval()
        return self

    def train(self) -> "BaseModelWrapper":
        self._model.train()
        return self

    def state_dict(self) -> Dict[str, Any]:
        return self._model.state_dict()

    def named_modules(self) -> Iterator[Tuple[str, nn.Module]]:
        return self._model.named_modules()

    def get_submodule(self, target: str) -> nn.Module:
        return self._model.get_submodule(target)
    

    def freeze(self) -> "BaseModelWrapper":
        for param in self._model.parameters():
            param.requires_grad = False
        return self

    def unfreeze(self) -> "BaseModelWrapper":
        for param in self._model.parameters():
            param.requires_grad = True
        return self

    @abstractmethod
    def clone(self) -> "BaseModelWrapper":
        pass

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self._model(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)