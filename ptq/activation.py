from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ActivationQuantizationConfig:
    num_bits: int = 8
    symmetric: bool = True
    granularity: str = "per-token"
    eps: float = 1e-8


class ActivationFakeQuantizer(nn.Module):
    def __init__(self, config: ActivationQuantizationConfig) -> None:
        super().__init__()
        self.config = config

    def forward(self, activation: torch.Tensor) -> torch.Tensor:
        if not torch.is_floating_point(activation):
            return activation

        original_dtype = activation.dtype
        activation_float = activation.float()
        qmin, qmax = self._quant_bounds()
        reduce_dims = self._reduce_dims(activation_float)

        if self.config.symmetric:
            max_abs = activation_float.abs().amax(dim=reduce_dims, keepdim=True)
            scale = torch.clamp(max_abs / max(qmax, 1), min=self.config.eps)
            quantized = torch.round(activation_float / scale).clamp(qmin, qmax)
            dequantized = quantized * scale
        else:
            min_values = activation_float.amin(dim=reduce_dims, keepdim=True)
            max_values = activation_float.amax(dim=reduce_dims, keepdim=True)
            scale = torch.clamp((max_values - min_values) / max(qmax - qmin, 1), min=self.config.eps)
            zero = torch.clamp(torch.round(qmin - min_values / scale), qmin, qmax)
            quantized = torch.round(activation_float / scale + zero).clamp(qmin, qmax)
            dequantized = (quantized - zero) * scale

        return dequantized.to(dtype=original_dtype)

    def _quant_bounds(self) -> tuple[int, int]:
        if self.config.num_bits < 2:
            raise ValueError(f"activation num_bits must be at least 2, got {self.config.num_bits}")
        if self.config.symmetric:
            return -(2 ** (self.config.num_bits - 1)), (2 ** (self.config.num_bits - 1)) - 1
        return 0, (2**self.config.num_bits) - 1

    def _reduce_dims(self, activation: torch.Tensor) -> tuple[int, ...]:
        if self.config.granularity == "per-tensor":
            return tuple(range(activation.ndim))
        if self.config.granularity == "per-token":
            return (activation.ndim - 1,)
        raise ValueError(
            "Unknown activation quantization granularity: "
            f"{self.config.granularity}. Expected 'per-token' or 'per-tensor'."
        )


class ActivationQuantizedLinear(nn.Module):
    def __init__(self, linear: nn.Linear, quantizer: ActivationFakeQuantizer) -> None:
        super().__init__()
        self.activation_quantizer = quantizer
        self.linear = linear
        self.in_features = linear.in_features
        self.out_features = linear.out_features

    @property
    def weight(self) -> torch.nn.Parameter:
        return self.linear.weight

    @property
    def bias(self) -> torch.nn.Parameter | None:
        return self.linear.bias

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.linear(self.activation_quantizer(input))


def wrap_linear_with_activation_quantization(
    linear: nn.Linear,
    config: ActivationQuantizationConfig,
) -> ActivationQuantizedLinear:
    return ActivationQuantizedLinear(linear=linear, quantizer=ActivationFakeQuantizer(config))


def activation_calibration_module(module: nn.Module) -> nn.Module:
    if isinstance(module, ActivationQuantizedLinear):
        return module.linear
    return module
