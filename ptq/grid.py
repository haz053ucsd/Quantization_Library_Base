from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Sequence

import torch


@dataclass(frozen=True)
class QuantizationParameters:
    scales: torch.Tensor
    zeros: torch.Tensor
    qmin: int
    qmax: int
    symmetric: bool


@dataclass(frozen=True)
class GroupedQuantizationParameters(QuantizationParameters):
    original_shape: torch.Size
    padded_features: int
    group_size: int


class GroupedTensorLayout:
    def __init__(self, group_size: int = 128) -> None:
        self.group_size = group_size

    def reshape(self, tensor: torch.Tensor) -> tuple[torch.Tensor, int]:
        if tensor.dim() != 2:
            raise ValueError(f"Expected a 2D tensor, got shape {tuple(tensor.shape)}")
        rows, features = tensor.shape
        groups = (features + self.group_size - 1) // self.group_size
        padded_features = groups * self.group_size
        if padded_features != features:
            pad = tensor.new_zeros(rows, padded_features - features)
            tensor = torch.cat([tensor, pad], dim=1)
        return tensor.reshape(rows, groups, self.group_size), padded_features

    def restore(self, grouped: torch.Tensor, original_shape: torch.Size, padded_features: int) -> torch.Tensor:
        restored = grouped.reshape(original_shape[0], padded_features)
        return restored[:, : original_shape[1]]


class GridSelector(ABC):
    def __init__(
        self,
        num_bits: int = 4,
        symmetric: bool = True,
        eps: float = 1e-8,
    ) -> None:
        self.num_bits = num_bits
        self.symmetric = symmetric
        self.eps = eps

    @abstractmethod
    def select(
        self,
        tensor: torch.Tensor,
        importance: Optional[torch.Tensor] = None,
    ) -> QuantizationParameters:
        pass

    def quant_bounds(self) -> tuple[int, int]:
        if self.symmetric:
            return -(2 ** (self.num_bits - 1)), (2 ** (self.num_bits - 1)) - 1
        return 0, (2**self.num_bits) - 1

    def importance_weights(
        self,
        tensor: torch.Tensor,
        importance: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if importance is None:
            return torch.ones((1, tensor.shape[1]), device=tensor.device, dtype=tensor.dtype)
        if importance.dim() != 1 or importance.numel() != tensor.shape[1]:
            raise ValueError(
                f"importance must be 1D with length {tensor.shape[1]}, got shape {tuple(importance.shape)}"
            )
        return importance.to(device=tensor.device, dtype=tensor.dtype).reshape(1, -1)


class MinMaxGridSelector(GridSelector):
    def select(
        self,
        tensor: torch.Tensor,
        importance: Optional[torch.Tensor] = None,
    ) -> QuantizationParameters:
        _ = importance
        if tensor.dim() != 2:
            raise ValueError(f"Expected a 2D tensor, got shape {tuple(tensor.shape)}")

        qmin, qmax = self.quant_bounds()
        if self.symmetric:
            max_abs = tensor.abs().amax(dim=1, keepdim=True)
            scales = torch.clamp(max_abs / max(qmax, 1), min=self.eps)
            zeros = torch.zeros_like(scales)
        else:
            min_values = tensor.amin(dim=1, keepdim=True)
            max_values = tensor.amax(dim=1, keepdim=True)
            scales = torch.clamp((max_values - min_values) / max(qmax - qmin, 1), min=self.eps)
            zeros = torch.clamp(torch.round(qmin - min_values / scales), qmin, qmax)

        return QuantizationParameters(
            scales=scales,
            zeros=zeros,
            qmin=qmin,
            qmax=qmax,
            symmetric=self.symmetric,
        )


class ClipSearchGridSelector(GridSelector):
    def __init__(
        self,
        num_bits: int = 4,
        symmetric: bool = True,
        clip_ratios: Optional[Sequence[float]] = None,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(num_bits=num_bits, symmetric=symmetric, eps=eps)
        self.clip_ratios = tuple(clip_ratios or (1.0, 0.99, 0.98, 0.97, 0.96, 0.95, 0.9, 0.85, 0.8))

    def select(
        self,
        tensor: torch.Tensor,
        importance: Optional[torch.Tensor] = None,
    ) -> QuantizationParameters:
        if tensor.dim() != 2:
            raise ValueError(f"Expected a 2D tensor, got shape {tuple(tensor.shape)}")

        qmin, qmax = self.quant_bounds()
        weights = self.importance_weights(tensor, importance)
        quantizer = TensorQuantizer()
        best_losses = torch.full(
            (tensor.shape[0], 1),
            float("inf"),
            device=tensor.device,
            dtype=tensor.dtype,
        )
        best_scales = torch.empty_like(best_losses)
        best_zeros = torch.empty_like(best_losses)

        for clip_ratio in self.clip_ratios:
            candidate = self._parameters_for_clip_ratio(tensor, clip_ratio, qmin, qmax)
            dequantized = quantizer.quantize_dequantize(tensor, candidate)
            losses = ((dequantized - tensor).pow(2) * weights).sum(dim=1, keepdim=True)
            update_mask = losses < best_losses

            best_losses = torch.where(update_mask, losses, best_losses)
            best_scales = torch.where(update_mask, candidate.scales, best_scales)
            best_zeros = torch.where(update_mask, candidate.zeros, best_zeros)

        return QuantizationParameters(
            scales=best_scales,
            zeros=best_zeros,
            qmin=qmin,
            qmax=qmax,
            symmetric=self.symmetric,
        )

    def _parameters_for_clip_ratio(
        self,
        tensor: torch.Tensor,
        clip_ratio: float,
        qmin: int,
        qmax: int,
    ) -> QuantizationParameters:
        if self.symmetric:
            max_abs = tensor.abs().amax(dim=1, keepdim=True) * clip_ratio
            scales = torch.clamp(max_abs / max(qmax, 1), min=self.eps)
            zeros = torch.zeros_like(scales)
        else:
            min_values = tensor.amin(dim=1, keepdim=True) * clip_ratio
            max_values = tensor.amax(dim=1, keepdim=True) * clip_ratio
            scales = torch.clamp((max_values - min_values) / max(qmax - qmin, 1), min=self.eps)
            zeros = torch.clamp(torch.round(qmin - min_values / scales), qmin, qmax)

        return QuantizationParameters(
            scales=scales,
            zeros=zeros,
            qmin=qmin,
            qmax=qmax,
            symmetric=self.symmetric,
        )


class TensorQuantizer:
    def quantize_dequantize(
        self,
        tensor: torch.Tensor,
        parameters: QuantizationParameters,
    ) -> torch.Tensor:
        if parameters.symmetric:
            quantized = torch.round(tensor / parameters.scales).clamp(parameters.qmin, parameters.qmax)
            return quantized * parameters.scales

        quantized = torch.round(tensor / parameters.scales + parameters.zeros).clamp(parameters.qmin, parameters.qmax)
        return (quantized - parameters.zeros) * parameters.scales


class GroupedTensorQuantizer:
    def __init__(
        self,
        layout: Optional[GroupedTensorLayout] = None,
        tensor_quantizer: Optional[TensorQuantizer] = None,
    ) -> None:
        self.layout = layout or GroupedTensorLayout()
        self.tensor_quantizer = tensor_quantizer or TensorQuantizer()

    def quantize_dequantize(
        self,
        tensor: torch.Tensor,
        parameters: GroupedQuantizationParameters,
    ) -> torch.Tensor:
        grouped, padded_features = self.layout.reshape(tensor)
        if padded_features != parameters.padded_features:
            raise ValueError(
                f"Grouped tensor padded width {padded_features} does not match parameters {parameters.padded_features}"
            )
        grouped_dequantized = self.tensor_quantizer.quantize_dequantize(grouped, parameters)
        return self.layout.restore(grouped_dequantized, parameters.original_shape, parameters.padded_features)


def select_grouped_quantization_parameters(
    tensor: torch.Tensor,
    selector: GridSelector,
    layout: GroupedTensorLayout,
) -> GroupedQuantizationParameters:
    grouped, padded_features = layout.reshape(tensor)
    flat_groups = grouped.reshape(-1, layout.group_size)
    flat_parameters = selector.select(flat_groups)
    return GroupedQuantizationParameters(
        scales=flat_parameters.scales.reshape(grouped.shape[0], grouped.shape[1], 1),
        zeros=flat_parameters.zeros.reshape(grouped.shape[0], grouped.shape[1], 1),
        qmin=flat_parameters.qmin,
        qmax=flat_parameters.qmax,
        symmetric=flat_parameters.symmetric,
        original_shape=tensor.shape,
        padded_features=padded_features,
        group_size=layout.group_size,
    )
