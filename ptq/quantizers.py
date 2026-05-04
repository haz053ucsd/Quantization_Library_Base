from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .calibration import LayerInputs
from .grid import (
    ClipSearchGridSelector,
    GridSelector,
    GroupedTensorLayout,
    GroupedTensorQuantizer,
    MinMaxGridSelector,
    QuantizationParameters,
    TensorQuantizer,
    select_grouped_quantization_parameters,
)


def linear_input_matrix(
    inputs: list[LayerInputs],
    max_batches: Optional[int] = None,
) -> torch.Tensor:
    if not inputs:
        raise ValueError("No calibration inputs were captured.")

    matrices = []
    selected = inputs if max_batches is None else inputs[:max_batches]
    for item in selected:
        if not item.args or not torch.is_tensor(item.args[0]):
            raise ValueError("Expected the first captured Linear input to be a tensor.")
        activation = item.args[0].detach()
        matrices.append(activation.reshape(-1, activation.shape[-1]))

    return torch.cat(matrices, dim=0)


def activation_column_order(inputs: list[LayerInputs]) -> torch.Tensor:
    activation = linear_input_matrix(inputs).float()
    scores = activation.pow(2).mean(dim=0)
    return torch.argsort(scores, descending=True)


def inverse_cholesky_with_jitter(
    hessian: torch.Tensor,
    damping: torch.Tensor,
    damping_multipliers: Sequence[float] = (1.0, 10.0, 100.0, 1000.0),
) -> torch.Tensor:
    diag = torch.arange(hessian.shape[0], device=hessian.device)
    base = 0.5 * (hessian + hessian.T)

    for multiplier in damping_multipliers:
        candidate = base.clone()
        if multiplier:
            candidate[diag, diag] += damping * multiplier
        try:
            inverse = torch.linalg.inv(candidate)
        except torch.linalg.LinAlgError:
            continue
        inverse = 0.5 * (inverse + inverse.T)
        chol, info = torch.linalg.cholesky_ex(inverse, upper=True)
        if torch.all(info == 0):
            return chol

    candidate = base.clone()
    candidate[diag, diag] += damping * damping_multipliers[-1]
    eigenvalues, eigenvectors = torch.linalg.eigh(candidate)
    inverse_eigenvalues = eigenvalues.clamp_min(damping).reciprocal()
    inverse = (eigenvectors * inverse_eigenvalues.unsqueeze(0)) @ eigenvectors.T
    inverse = 0.5 * (inverse + inverse.T)
    chol, info = torch.linalg.cholesky_ex(inverse, upper=True)
    if torch.all(info == 0):
        return chol

    inverse[diag, diag] += damping
    return torch.linalg.cholesky(inverse, upper=True)


def _power_iteration(matrix: torch.Tensor, steps: int, eps: float = 1e-12) -> torch.Tensor:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected a square matrix, got shape {tuple(matrix.shape)}")
    if matrix.numel() == 0:
        return torch.zeros((), device=matrix.device, dtype=matrix.dtype)

    vector = torch.rand(matrix.shape[1], device=matrix.device, dtype=matrix.dtype)
    for _ in range(steps):
        next_vector = torch.mv(matrix, vector)
        next_norm = torch.norm(next_vector)
        vector = next_vector / (next_norm + eps)
    return torch.dot(vector, torch.mv(matrix, vector))


class WeightQuantizationAlgorithm(ABC):
    needs_reference_inputs: bool = False
    needs_quant_inputs: bool = False

    @abstractmethod
    def quantize(
        self,
        layer_name: str,
        module_name: str,
        reference_inputs: list[LayerInputs],
        quant_inputs: list[LayerInputs],
        weight: torch.Tensor,
    ) -> torch.Tensor:
        pass


class MinMaxDataDependentQuantizer(WeightQuantizationAlgorithm):
    """RTN"""

    def __init__(
        self,
        grid_selector: Optional[GridSelector] = None,
        layout: Optional[GroupedTensorLayout] = None,
        grouped_tensor_quantizer: Optional[GroupedTensorQuantizer] = None,
    ) -> None:
        self.grid_selector = grid_selector or MinMaxGridSelector()
        self.layout = layout or GroupedTensorLayout()
        self.grouped_tensor_quantizer = grouped_tensor_quantizer or GroupedTensorQuantizer(layout=self.layout)

    def quantize(
        self,
        layer_name: str,
        module_name: str,
        reference_inputs: list[LayerInputs],
        quant_inputs: list[LayerInputs],
        weight: torch.Tensor,
    ) -> torch.Tensor:
        _ = (layer_name, module_name, reference_inputs, quant_inputs)
        parameters = select_grouped_quantization_parameters(
            tensor=weight,
            selector=self.grid_selector,
            layout=self.layout,
        )
        return self.grouped_tensor_quantizer.quantize_dequantize(weight, parameters)
    

class AWQWeightQuantizer(WeightQuantizationAlgorithm):
    """Local AWQ-style activation-aware weight fake quantization."""

    needs_quant_inputs: bool = True

    def __init__(
        self,
        grid_selector: Optional[GridSelector] = None,
        layout: Optional[GroupedTensorLayout] = None,
        grouped_tensor_quantizer: Optional[GroupedTensorQuantizer] = None,
        max_input_batches: Optional[int] = 1,
        n_grid: int = 20,
        eps: float = 1e-4,
    ) -> None:
        self.grid_selector = grid_selector or MinMaxGridSelector()
        self.layout = layout or GroupedTensorLayout()
        self.grouped_tensor_quantizer = grouped_tensor_quantizer or GroupedTensorQuantizer(layout=self.layout)
        self.max_input_batches = max_input_batches
        self.n_grid = n_grid
        self.eps = eps

    def quantize(
        self,
        layer_name: str,
        module_name: str,
        reference_inputs: list[LayerInputs],
        quant_inputs: list[LayerInputs],
        weight: torch.Tensor,
    ) -> torch.Tensor:
        _ = (layer_name, module_name, reference_inputs)
        activation = linear_input_matrix(quant_inputs, max_batches=self.max_input_batches)
        activation = activation.to(device=weight.device, dtype=torch.float32)
        weight_float = weight.to(dtype=torch.float32)

        target = F.linear(activation, weight_float)
        x_scale = activation.abs().mean(dim=0).clamp(min=self.eps)

        best_loss = torch.tensor(float("inf"), device=weight.device)
        best_weight = weight_float
        for ratio_index in range(self.n_grid):
            ratio = ratio_index / self.n_grid
            scales = self._awq_scales(x_scale, ratio)
            candidate = self._quantize_scaled_weight(weight_float, scales)
            loss = F.mse_loss(F.linear(activation, candidate), target)
            if loss < best_loss:
                best_loss = loss
                best_weight = candidate

        return best_weight.to(dtype=weight.dtype)

    def _awq_scales(self, x_scale: torch.Tensor, ratio: float) -> torch.Tensor:
        scales = x_scale.pow(ratio).clamp(min=self.eps)
        normalizer = torch.sqrt(scales.max() * scales.min()).clamp(min=self.eps)
        return scales / normalizer

    def _quantize_scaled_weight(self, weight: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        scales = scales.to(device=weight.device, dtype=weight.dtype)
        scaled_weight = weight * scales.view(1, -1)
        parameters = select_grouped_quantization_parameters(
            tensor=scaled_weight,
            selector=self.grid_selector,
            layout=self.layout,
        )
        quantized_scaled_weight = self.grouped_tensor_quantizer.quantize_dequantize(scaled_weight, parameters)
        return quantized_scaled_weight / scales.view(1, -1)


class QEPWeightQuantizer(WeightQuantizationAlgorithm):
    """QEP decorator for layer-wise weight quantizers."""

    needs_reference_inputs: bool = True
    needs_quant_inputs: bool = True

    def __init__(
        self,
        base_quantizer: Optional[WeightQuantizationAlgorithm] = None,
        alpha: float = 1.0,
        dampening: float = 0.01,
        max_input_batches: Optional[int] = 1,
        skip_module_suffixes: Sequence[str] = ("down_proj",),
    ) -> None:
        self._base_quantizer = base_quantizer or MinMaxDataDependentQuantizer()
        self.alpha = alpha
        self.dampening = dampening
        self.max_input_batches = max_input_batches
        self.skip_module_suffixes = tuple(skip_module_suffixes)

    @property
    def grid_selector(self) -> Optional[GridSelector]:
        return getattr(self._base_quantizer, "grid_selector", None)

    @grid_selector.setter
    def grid_selector(self, value: GridSelector) -> None:
        if hasattr(self._base_quantizer, "grid_selector"):
            self._base_quantizer.grid_selector = value

    def quantize(
        self,
        layer_name: str,
        module_name: str,
        reference_inputs: list[LayerInputs],
        quant_inputs: list[LayerInputs],
        weight: torch.Tensor,
    ) -> torch.Tensor:
        if self.skip_module_suffixes and any(module_name.endswith(suffix) for suffix in self.skip_module_suffixes):
            return self._base_quantizer.quantize(
                layer_name=layer_name,
                module_name=module_name,
                reference_inputs=reference_inputs,
                quant_inputs=quant_inputs,
                weight=weight,
            )

        quant_activation = linear_input_matrix(quant_inputs, max_batches=self.max_input_batches)
        reference_activation = linear_input_matrix(reference_inputs, max_batches=self.max_input_batches)
        quant_activation = quant_activation.to(device=weight.device, dtype=torch.float32)
        reference_activation = reference_activation.to(device=weight.device, dtype=torch.float32)
        weight_float = weight.to(dtype=torch.float32)

        hessian = quant_activation.T @ quant_activation
        diag = torch.arange(hessian.shape[0], device=hessian.device)
        damping = torch.clamp(self.dampening * torch.mean(torch.diag(hessian)), min=1e-8)
        hessian[diag, diag] += damping

        correction_left = weight_float @ (reference_activation - quant_activation).T @ quant_activation
        correction = torch.linalg.solve(hessian.T, correction_left.T).T
        corrected_weight = weight_float + self.alpha * correction

        return self._base_quantizer.quantize(
            layer_name=layer_name,
            module_name=module_name,
            reference_inputs=reference_inputs,
            quant_inputs=quant_inputs,
            weight=corrected_weight.to(dtype=weight.dtype),
        )



class GPTQWeightQuantizer(WeightQuantizationAlgorithm):
    """GPTQ-style sequential weight quantization with optional dynamic group grids."""

    needs_quant_inputs: bool = True

    def __init__(
        self,
        dampening: float = 0.01,
        grid_selector: Optional[GridSelector] = None,
        tensor_quantizer: Optional[TensorQuantizer] = None,
        group_layout: Optional[GroupedTensorLayout] = None,
        block_size: int = 128,
        act_order: bool = False,
        max_input_batches: Optional[int] = 1,
    ) -> None:
        self.grid_selector = grid_selector or ClipSearchGridSelector()
        self.tensor_quantizer = tensor_quantizer or TensorQuantizer()
        self.group_layout = group_layout or GroupedTensorLayout()
        self.block_size = block_size
        self.dampening = dampening
        self.act_order = act_order
        self.max_input_batches = max_input_batches

    def quantize(
        self,
        layer_name: str,
        module_name: str,
        reference_inputs: list[LayerInputs],
        quant_inputs: list[LayerInputs],
        weight: torch.Tensor,
    ) -> torch.Tensor:
        _ = (layer_name, module_name, reference_inputs)
        # [num_tokens, in_features]
        activation = linear_input_matrix(quant_inputs, max_batches=self.max_input_batches)
        activation = activation.to(device=weight.device, dtype=torch.float32)
        # ordered_weight: [out_features, in_features], ordered_activation: [num_tokens, in_features]
        ordered_weight, ordered_activation, inverse_order = self._apply_order(weight, activation)
        dead = ordered_activation.pow(2).sum(dim=0) == 0
        if torch.any(dead):
            ordered_weight = ordered_weight.clone()
            ordered_weight[:, dead] = 0
        # [in_features, in_features]
        hessian_chol_inv = self._cholesky_inverse_hessian(ordered_activation)
        # [in_features]
        hessian_diag = ordered_activation.pow(2).sum(dim=0)

        # [out_features, in_features]
        working_weight = ordered_weight.to(dtype=torch.float32).clone()
        # [out_features, in_features]
        quantized_weight = torch.empty_like(working_weight)
        num_columns = working_weight.shape[1]

        for group_start in range(0, num_columns, self.group_layout.group_size):
            group_end = min(group_start + self.group_layout.group_size, num_columns)
            err_mat = torch.zeros(
                working_weight.shape[0],
                group_end - group_start,
                device=working_weight.device,
                dtype=working_weight.dtype,
            )
            group_parameters = self._select_group_parameters(
                working_weight=working_weight,
                group_start=group_start,
                group_end=group_end,
                importance=hessian_diag[group_start:group_end],
            )

            for column in range(group_start, group_end):
                quantized_column = self.tensor_quantizer.quantize_dequantize(
                    working_weight[:, column : column + 1],
                    group_parameters,
                ).squeeze(1)
                quantized_weight[:, column] = quantized_column

                err_mat[:, column - group_start] = (working_weight[:, column] - quantized_column) / hessian_chol_inv[column, column]
                if column + 1 < group_end:
                    working_weight[:, column + 1 :group_end] -= err_mat[:, column - group_start].unsqueeze(1) * hessian_chol_inv[column, column + 1 :group_end]
            working_weight[:, group_end :] -= err_mat @ hessian_chol_inv[group_start:group_end, group_end:]

        if inverse_order is not None:
            quantized_weight = quantized_weight.index_select(dim=1, index=inverse_order)
        return quantized_weight.to(dtype=weight.dtype)

    def _select_group_parameters(
        self,
        working_weight: torch.Tensor,
        group_start: int,
        group_end: int,
        importance: torch.Tensor,
    ) -> QuantizationParameters:
        return self.grid_selector.select(
            working_weight[:, group_start:group_end],
            importance=importance,
        )

    def _apply_order(
        self,
        weight: torch.Tensor,
        activation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        order = None
        inverse_order = None
        if order is None and self.act_order:
            # scores/order/inverse_order: [in_features]
            scores = activation.pow(2).mean(dim=0)
            order = torch.argsort(scores, descending=True)
            inverse_order = torch.empty_like(order)
            inverse_order[order] = torch.arange(order.numel(), device=order.device)
        if order is None:
            return weight, activation, None

        order = order.to(device=weight.device, dtype=torch.long)
        inverse_order = inverse_order.to(device=weight.device, dtype=torch.long)
        return weight.index_select(1, order), activation.index_select(1, order), inverse_order

    def _cholesky_inverse_hessian(self, activation: torch.Tensor) -> torch.Tensor:
        # [in_features, in_features]
        hessian = activation.t().matmul(activation)
        dead = torch.diag(hessian) == 0
        if torch.any(dead):
            dead_idx = torch.nonzero(dead, as_tuple=False).flatten()
            hessian[dead_idx, dead_idx] = 1
        # [in_features]
        diag = torch.arange(hessian.shape[0], device=hessian.device)
        damping = torch.clamp(self.dampening * torch.mean(torch.diag(hessian)), min=1e-8)
        hessian[diag, diag] += damping
        # [in_features, in_features]
        inverse = torch.linalg.inv(hessian)
        return torch.linalg.cholesky(inverse, upper=True)

class GPTAQWeightQuantizer(GPTQWeightQuantizer):
    """GPTAQ-style GPTQ with activation-mismatch correction."""

    needs_reference_inputs: bool = True

    def __init__(
        self,
        dampening: float = 0.01,
        grid_selector: Optional[GridSelector] = None,
        tensor_quantizer: Optional[TensorQuantizer] = None,
        group_layout: Optional[GroupedTensorLayout] = None,
        block_size: int = 128,
        act_order: bool = False,
        max_input_batches: Optional[int] = 1,
        alpha: float = 0.25,
    ) -> None:
        super().__init__(
            dampening=dampening,
            grid_selector=grid_selector,
            tensor_quantizer=tensor_quantizer,
            group_layout=group_layout,
            block_size=block_size,
            act_order=act_order,
            max_input_batches=max_input_batches,
        )
        self.alpha = alpha

    def quantize(
        self,
        layer_name: str,
        module_name: str,
        reference_inputs: list[LayerInputs],
        quant_inputs: list[LayerInputs],
        weight: torch.Tensor,
    ) -> torch.Tensor:
        _ = (layer_name, module_name)
        activation_quant = linear_input_matrix(quant_inputs, max_batches=self.max_input_batches)
        activation_original = linear_input_matrix(reference_inputs, max_batches=self.max_input_batches)
        activation_quant = activation_quant.to(device=weight.device, dtype=torch.float32)
        activation_original = activation_original.to(device=weight.device, dtype=torch.float32)

        ordered_weight, ordered_activation, inverse_order = self._apply_order(weight, activation_quant)
        ordered_activation_original = self._apply_inverse_order(activation_original, inverse_order)

        hessian_chol_inv = self._cholesky_inverse_hessian(ordered_activation)
        hessian_diag = ordered_activation.pow(2).sum(dim=0)
        correction_matrix = self._activation_correction_matrix(
            ordered_activation_original=ordered_activation_original,
            ordered_activation_quant=ordered_activation,
            hessian_chol_inv=hessian_chol_inv,  # upper
        )

        working_weight = ordered_weight.to(dtype=torch.float32).clone()
        quantized_weight = torch.empty_like(working_weight)
        num_columns = working_weight.shape[1]

        for group_start in range(0, num_columns, self.group_layout.group_size):
            group_end = min(group_start + self.group_layout.group_size, num_columns)
            group_parameters = self._select_group_parameters(
                working_weight=working_weight,
                group_start=group_start,
                group_end=group_end,
                importance=hessian_diag[group_start:group_end],
            )
            err_mat = torch.zeros(
                working_weight.shape[0],
                group_end - group_start,
                device=working_weight.device,
                dtype=working_weight.dtype,
            )

            for column in range(group_start, group_end):
                working_column = working_weight[:, column]
                quantized_column = self.tensor_quantizer.quantize_dequantize(
                    working_column.unsqueeze(1),
                    group_parameters,
                ).squeeze(1)
                quantized_weight[:, column] = quantized_column

                err_mat[:, column - group_start] = (working_column - quantized_column) / hessian_chol_inv[column, column]
                if column + 1 < group_end:
                    working_weight[:, column + 1 :group_end] -= (
                        err_mat[:, column - group_start].unsqueeze(1) * hessian_chol_inv[column, column + 1 :group_end]
                        - working_column.unsqueeze(1) * correction_matrix[column, column + 1 :group_end]
                    )
            working_weight[:, group_end:] -= (
                        err_mat @ hessian_chol_inv[group_start:group_end, group_end:]
                        - working_weight[:, group_start:group_end] @ correction_matrix[group_start:group_end, group_end:]
                    )

        if inverse_order is not None:
            quantized_weight = quantized_weight.index_select(dim=1, index=inverse_order)
        return quantized_weight.to(dtype=weight.dtype)

    def _activation_correction_matrix(
        self,
        ordered_activation_original: torch.Tensor,
        ordered_activation_quant: torch.Tensor,
        hessian_chol_inv: torch.Tensor,
    ) -> torch.Tensor:
        delta_hessian = (ordered_activation_original - ordered_activation_quant).T @ ordered_activation_quant
        upper_delta = (delta_hessian @ hessian_chol_inv.T).triu(diagonal=1)
        return self.alpha * upper_delta @ hessian_chol_inv

    def _apply_inverse_order(
        self,
        activation: torch.Tensor,
        inverse_order: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if inverse_order is None:
            return activation
        order = torch.argsort(inverse_order.to(device=activation.device, dtype=torch.long))
        return activation.index_select(1, order)


class GPTQ_layer_quantization(GPTQWeightQuantizer):
    """Backward-compatible alias for GPTQWeightQuantizer."""

    def __init__(
        self,
        reg: float = 0.01,
        grid_selector: Optional[GridSelector] = None,
        tensor_quantizer: Optional[TensorQuantizer] = None,
        group_layout: Optional[GroupedTensorLayout] = None,
    ) -> None:
        super().__init__(
            dampening=reg,
            grid_selector=grid_selector,
            tensor_quantizer=tensor_quantizer,
            group_layout=group_layout,
        )


class QronosWeightQuantizer(GPTQWeightQuantizer):
    """Qronos-style sequential weight quantization with optional dynamic group grids."""

    needs_reference_inputs: bool = True
    needs_quant_inputs: bool = True

    def __init__(
        self,
        alpha: float = 1e-4,
        grid_selector: Optional[GridSelector] = None,
        tensor_quantizer: Optional[TensorQuantizer] = None,
        group_layout: Optional[GroupedTensorLayout] = None,
        block_size: int = 128,
        act_order: bool = False,
        max_input_batches: Optional[int] = 1,
    ) -> None:
        super().__init__(
            dampening=0.01,
            grid_selector=grid_selector,
            tensor_quantizer=tensor_quantizer,
            group_layout=group_layout,
            block_size=block_size,
            act_order=act_order,
            max_input_batches=max_input_batches,
        )
        self.alpha = alpha

    def quantize(
        self,
        layer_name: str,
        module_name: str,
        reference_inputs: list[LayerInputs],
        quant_inputs: list[LayerInputs],
        weight: torch.Tensor,
    ) -> torch.Tensor:
        _ = (layer_name, module_name, reference_inputs)
        # [num_tokens, in_features]
        activation_quant = linear_input_matrix(quant_inputs, max_batches=self.max_input_batches)
        activation_original = linear_input_matrix(reference_inputs, max_batches=self.max_input_batches)
        activation_quant = activation_quant.to(device=weight.device, dtype=torch.float32)
        activation_original = activation_original.to(device=weight.device, dtype=torch.float32)

        # ordered_weight: [out_features, in_features], ordered_activation: [num_tokens, in_features]
        ordered_weight, ordered_activation, inverse_order = self._apply_order(
            weight,
            activation_quant,
        )
        ordered_activation_original = self._order_activation_like(activation_original, inverse_order)
        # [in_features, in_features]
        H_mat = ordered_activation.T @ ordered_activation
        dead = torch.diag(H_mat) == 0
        if torch.any(dead):
            dead_idx = torch.nonzero(dead, as_tuple=False).flatten()
            H_mat[dead_idx, dead_idx] = 1
            ordered_weight = ordered_weight.clone()
            ordered_weight[:, dead] = 0
        damping = self._hessian_damping(H_mat)
        hessian_chol_inv = self._cholesky_inverse_hessian(H_mat, damping=damping)
        G_mat = ordered_activation.T @ ordered_activation_original
        # [in_features]
        hessian_diag = torch.diag(H_mat)

        # [out_features, in_features]
        working_weight = ordered_weight.to(dtype=torch.float32).clone()
        original_weight_math = working_weight.T.clone()
        # [out_features, in_features]
        quantized_weight = torch.empty_like(working_weight)
        num_columns = working_weight.shape[1]

        for group_start in range(0, num_columns, self.group_layout.group_size):
            group_end = min(group_start + self.group_layout.group_size, num_columns)
            group_parameters = self._select_group_parameters(
                working_weight=working_weight,
                group_start=group_start,
                group_end=group_end,
                importance=hessian_diag[group_start:group_end],
            )
            err_mat = torch.zeros(
                working_weight.shape[0],
                group_end - group_start,
                device=working_weight.device,
                dtype=working_weight.dtype,
            )
            for column in range(group_start, group_end):
                H00 = torch.clamp(H_mat[0, 0], min=1e-8)
                if column == 0:
                    temp = (G_mat[0, :] @ original_weight_math) / H00
                    if num_columns > 1:
                        temp -= (H_mat[0, 1:] @ original_weight_math[1:, :]) / H00
                    quantized_weight[:, 0] = self.tensor_quantizer.quantize_dequantize(
                        temp.unsqueeze(1),
                        group_parameters,
                    ).squeeze(1)
                    if num_columns > 1:
                        inverse_tail = hessian_chol_inv[1:, 1:].T @ hessian_chol_inv[1:, 1:]
                        G_damp = G_mat.clone()
                        diag = torch.arange(num_columns, device=G_damp.device)
                        G_damp[diag, diag] += damping
                        rhs = inverse_tail @ (
                            G_damp[1:, :] @ original_weight_math
                            - H_mat[1:, 0].unsqueeze(-1) * quantized_weight[:, 0].unsqueeze(0)
                        )
                        working_weight[:, 1:] = rhs.T
                    del temp
                    continue

                quantized_column = self.tensor_quantizer.quantize_dequantize(
                    working_weight[:, column : column + 1],
                    group_parameters,
                ).squeeze(1)
                quantized_weight[:, column] = quantized_column

                err_mat[:, column - group_start] = (working_weight[:, column] - quantized_column) / hessian_chol_inv[column, column]
                if column + 1 < group_end:
                    working_weight[:, column + 1 :group_end] -= err_mat[:, column - group_start].unsqueeze(1) * hessian_chol_inv[column, column + 1 :group_end]
            working_weight[:, group_end:] -= err_mat @ hessian_chol_inv[group_start:group_end, group_end:]

        if inverse_order is not None:
            quantized_weight = quantized_weight.index_select(dim=1, index=inverse_order)
        return quantized_weight.to(dtype=weight.dtype)

    def _order_activation_like(
        self,
        activation: torch.Tensor,
        inverse_order: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if inverse_order is None:
            return activation
        order = torch.argsort(inverse_order.to(device=activation.device, dtype=torch.long))
        return activation.index_select(1, order)

    def _hessian_damping(self, hessian: torch.Tensor) -> torch.Tensor:
        sigma_max = _power_iteration(hessian, steps=30)
        return torch.clamp(self.alpha * sigma_max, min=1e-8)

    def _cholesky_inverse_hessian(self, hessian: torch.Tensor, damping: torch.Tensor) -> torch.Tensor:
        # [in_features, in_features]
        damped_hessian = hessian.clone()
        diag = torch.arange(damped_hessian.shape[0], device=damped_hessian.device)
        damped_hessian[diag, diag] += damping
        inverse = torch.linalg.inv(damped_hessian)
        return torch.linalg.cholesky(inverse, upper=True)
