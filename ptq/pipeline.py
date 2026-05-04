from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional

import torch

from model_management import ModelManager, QuantModel
from .activation import (
    ActivationQuantizationConfig,
    ActivationQuantizedLinear,
    activation_calibration_module,
    wrap_linear_with_activation_quantization,
)
from .calibration import CalibrationInputCache, LayerInputs
from .grid import ClipSearchGridSelector, GroupedTensorLayout, MinMaxGridSelector
from .layers import LayerNavigator, LayerSpec
from .quantizers import (
    MinMaxDataDependentQuantizer,
    WeightQuantizationAlgorithm,
    linear_input_matrix,
)


@dataclass(frozen=True)
class PTQConfig:
    num_bits: int = 4
    group_size: int = 128
    block_size: int = 128
    symmetric: bool = True
    max_calibration_batches: Optional[int] = None
    act_order: bool = False
    activation_quant: bool = False
    activation_num_bits: int = 8
    activation_symmetric: bool = True
    activation_granularity: str = "per-token"
    grid_selection: str = "clip-search"
    local_block_replay: bool = False
    local_block_input_source: str = "reference"

    def resolved_grid_selection(self) -> str:
        if self.grid_selection not in {"minmax", "clip-search"}:
            raise ValueError(f"Unknown grid_selection: {self.grid_selection}")
        return self.grid_selection

    def resolved_local_block_input_source(self) -> str:
        if self.local_block_input_source not in {"reference", "quantized"}:
            raise ValueError(
                "Unknown local_block_input_source: "
                f"{self.local_block_input_source}. Expected 'reference' or 'quantized'."
            )
        return self.local_block_input_source


@dataclass(frozen=True)
class BlockInputStreams:
    reference_inputs: Optional[list[LayerInputs]] = None
    quant_inputs: Optional[list[LayerInputs]] = None


class ModelQuantizer:
    def __init__(
        self,
        manager: ModelManager,
        config: Optional[PTQConfig] = None,
        navigator: Optional[LayerNavigator] = None,
        weight_algorithm: Optional[WeightQuantizationAlgorithm] = None,
        debug_quant_stats: bool = False,
    ) -> None:
        self.manager = manager
        self.config = config or PTQConfig()
        self.navigator = navigator or LayerNavigator()
        self.group_layout = GroupedTensorLayout(group_size=self.config.group_size)
        self.weight_algorithm = weight_algorithm or MinMaxDataDependentQuantizer(
            grid_selector=MinMaxGridSelector(
                num_bits=self.config.num_bits,
                symmetric=self.config.symmetric,
            ),
            layout=self.group_layout,
        )
        self._configure_algorithm_grid_selection(self.weight_algorithm)
        self.input_cache = CalibrationInputCache(max_batches=self.config.max_calibration_batches)
        self.debug_quant_stats = debug_quant_stats

    def quantize_model(
        self,
        calibration_batches: Iterable[Any],
    ) -> QuantModel:
        batches = list(calibration_batches)
        quant_model = self._ensure_quant_model()
        if self.config.activation_quant:
            self._wrap_quant_linears_for_activation_quantization()
        self.quantize_decoder_layers(batches)
        quant_model.mark_quantized()
        return quant_model

    def quantize_decoder_layers(
        self,
        calibration_batches: list[Any],
    ) -> None:
        reference = self.manager.reference_model
        quant = self._ensure_quant_model()
        reference.eval().freeze()
        quant.eval()

        layer_specs = list(self.navigator.iter_layer_specs(reference.model, quant.model))
        if not layer_specs:
            return

        need_reference_stream, need_quant_stream = self._calibration_stream_requirements(
            weight_algorithm=self.weight_algorithm,
        )
        calibration_streams = self._initial_block_streams(
            spec=layer_specs[0],
            batches=calibration_batches,
            need_reference_inputs=need_reference_stream,
            need_quant_inputs=need_quant_stream,
        )

        for index, spec in enumerate(layer_specs):
            reference_linears = dict(self.navigator.iter_quantizable_linears(spec.reference_layer))
            if reference_linears:
                quant_linears = {
                    module_name: spec.quant_layer.get_submodule(module_name)
                    for module_name in reference_linears
                }
                self._quantize_layer_linears(
                    layer_name=spec.name,
                    reference_layer=spec.reference_layer,
                    quant_layer=spec.quant_layer,
                    reference_linears=reference_linears,
                    quant_linears=quant_linears,
                    weight_for_module=lambda module_name: quant_linears[module_name].weight,
                    weight_algorithm=self.weight_algorithm,
                    calibration_streams=calibration_streams,
                    phase="quant",
                )

            if index + 1 < len(layer_specs):
                calibration_streams = self._advance_block_streams(
                    reference_layer=spec.reference_layer,
                    quant_layer=spec.quant_layer,
                    streams=calibration_streams,
                )

    def _quantize_layer_linears(
        self,
        layer_name: str,
        reference_layer: torch.nn.Module,
        quant_layer: torch.nn.Module,
        reference_linears: dict[str, torch.nn.Module],
        quant_linears: dict[str, torch.nn.Module],
        weight_for_module: Callable[[str], torch.Tensor],
        weight_algorithm: WeightQuantizationAlgorithm,
        calibration_streams: BlockInputStreams,
        phase: str,
    ) -> None:
        quant = self._ensure_quant_model()
        need_reference_inputs = getattr(weight_algorithm, "needs_reference_inputs", False)
        need_quant_inputs = getattr(weight_algorithm, "needs_quant_inputs", False) or self.debug_quant_stats

        if need_reference_inputs:
            reference = self.manager.reference_model
            reference_boundary_inputs = self._select_reference_boundary_inputs(
                streams=calibration_streams,
                layer_name=layer_name,
            )
            reference_linear_inputs = self.input_cache.collect_many_from_layer(
                reference_layer,
                reference_linears,
                reference_boundary_inputs,
                device=reference.device,
            )
        else:
            reference_linear_inputs = {module_name: [] for module_name in reference_linears}

        quant_boundary_inputs = (
            self._select_quant_boundary_inputs(
                streams=calibration_streams,
                layer_name=layer_name,
            )
            if need_quant_inputs
            else None
        )

        for module_name, quant_linear in quant_linears.items():
            if need_quant_inputs:
                quant_inputs = self.input_cache.collect_from_layer(
                    quant_layer,
                    activation_calibration_module(quant_linear),
                    quant_boundary_inputs,
                    device=quant.device,
                )
            else:
                quant_inputs = []
            weight = weight_for_module(module_name).detach()
            if self.debug_quant_stats:
                print(self._format_quant_stats(phase, layer_name, module_name, weight, quant_inputs))
            try:
                quantized_weight = weight_algorithm.quantize(
                    layer_name=layer_name,
                    module_name=module_name,
                    reference_inputs=reference_linear_inputs[module_name],
                    quant_inputs=quant_inputs,
                    weight=weight,
                )
            except Exception as exc:
                raise RuntimeError(
                    "Layer linear quantization failed for "
                    f"{phase} {layer_name}.{module_name} with {type(weight_algorithm).__name__}. "
                    f"weight_shape={tuple(weight.shape)}, "
                    f"num_quant_inputs={len(quant_inputs)}, "
                    f"num_reference_inputs={len(reference_linear_inputs[module_name])}."
                ) from exc
            with torch.no_grad():
                quant_linear.weight.copy_(quantized_weight.to(quant_linear.weight.device))
            quant.register_quantized_layer(f"{layer_name}.{module_name}", quant_linear)
            del quant_inputs, quantized_weight
        del reference_linear_inputs

    def _select_reference_boundary_inputs(
        self,
        streams: BlockInputStreams,
        layer_name: str,
    ) -> list[LayerInputs]:
        if self.config.local_block_replay:
            return self._stream_inputs_for_source(
                streams,
                source=self.config.resolved_local_block_input_source(),
                layer_name=layer_name,
            )
        return self._require_stream(streams.reference_inputs, stream_name="reference", layer_name=layer_name)

    def _select_quant_boundary_inputs(
        self,
        streams: BlockInputStreams,
        layer_name: str,
    ) -> list[LayerInputs]:
        if self.config.local_block_replay:
            return self._stream_inputs_for_source(
                streams,
                source=self.config.resolved_local_block_input_source(),
                layer_name=layer_name,
            )
        return self._require_stream(streams.quant_inputs, stream_name="quantized", layer_name=layer_name)

    def _stream_inputs_for_source(
        self,
        streams: BlockInputStreams,
        source: str,
        layer_name: str,
    ) -> list[LayerInputs]:
        if source == "reference":
            return self._require_stream(streams.reference_inputs, stream_name="reference", layer_name=layer_name)
        if source == "quantized":
            return self._require_stream(streams.quant_inputs, stream_name="quantized", layer_name=layer_name)
        raise ValueError(f"Unknown block input source: {source}")

    def _calibration_stream_requirements(
        self,
        weight_algorithm: WeightQuantizationAlgorithm,
    ) -> tuple[bool, bool]:
        need_reference_stream = getattr(weight_algorithm, "needs_reference_inputs", False)
        need_quant_stream = getattr(weight_algorithm, "needs_quant_inputs", False) or self.debug_quant_stats
        if self.config.local_block_replay:
            need_stream = need_reference_stream or need_quant_stream
            if self.config.resolved_local_block_input_source() == "reference":
                return need_stream, False
            return False, need_stream
        return need_reference_stream, need_quant_stream

    def _initial_block_streams(
        self,
        spec: LayerSpec,
        batches: list[Any],
        need_reference_inputs: bool,
        need_quant_inputs: bool,
    ) -> BlockInputStreams:
        reference = self.manager.reference_model
        quant = self._ensure_quant_model()
        reference_inputs = (
            self.input_cache.collect(reference.model, spec.reference_layer, batches, device=reference.device)
            if need_reference_inputs
            else None
        )
        quant_inputs = (
            self.input_cache.collect(quant.model, spec.quant_layer, batches, device=quant.device)
            if need_quant_inputs
            else None
        )
        return BlockInputStreams(reference_inputs=reference_inputs, quant_inputs=quant_inputs)

    def _advance_block_streams(
        self,
        reference_layer: torch.nn.Module,
        quant_layer: torch.nn.Module,
        streams: BlockInputStreams,
    ) -> BlockInputStreams:
        reference = self.manager.reference_model
        quant = self._ensure_quant_model()
        reference_inputs = (
            self.input_cache.advance_layer_inputs(reference_layer, streams.reference_inputs, device=reference.device)
            if streams.reference_inputs is not None
            else None
        )
        quant_inputs = (
            self.input_cache.advance_layer_inputs(quant_layer, streams.quant_inputs, device=quant.device)
            if streams.quant_inputs is not None
            else None
        )
        return BlockInputStreams(reference_inputs=reference_inputs, quant_inputs=quant_inputs)

    def _require_stream(
        self,
        inputs: Optional[list[LayerInputs]],
        stream_name: str,
        layer_name: str,
    ) -> list[LayerInputs]:
        if inputs is None:
            raise RuntimeError(f"Missing {stream_name} block input stream for {layer_name}.")
        return inputs

    def _format_quant_stats(
        self,
        phase: str,
        layer_name: str,
        module_name: str,
        weight: torch.Tensor,
        quant_inputs: list[Any],
    ) -> str:
        activation = linear_input_matrix(quant_inputs, max_batches=self.config.max_calibration_batches).float()
        hessian_diag = activation.pow(2).sum(dim=0)
        zero_columns = int((hessian_diag == 0).sum().item())
        finite_activation = torch.isfinite(activation)
        finite_weight = torch.isfinite(weight)
        finite_hessian_diag = torch.isfinite(hessian_diag)
        return (
            f"[quant-stats] {phase} {layer_name}.{module_name} "
            f"weight_shape={tuple(weight.shape)} "
            f"act_shape={tuple(activation.shape)} "
            f"weight_finite={int(finite_weight.sum().item())}/{weight.numel()} "
            f"act_finite={int(finite_activation.sum().item())}/{activation.numel()} "
            f"hdiag_finite={int(finite_hessian_diag.sum().item())}/{hessian_diag.numel()} "
            f"hdiag_min={float(hessian_diag.min().item()):.6e} "
            f"hdiag_max={float(hessian_diag.max().item()):.6e} "
            f"zero_hdiag_cols={zero_columns}"
        )

    def _ensure_quant_model(self) -> QuantModel:
        if self.manager.quant_model is None:
            return self.manager.build_quant_model_from_reference(
                quant_config={
                    "num_bits": self.config.num_bits,
                    "group_size": self.config.group_size,
                    "block_size": self.config.block_size,
                    "symmetric": self.config.symmetric,
                    "grid_selection": self.config.resolved_grid_selection(),
                    "activation_quant": self.config.activation_quant,
                    "activation_num_bits": self.config.activation_num_bits,
                    "activation_symmetric": self.config.activation_symmetric,
                    "activation_granularity": self.config.activation_granularity,
                    "local_block_replay": self.config.local_block_replay,
                    "local_block_input_source": self.config.resolved_local_block_input_source(),
                }
            )
        return self.manager.quant_model

    def _configure_algorithm_grid_selection(self, algorithm: WeightQuantizationAlgorithm) -> None:
        if not hasattr(algorithm, "grid_selector"):
            return
        selection = self.config.resolved_grid_selection()
        if selection == "minmax":
            algorithm.grid_selector = MinMaxGridSelector(
                num_bits=self.config.num_bits,
                symmetric=self.config.symmetric,
                eps=1e-8,
            )
            return
        algorithm.grid_selector = ClipSearchGridSelector(
            num_bits=self.config.num_bits,
            symmetric=self.config.symmetric,
            eps=1e-8,
        )

    def _wrap_quant_linears_for_activation_quantization(self) -> None:
        reference = self.manager.reference_model
        quant = self._ensure_quant_model()
        activation_config = ActivationQuantizationConfig(
            num_bits=self.config.activation_num_bits,
            symmetric=self.config.activation_symmetric,
            granularity=self.config.activation_granularity,
        )

        for spec in self.navigator.iter_layer_specs(reference.model, quant.model):
            for module_name, module in list(self.navigator.iter_quantizable_linears(spec.quant_layer)):
                if isinstance(module, ActivationQuantizedLinear):
                    continue
                if not isinstance(module, torch.nn.Linear):
                    continue
                self._replace_submodule(
                    root=spec.quant_layer,
                    module_name=module_name,
                    replacement=wrap_linear_with_activation_quantization(module, activation_config),
                )

    def _replace_submodule(self, root: torch.nn.Module, module_name: str, replacement: torch.nn.Module) -> None:
        parent_name, _, child_name = module_name.rpartition(".")
        parent = root.get_submodule(parent_name) if parent_name else root
        setattr(parent, child_name, replacement)
