from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ZeroShotResult:
    tasks: tuple[str, ...]
    results: dict[str, dict[str, Any]]
    versions: dict[str, Any]
    raw: dict[str, Any]


def evaluate_zeroshot(
    model: Any,
    tokenizer: Any,
    tasks: Sequence[str],
    device: Optional[torch.device] = None,
    batch_size: str | int = "auto",
    num_fewshot: int = 0,
    limit: Optional[int | float] = None,
    trust_remote_code: bool = False,
) -> ZeroShotResult:
    """Evaluate an in-memory causal LM with lm-evaluation-harness."""
    lm_eval = _import_lm_eval()
    module = _unwrap_model(model)
    was_training = module.training
    module.eval()

    try:
        lm = _build_hflm(
            module=module,
            tokenizer=tokenizer,
            device=_device_string(device),
            batch_size=batch_size,
            trust_remote_code=trust_remote_code,
        )
        raw = _simple_evaluate(
            lm_eval=lm_eval,
            model=lm,
            tasks=list(tasks),
            batch_size=batch_size,
            num_fewshot=num_fewshot,
            limit=limit,
        )
    finally:
        if was_training:
            module.train()

    return ZeroShotResult(
        tasks=tuple(tasks),
        results=raw.get("results", {}),
        versions=raw.get("versions", {}),
        raw=raw,
    )


def _import_lm_eval():
    try:
        import lm_eval
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Zero-shot evaluation requires lm-evaluation-harness. "
            "Install it in this environment, for example: "
            "pip install lm-eval"
        ) from exc
    return lm_eval


def _build_hflm(
    module: nn.Module,
    tokenizer: Any,
    device: Optional[str],
    batch_size: str | int,
    trust_remote_code: bool,
) -> Any:
    try:
        from lm_eval.models.huggingface import HFLM
    except ImportError:
        from lm_eval.models.huggingface import HuggingFaceAutoLM as HFLM

    kwargs = {
        "pretrained": module,
        "tokenizer": tokenizer,
        "batch_size": batch_size,
        "trust_remote_code": trust_remote_code,
    }
    if device is not None:
        kwargs["device"] = device

    try:
        return HFLM(**kwargs)
    except TypeError:
        kwargs.pop("trust_remote_code", None)
        return HFLM(**kwargs)


def _simple_evaluate(
    lm_eval: Any,
    model: Any,
    tasks: list[str],
    batch_size: str | int,
    num_fewshot: int,
    limit: Optional[int | float],
) -> dict[str, Any]:
    try:
        simple_evaluate = lm_eval.simple_evaluate
    except AttributeError:
        from lm_eval import evaluator

        simple_evaluate = evaluator.simple_evaluate

    try:
        return simple_evaluate(
            model=model,
            tasks=tasks,
            num_fewshot=num_fewshot,
            batch_size=batch_size,
            limit=limit,
        )
    except TypeError:
        return simple_evaluate(
            model=model,
            tasks=tasks,
            num_fewshot=num_fewshot,
            limit=limit,
        )


def _unwrap_model(model: Any) -> nn.Module:
    if isinstance(model, nn.Module):
        return model
    if hasattr(model, "model") and isinstance(model.model, nn.Module):
        return model.model
    raise TypeError("Expected an nn.Module or a model wrapper with an nn.Module 'model' property.")


def _device_string(device: Optional[torch.device]) -> Optional[str]:
    if device is None:
        return None
    if device.type == "cuda" and device.index is None:
        return "cuda:0"
    return str(device)
