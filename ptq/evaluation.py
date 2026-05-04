from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Any, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .calibration import forward_batch, move_batch_to_device


@dataclass(frozen=True)
class PerplexityResult:
    perplexity: float
    mean_nll: float
    num_tokens: int
    num_batches: int


def evaluate_perplexity(
    model: Any,
    eval_batches: Iterable[Any],
    device: Optional[torch.device] = None,
    max_batches: Optional[int] = None,
    ignore_index: int = -100,
) -> PerplexityResult:
    module = _unwrap_model(model)
    device = device if device is not None else getattr(model, "device", None)
    was_training = module.training
    module.eval()

    total_nll = 0.0
    total_tokens = 0
    num_batches = 0

    try:
        with torch.no_grad():
            for batch_index, batch in enumerate(eval_batches):
                if max_batches is not None and batch_index >= max_batches:
                    break

                batch = move_batch_to_device(batch, device)
                labels = _labels_from_batch(batch)
                output = forward_batch(module, batch)
                logits = _get_logits(output)
                batch_nll, batch_tokens = _causal_lm_nll(
                    logits=logits,
                    labels=labels.to(logits.device),
                    ignore_index=ignore_index,
                )
                total_nll += batch_nll
                total_tokens += batch_tokens
                num_batches += 1
    finally:
        if was_training:
            module.train()

    if total_tokens == 0:
        raise ValueError("No valid tokens were found while evaluating perplexity.")

    mean_nll = total_nll / total_tokens
    return PerplexityResult(
        perplexity=exp(mean_nll),
        mean_nll=mean_nll,
        num_tokens=total_tokens,
        num_batches=num_batches,
    )


class PerplexityEvaluator:
    def __init__(
        self,
        device: Optional[torch.device] = None,
        max_batches: Optional[int] = None,
        ignore_index: int = -100,
    ) -> None:
        self.device = device
        self.max_batches = max_batches
        self.ignore_index = ignore_index

    def evaluate(self, model: Any, eval_batches: Iterable[Any]) -> PerplexityResult:
        return evaluate_perplexity(
            model=model,
            eval_batches=eval_batches,
            device=self.device,
            max_batches=self.max_batches,
            ignore_index=self.ignore_index,
        )


def _unwrap_model(model: Any) -> nn.Module:
    if isinstance(model, nn.Module):
        return model
    if hasattr(model, "model") and isinstance(model.model, nn.Module):
        return model.model
    raise TypeError("Expected an nn.Module or a model wrapper with an nn.Module 'model' property.")


def _labels_from_batch(batch: Any) -> torch.Tensor:
    if isinstance(batch, dict):
        if "labels" in batch:
            return batch["labels"]
        if "input_ids" in batch:
            return batch["input_ids"]
    if isinstance(batch, tuple) and batch and torch.is_tensor(batch[0]):
        return batch[0]
    if torch.is_tensor(batch):
        return batch
    raise ValueError(
        "Could not infer labels for perplexity. Provide batches with 'labels', "
        "'input_ids', a first tensor tuple item, or a tensor batch."
    )


def _get_logits(output: Any) -> torch.Tensor:
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, dict) and "logits" in output:
        return output["logits"]
    if isinstance(output, tuple):
        return output[0]
    if torch.is_tensor(output):
        return output
    raise ValueError("Could not infer logits from model output.")


def _causal_lm_nll(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int,
) -> tuple[float, int]:
    if logits.ndim < 3:
        raise ValueError(f"Expected logits with shape [..., seq_len, vocab], got {tuple(logits.shape)}")
    if labels.ndim != logits.ndim - 1:
        raise ValueError(
            f"Expected labels with one fewer dimension than logits, got "
            f"logits={tuple(logits.shape)}, labels={tuple(labels.shape)}"
        )
    if logits.shape[:-1] != labels.shape:
        raise ValueError(
            f"Expected logits prefix shape to match labels, got "
            f"logits={tuple(logits.shape)}, labels={tuple(labels.shape)}"
        )
    if logits.shape[-2] < 2:
        raise ValueError("Need at least two sequence positions to compute causal LM perplexity.")

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    valid_tokens = int((shift_labels != ignore_index).sum().item())
    if valid_tokens == 0:
        return 0.0, 0

    loss_sum = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index,
        reduction="sum",
    )
    return float(loss_sum.item()), valid_tokens
