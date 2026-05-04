from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model_management import ModelManager, ReferenceModel
from ptq import (
    AWQWeightQuantizer,
    ClipSearchGridSelector,
    GPTAQWeightQuantizer,
    GPTQWeightQuantizer,
    GroupedTensorLayout,
    ModelQuantizer,
    MinMaxGridSelector,
    MinMaxDataDependentQuantizer,
    PTQConfig,
    QEPWeightQuantizer,
    QronosWeightQuantizer,
    evaluate_perplexity,
    evaluate_zeroshot,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Qwen + WikiText perplexity smoke test for the PTQ framework."
    )
    parser.add_argument("--model-name", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--dataset-name", default="wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--calibration-split", default="train")
    parser.add_argument("--calibration-offset-batches", type=int, default=0)
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--calibration-batches", type=int, default=1)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--num-bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument(
        "--algorithm",
        choices=(
            "rtn",
            "awq",
            "gptq",
            "gptaq",
            "qronos",
            "qep-rtn",
            "qep-awq",
            "qep-gptq",
            "qep-gptaq",
            "qep-qronos",
        ),
        default="rtn",
    )
    parser.add_argument("--qep", action="store_true", help="Apply QEP to the selected weight quantizer.")
    parser.add_argument("--qep-alpha", type=float, default=1.0)
    parser.add_argument("--qep-dampening", type=float, default=0.01)
    parser.add_argument(
        "--qep-skip-down-proj",
        action="store_true",
        default=True,
        help="Skip QEP correction for modules ending in 'down_proj' while still applying the base quantizer.",
    )
    parser.add_argument(
        "--no-qep-skip-down-proj",
        dest="qep_skip_down_proj",
        action="store_false",
        help="Apply QEP correction to down_proj as well.",
    )
    parser.add_argument("--dampening", type=float, default=0.01)
    parser.add_argument("--qronos-alpha", type=float, default=1e-4)
    parser.add_argument("--awq-grid-size", type=int, default=20)
    parser.add_argument("--gptaq-alpha", type=float, default=0.25)
    parser.add_argument("--debug-quant-stats", action="store_true")
    parser.add_argument("--act-order", action="store_true")
    parser.add_argument("--local-block-replay", action="store_true")
    parser.add_argument("--local-block-input-source", choices=("reference", "quantized"), default="reference")
    parser.add_argument(
        "--activation-quant",
        action="store_true",
        help="Enable fake activation quantization on quantized Linear inputs. Disabled by default.",
    )
    parser.add_argument("--activation-num-bits", type=int, default=8)
    parser.add_argument(
        "--activation-granularity",
        choices=("per-token", "per-tensor"),
        default="per-token",
    )
    parser.add_argument("--activation-asymmetric", action="store_true")
    parser.add_argument(
        "--grid-selection",
        choices=("minmax", "clip-search"),
        default="clip-search",
        help=(
            "How GPTQ/GPTAQ/Qronos select a quantization grid for each group: "
            "direct min-max or clip-ratio search."
        ),
    )
    parser.add_argument("--skip-reference-eval", action="store_true")
    parser.add_argument("--run-zeroshot", action="store_true")
    parser.add_argument("--run-reference-zeroshot", action="store_true")
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument(
        "--zeroshot-tasks",
        default="piqa,hellaswag,arc_easy,arc_challenge",
        help="Comma-separated lm-evaluation-harness task names.",
    )
    parser.add_argument("--zeroshot-num-fewshot", type=int, default=0)
    parser.add_argument("--zeroshot-batch-size", default="auto")
    parser.add_argument(
        "--zeroshot-limit",
        type=parse_optional_limit,
        default=None,
        help="Optional lm-eval limit, useful for smoke tests. Accepts int counts or float fractions.",
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    parser.add_argument("--dtype", default="auto", choices=("auto", "float32", "float16", "bfloat16"))
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv-output", default=None)
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    )
    model.config.use_cache = False
    model.to(device)

    print(f"Loading dataset: {args.dataset_name}/{args.dataset_config}")
    dataset = load_dataset(args.dataset_name, args.dataset_config)
    calibration_batches = build_lm_batches(
        tokenizer=tokenizer,
        texts=dataset[args.calibration_split]["text"],
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        num_batches=args.calibration_batches,
        offset_batches=args.calibration_offset_batches,
    )
    eval_batches = build_lm_batches(
        tokenizer=tokenizer,
        texts=dataset[args.eval_split]["text"],
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        num_batches=args.eval_batches,
    )
    print(
        f"Prepared {len(calibration_batches)} calibration batches and "
        f"{len(eval_batches)} eval batches."
    )

    reference = ReferenceModel(model=model, device=device, freeze=True)
    manager = ModelManager(reference_model=reference)

    reference_result = None
    reference_zeroshot = None
    if not args.skip_reference_eval:
        print("Evaluating reference perplexity...")
        reference_result = evaluate_perplexity(
            model=manager.reference_model,
            eval_batches=eval_batches,
            device=device,
        )
        print_result("reference", reference_result)
    if args.run_zeroshot and args.run_reference_zeroshot:
        zeroshot_tasks = parse_tasks(args.zeroshot_tasks)
        print(f"Evaluating reference zero-shot tasks: {','.join(zeroshot_tasks)}")
        reference_zeroshot = evaluate_zeroshot(
            model=manager.reference_model,
            tokenizer=tokenizer,
            tasks=zeroshot_tasks,
            device=device,
            batch_size=args.zeroshot_batch_size,
            num_fewshot=args.zeroshot_num_fewshot,
            limit=args.zeroshot_limit,
            trust_remote_code=args.trust_remote_code,
        )
        print_zeroshot_result("reference", reference_zeroshot)
    if args.reference_only:
        if args.csv_output is not None:
            append_csv_result(
                path=Path(args.csv_output),
                args=args,
                config=None,
                reference_result=reference_result,
                quant_result=None,
                reference_zeroshot=reference_zeroshot,
                quant_zeroshot=None,
                quantized_layers=0,
            )
        return

    config = PTQConfig(
        num_bits=args.num_bits,
        group_size=args.group_size,
        block_size=args.block_size,
        symmetric=True,
        max_calibration_batches=args.calibration_batches,
        act_order=args.act_order,
        activation_quant=args.activation_quant,
        activation_num_bits=args.activation_num_bits,
        activation_symmetric=not args.activation_asymmetric,
        activation_granularity=args.activation_granularity,
        grid_selection=args.grid_selection,
        local_block_replay=args.local_block_replay,
        local_block_input_source=args.local_block_input_source,
    )
    weight_algorithm = build_weight_algorithm(
        args.algorithm,
        args.dampening,
        args,
        config,
        qep_enabled=args.qep,
        qep_alpha=args.qep_alpha,
        qep_skip_down_proj=args.qep_skip_down_proj,
    )

    print(f"Running PTQ with config: {asdict(config)}")
    print(
        f"Weight algorithm: {args.algorithm} "
        f"dampening={args.dampening} qronos_alpha={args.qronos_alpha} "
        f"qep={uses_qep(args.algorithm, args.qep)}"
    )
    quantizer = ModelQuantizer(
        manager=manager,
        config=config,
        weight_algorithm=weight_algorithm,
        debug_quant_stats=args.debug_quant_stats,
    )
    quant_model = quantizer.quantize_model(calibration_batches)
    print(f"Quantized layers: {len(quant_model.quantized_layers)}")

    print("Evaluating quantized perplexity...")
    quant_result = evaluate_perplexity(
        model=quant_model,
        eval_batches=eval_batches,
        device=device,
    )
    print_result("quantized", quant_result)
    quant_zeroshot = None
    if args.run_zeroshot:
        zeroshot_tasks = parse_tasks(args.zeroshot_tasks)
        print(f"Evaluating quantized zero-shot tasks: {','.join(zeroshot_tasks)}")
        quant_zeroshot = evaluate_zeroshot(
            model=quant_model,
            tokenizer=tokenizer,
            tasks=zeroshot_tasks,
            device=device,
            batch_size=args.zeroshot_batch_size,
            num_fewshot=args.zeroshot_num_fewshot,
            limit=args.zeroshot_limit,
            trust_remote_code=args.trust_remote_code,
        )
        print_zeroshot_result("quantized", quant_zeroshot)
    if args.csv_output is not None:
        append_csv_result(
            path=Path(args.csv_output),
            args=args,
            config=config,
            reference_result=reference_result,
            quant_result=quant_result,
            reference_zeroshot=reference_zeroshot,
            quant_zeroshot=quant_zeroshot,
            quantized_layers=len(quant_model.quantized_layers),
        )


def build_weight_algorithm(
    algorithm: str,
    dampening: float,
    args: argparse.Namespace,
    config: PTQConfig,
    qep_enabled: bool = False,
    qep_alpha: float = 1.0,
    qep_skip_down_proj: bool = False,
):
    base_algorithm, algorithm_qep_enabled = parse_qep_algorithm(algorithm)
    group_layout = GroupedTensorLayout(group_size=config.group_size)
    grid_selector = build_grid_selector(config, num_bits=config.num_bits, symmetric=config.symmetric)
    if base_algorithm == "rtn":
        quantizer = MinMaxDataDependentQuantizer(
            grid_selector=MinMaxGridSelector(
                num_bits=config.num_bits,
                symmetric=config.symmetric,
                eps=1e-8,
            ),
            layout=group_layout,
        )
    elif base_algorithm == "awq":
        quantizer = AWQWeightQuantizer(
            grid_selector=MinMaxGridSelector(
                num_bits=config.num_bits,
                symmetric=config.symmetric,
                eps=1e-8,
            ),
            layout=group_layout,
            max_input_batches=args.calibration_batches,
            n_grid=args.awq_grid_size,
        )
    elif base_algorithm == "qronos":
        quantizer = QronosWeightQuantizer(
            alpha=args.qronos_alpha,
            grid_selector=grid_selector,
            group_layout=group_layout,
            block_size=config.block_size,
            act_order=args.act_order,
            max_input_batches=args.calibration_batches,
        )
    elif base_algorithm == "gptaq":
        quantizer = GPTAQWeightQuantizer(
            dampening=dampening,
            grid_selector=grid_selector,
            group_layout=group_layout,
            block_size=config.block_size,
            act_order=args.act_order,
            max_input_batches=args.calibration_batches,
            alpha=args.gptaq_alpha,
        )
    else:
        quantizer = GPTQWeightQuantizer(
            dampening=dampening,
            grid_selector=grid_selector,
            group_layout=group_layout,
            block_size=config.block_size,
            act_order=args.act_order,
            max_input_batches=args.calibration_batches,
        )

    if qep_enabled or algorithm_qep_enabled:
        return QEPWeightQuantizer(
            base_quantizer=quantizer,
            alpha=qep_alpha,
            dampening=args.qep_dampening,
            max_input_batches=args.calibration_batches,
            skip_module_suffixes=("down_proj",) if qep_skip_down_proj else (),
        )
    return quantizer


def parse_qep_algorithm(algorithm: str) -> tuple[str, bool]:
    if algorithm.startswith("qep-"):
        return algorithm.removeprefix("qep-"), True
    return algorithm, False


def uses_qep(algorithm: str, flag_enabled: bool) -> bool:
    return flag_enabled or algorithm.startswith("qep-")


def uses_algorithm(args: argparse.Namespace, base_algorithm: str) -> bool:
    stage1_algorithm, _ = parse_qep_algorithm(args.algorithm)
    return base_algorithm == stage1_algorithm


def build_lm_batches(
    tokenizer,
    texts: Iterable[str],
    sequence_length: int,
    batch_size: int,
    num_batches: int,
    offset_batches: int = 0,
) -> list[dict[str, torch.Tensor]]:
    target_tokens = sequence_length * batch_size * num_batches
    offset_tokens = sequence_length * batch_size * offset_batches
    token_ids: list[int] = []
    eos_token_id = tokenizer.eos_token_id

    for text in texts:
        if not text or not text.strip():
            continue
        encoded = tokenizer(text, add_special_tokens=False).input_ids
        token_ids.extend(encoded)
        if eos_token_id is not None:
            token_ids.append(eos_token_id)
        if len(token_ids) >= offset_tokens + target_tokens + sequence_length:
            break

    tokens_per_batch = sequence_length * batch_size
    usable_tokens = ((len(token_ids) - offset_tokens) // tokens_per_batch) * tokens_per_batch
    usable_tokens = min(usable_tokens, target_tokens)
    if usable_tokens < tokens_per_batch:
        raise ValueError(
            f"Not enough tokens to build one batch after offset: got {len(token_ids)}, "
            f"offset={offset_tokens}, need at least {offset_tokens + tokens_per_batch}."
        )

    input_ids = torch.tensor(token_ids[offset_tokens : offset_tokens + usable_tokens], dtype=torch.long)
    input_ids = input_ids.view(-1, batch_size, sequence_length).transpose(0, 1)

    batches = []
    for batch_index in range(input_ids.shape[1]):
        batch_input_ids = input_ids[:, batch_index, :].contiguous()
        batches.append(
            {
                "input_ids": batch_input_ids,
                "attention_mask": torch.ones_like(batch_input_ids),
            }
        )
    return batches


def parse_tasks(tasks_arg: str) -> list[str]:
    tasks = [task.strip() for task in tasks_arg.split(",") if task.strip()]
    if not tasks:
        raise ValueError("At least one zero-shot task is required.")
    return tasks


def parse_optional_limit(value: str) -> int | float:
    if "." in value:
        return float(value)
    return int(value)


def build_grid_selector(
    config: PTQConfig,
    num_bits: int,
    symmetric: bool,
):
    if config.resolved_grid_selection() == "minmax":
        return MinMaxGridSelector(
            num_bits=num_bits,
            symmetric=symmetric,
            eps=1e-8,
        )
    return ClipSearchGridSelector(
        num_bits=num_bits,
        symmetric=symmetric,
        eps=1e-8,
    )


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(dtype_arg: str, device: torch.device) -> torch.dtype:
    if dtype_arg == "float32":
        return torch.float32
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "bfloat16":
        return torch.bfloat16
    if device.type == "cuda":
        return torch.float16
    return torch.float32


def print_result(name: str, result) -> None:
    print(
        f"{name}: ppl={result.perplexity:.4f}, "
        f"mean_nll={result.mean_nll:.4f}, "
        f"tokens={result.num_tokens}, batches={result.num_batches}"
    )


def print_zeroshot_result(name: str, result) -> None:
    print(f"{name} zero-shot:")
    for task in result.tasks:
        metrics = result.results.get(task, {})
        if not metrics:
            print(f"  {task}: no metrics returned")
            continue
        metric_parts = []
        for metric_name, metric_value in sorted(metrics.items()):
            if metric_name.endswith("_stderr") or metric_name == "alias":
                continue
            if isinstance(metric_value, (int, float)):
                metric_parts.append(f"{metric_name}={metric_value:.4f}")
        if metric_parts:
            print(f"  {task}: " + ", ".join(metric_parts))
        else:
            print(f"  {task}: {metrics}")


def append_csv_result(
    path: Path,
    args: argparse.Namespace,
    config: PTQConfig | None,
    reference_result,
    quant_result,
    reference_zeroshot,
    quant_zeroshot,
    quantized_layers: int,
) -> None:
    row = {
        "run_name": args.run_name or "",
        "model_name": args.model_name,
        "num_bits": args.num_bits,
        "group_size": args.group_size,
        "block_size": args.block_size,
        "algorithm": args.algorithm,
        "awq_grid_size": args.awq_grid_size if uses_algorithm(args, "awq") else "",
        "gptaq_alpha": args.gptaq_alpha if uses_algorithm(args, "gptaq") else "",
        "qronos_alpha": args.qronos_alpha if uses_algorithm(args, "qronos") else "",
        "qep": uses_qep(args.algorithm, args.qep),
        "qep_alpha": args.qep_alpha if uses_qep(args.algorithm, args.qep) else "",
        "qep_dampening": args.qep_dampening if uses_qep(args.algorithm, args.qep) else "",
        "qep_skip_down_proj": args.qep_skip_down_proj if uses_qep(args.algorithm, args.qep) else "",
        "dampening": args.dampening,
        "grid_selection": "" if config is None else config.resolved_grid_selection(),
        "act_order": args.act_order,
        "local_block_replay": args.local_block_replay,
        "local_block_input_source": args.local_block_input_source if args.local_block_replay else "",
        "activation_quant": args.activation_quant,
        "activation_num_bits": args.activation_num_bits if args.activation_quant else "",
        "activation_symmetric": (not args.activation_asymmetric) if args.activation_quant else "",
        "activation_granularity": args.activation_granularity if args.activation_quant else "",
        "calibration_batches": args.calibration_batches,
        "calibration_offset_batches": args.calibration_offset_batches,
        "eval_batches": args.eval_batches,
        "sequence_length": args.sequence_length,
        "batch_size": args.batch_size,
        "eval_split": args.eval_split,
        "calibration_split": args.calibration_split,
        "quantized_layers": quantized_layers,
        "reference_ppl": "" if reference_result is None else reference_result.perplexity,
        "reference_mean_nll": "" if reference_result is None else reference_result.mean_nll,
        "quantized_ppl": "" if quant_result is None else quant_result.perplexity,
        "quantized_mean_nll": "" if quant_result is None else quant_result.mean_nll,
        "quantized_tokens": "" if quant_result is None else quant_result.num_tokens,
    }
    if reference_zeroshot is not None:
        row.update(_flatten_zeroshot_metrics("reference", reference_zeroshot))
    if quant_zeroshot is not None:
        row.update(_flatten_zeroshot_metrics("quantized", quant_zeroshot))

    path.parent.mkdir(parents=True, exist_ok=True)
    existing_rows: list[dict[str, str]] = []
    existing_fieldnames: list[str] = []
    if path.exists() and path.stat().st_size > 0:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            existing_fieldnames = list(reader.fieldnames or [])
            existing_rows = list(reader)

    fieldnames = list(existing_fieldnames)
    for fieldname in row:
        if fieldname not in fieldnames:
            fieldnames.append(fieldname)

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for existing_row in existing_rows:
            writer.writerow(existing_row)
        writer.writerow(row)
    print(f"Wrote CSV result: {path}")


def _flatten_zeroshot_metrics(prefix: str, result) -> dict[str, float]:
    flattened: dict[str, float] = {}
    for task in result.tasks:
        for metric_name, metric_value in result.results.get(task, {}).items():
            if metric_name.endswith("_stderr") or metric_name == "alias":
                continue
            if isinstance(metric_value, (int, float)):
                flattened[f"{prefix}_zeroshot_{task}_{metric_name}"] = metric_value
    return flattened


if __name__ == "__main__":
    main()
