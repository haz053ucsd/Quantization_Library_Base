from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_SCRIPT = REPO_ROOT / "scripts" / "evaluate_qwen_wikitext.py"


@dataclass(frozen=True)
class Experiment:
    name: str
    algorithm: str
    dampening: float | None = None
    qronos_alpha: float | None = None
    act_order: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Qwen3-0.6B 4-bit weight-only PTQ algorithm suite."
    )
    parser.add_argument("--model-name", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--csv-output", default="results/qwen06b_4bit_weightonly_suite.csv")
    parser.add_argument("--dataset-name", default="wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--calibration-split", default="train")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--calibration-batches", type=int, default=1)
    parser.add_argument("--eval-batches", type=int, default=2)
    parser.add_argument("--calibration-offset-batches", type=int, default=0)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    parser.add_argument("--local-block-replay", action="store_true")
    parser.add_argument("--local-block-input-source", choices=("reference", "quantized"), default="reference")
    parser.add_argument("--qep-skip-down-proj", action="store_true", default=True)
    parser.add_argument("--no-qep-skip-down-proj", dest="qep_skip_down_proj", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiments = build_experiments(args.model_name)
    for index, experiment in enumerate(experiments, start=1):
        command = build_command(args, experiment)
        print(f"\n=== [{index}/{len(experiments)}] {experiment.name} ===", flush=True)
        print(" ".join(command), flush=True)
        if args.dry_run:
            continue
        subprocess.run(command, cwd=REPO_ROOT, check=True)


def build_experiments(model_name: str) -> list[Experiment]:
    model_alias = normalize_model_alias(model_name)
    return [
        Experiment(
            name=f"{model_alias}_4bit_rtn",
            algorithm="rtn",
        ),
        Experiment(
            name=f"{model_alias}_4bit_gptq_damp0p1_actorder",
            algorithm="gptq",
            dampening=0.1,
            act_order=True,
        ),
        Experiment(
            name=f"{model_alias}_4bit_gptaq_alpha0p25_actorder",
            algorithm="gptaq",
            act_order=True,
        ),
        Experiment(
            name=f"{model_alias}_4bit_qronos_alpha1em4_actorder",
            algorithm="qronos",
            qronos_alpha=1e-4,
            act_order=True,
        ),
        # # Experiment(
        #     name=f"{model_alias}_4bit_awq",
        #     algorithm="awq",
        # ),
        Experiment(
            name=f"{model_alias}_4bit_qep_gptq_damp0p1_actorder",
            algorithm="qep-gptq",
            dampening=0.1,
            act_order=True,
        ),
    ]


def normalize_model_alias(model_name: str) -> str:
    model_alias = model_name.rsplit("/", 1)[-1].lower()
    return model_alias.replace("-", "_").replace(".", "p")


def build_command(args: argparse.Namespace, experiment: Experiment) -> list[str]:
    command = [
        sys.executable,
        str(EVAL_SCRIPT),
        "--run-name",
        experiment.name,
        "--model-name",
        args.model_name,
        "--dataset-name",
        args.dataset_name,
        "--dataset-config",
        args.dataset_config,
        "--calibration-split",
        args.calibration_split,
        "--eval-split",
        args.eval_split,
        "--sequence-length",
        str(args.sequence_length),
        "--batch-size",
        str(args.batch_size),
        "--calibration-batches",
        str(args.calibration_batches),
        "--calibration-offset-batches",
        str(args.calibration_offset_batches),
        "--eval-batches",
        str(args.eval_batches),
        "--num-bits",
        "4",
        "--group-size",
        str(args.group_size),
        "--algorithm",
        experiment.algorithm,
        "--grid-selection",
        "minmax",
        "--gptaq-alpha",
        "0.25",
        "--awq-grid-size",
        "20",
        "--qep-alpha",
        "0.5",
        "--qep-dampening",
        "1.0",
        "--csv-output",
        args.csv_output,
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--skip-reference-eval",
    ]
    if experiment.dampening is not None:
        command.extend(["--dampening", str(experiment.dampening)])
    if experiment.qronos_alpha is not None:
        command.extend(["--qronos-alpha", str(experiment.qronos_alpha)])
    if experiment.act_order:
        command.append("--act-order")
    if args.local_block_replay:
        command.append("--local-block-replay")
        command.extend(["--local-block-input-source", args.local_block_input_source])
    if uses_qep_experiment(experiment):
        if args.qep_skip_down_proj:
            command.append("--qep-skip-down-proj")
        else:
            command.append("--no-qep-skip-down-proj")
    if args.trust_remote_code:
        command.append("--trust-remote-code")
    return command


def uses_qep_experiment(experiment: Experiment) -> bool:
    return experiment.algorithm.startswith("qep-")


if __name__ == "__main__":
    main()
