from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "experiments-tpu" / "train_lm.py"


@dataclass(frozen=True)
class ModelSize:
    name: str
    embed_dim: int
    layers: int
    heads: int


MODEL_SIZES = [
    ModelSize("e1408-l24-h16", embed_dim=1408, layers=24, heads=16),
    ModelSize("e1536-l22-h16", embed_dim=1536, layers=22, heads=16),
    ModelSize("e1536-l24-h16", embed_dim=1536, layers=24, heads=16),
]


def shell_join(command: list[str]) -> str:
    return " ".join(command)


def base_command(args: argparse.Namespace, output_dir: Path) -> list[str]:
    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--sequence-length",
        str(args.sequence_length),
        "--steps",
        str(args.steps),
        "--batch-size",
        str(args.batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--warmup-ratio",
        str(args.warmup_ratio),
        "--logging-steps",
        str(args.logging_steps),
        "--num-workers",
        str(args.num_workers),
        "--seed",
        str(args.seed),
        "--output-dir",
        str(output_dir),
    ]
    if args.data_path is not None:
        command.extend(["--data-path", *[str(path) for path in args.data_path]])
    if args.hf_repo is not None:
        command.extend(["--hf-repo", args.hf_repo])
    if args.hf_config is not None:
        command.extend(["--hf-config", args.hf_config])
    command.extend(["--hf-split", args.hf_split])
    if args.hf_streaming:
        command.append("--hf-streaming")
    if args.token_cache is not None:
        command.extend(["--token-cache", str(args.token_cache)])
    if args.text_field is not None:
        command.extend(["--text-field", args.text_field])
    if args.max_records is not None:
        command.extend(["--max-records", str(args.max_records)])
    if args.train_examples is not None:
        command.extend(["--train-examples", str(args.train_examples)])
    if not args.shuffle_offsets:
        command.append("--no-shuffle-offsets")
    return command


def command_for_run(
    args: argparse.Namespace,
    *,
    run_name: str,
    embed_dim: int,
    layers: int,
    heads: int,
    slots: int,
    temperature: float,
) -> list[str]:
    output_dir = Path(args.output_root) / args.stage / run_name
    command = base_command(args, output_dir)
    command.extend(
        [
            "--embed-dim",
            str(embed_dim),
            "--layers",
            str(layers),
            "--heads",
            str(heads),
            "--slots",
            str(slots),
            "--selector-temperature",
            str(temperature),
            "--selector-mode",
            "sigmoid",
        ]
    )
    return command


def build_commands(args: argparse.Namespace) -> list[list[str]]:
    if args.stage == "model-size":
        return [
            command_for_run(
                args,
                run_name=f"{size.name}-s{args.base_slots}-t{args.base_temperature:g}",
                embed_dim=size.embed_dim,
                layers=size.layers,
                heads=size.heads,
                slots=args.base_slots,
                temperature=args.base_temperature,
            )
            for size in MODEL_SIZES
        ]

    if args.stage == "slots":
        return [
            command_for_run(
                args,
                run_name=f"{args.model_name}-s{slots}-t{args.base_temperature:g}",
                embed_dim=args.embed_dim,
                layers=args.layers,
                heads=args.heads,
                slots=slots,
                temperature=args.base_temperature,
            )
            for slots in args.slots
        ]

    if args.stage == "temperature":
        return [
            command_for_run(
                args,
                run_name=f"{args.model_name}-s{args.base_slots}-t{temperature:g}",
                embed_dim=args.embed_dim,
                layers=args.layers,
                heads=args.heads,
                slots=args.base_slots,
                temperature=temperature,
            )
            for temperature in args.temperatures
        ]

    raise ValueError(f"unknown stage: {args.stage}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run staged TPU Tachia LM experiments.")
    parser.add_argument("--stage", choices=["model-size", "slots", "temperature"], required=True)
    parser.add_argument("--data-path", type=Path, nargs="+", default=None)
    parser.add_argument("--hf-repo", default=None)
    parser.add_argument("--hf-config", default=None)
    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--hf-streaming", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--token-cache", type=Path, default=None)
    parser.add_argument("--text-field", default=None)
    parser.add_argument("--jsonl-field", default=None, help="Deprecated alias for --text-field.")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--train-examples", type=int, default=None)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--steps", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", default="./output/tpu_lm")
    parser.add_argument("--shuffle-offsets", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--base-slots", type=int, default=128)
    parser.add_argument("--base-temperature", type=float, default=4.0)
    parser.add_argument("--slots", type=int, nargs="+", default=[64, 128, 256, 512])
    parser.add_argument("--temperatures", type=float, nargs="+", default=[2.0, 4.0, 8.0, 16.0])

    parser.add_argument("--model-name", default="e1536-l24-h16")
    parser.add_argument("--embed-dim", type=int, default=1536)
    parser.add_argument("--layers", type=int, default=24)
    parser.add_argument("--heads", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.text_field is None and args.jsonl_field is not None:
        args.text_field = args.jsonl_field
    if args.data_path is None and args.hf_repo is None:
        raise SystemExit("provide either --data-path or --hf-repo")
    commands = build_commands(args)
    for command in commands:
        print(shell_join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
