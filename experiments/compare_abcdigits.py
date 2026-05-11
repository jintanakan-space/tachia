from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from flax import nnx
from zlynx.trainer import Trainer, TrainerConfig

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tachia import create_model, create_standard_transformer
from tokenizer import PratchyaTokenizer
from train import ABCDigitsDataset, loss_fn


@dataclass(frozen=True)
class RunSpec:
    name: str
    model_type: str
    embed_dim: int
    num_layers: int
    num_heads: int
    num_slots: int | None = None
    selector_mode: str | None = None
    selector_temperature: float | None = None


def parameter_count(model: nnx.Module) -> int:
    return int(sum(variable[...].size for _, variable in nnx.to_flat_state(nnx.state(model, nnx.Param))))


def make_model(spec: RunSpec, vocab_size: int, seed: int) -> nnx.Module:
    if spec.model_type == "tachia":
        assert spec.num_slots is not None
        assert spec.selector_mode is not None
        return create_model(
            vocab_size=vocab_size,
            embed_dim=spec.embed_dim,
            num_layers=spec.num_layers,
            num_heads=spec.num_heads,
            num_slots=spec.num_slots,
            selector_mode=spec.selector_mode,
            selector_temperature=spec.selector_temperature or 1.0,
            seed=seed,
        )

    if spec.model_type == "standard":
        return create_standard_transformer(
            vocab_size=vocab_size,
            embed_dim=spec.embed_dim,
            num_layers=spec.num_layers,
            num_heads=spec.num_heads,
            seed=seed,
        )

    raise ValueError(f"unknown model_type: {spec.model_type}")


def nearest_standard_embed_dim(
    *,
    target_params: int,
    vocab_size: int,
    num_layers: int,
    num_heads: int,
    seed: int,
    min_dim: int = 16,
    max_dim: int = 1024,
) -> int:
    candidates = []
    step = num_heads * 2
    for embed_dim in range(min_dim, max_dim + 1, step):
        if embed_dim % num_heads != 0 or (embed_dim // num_heads) % 2 != 0:
            continue
        model = create_standard_transformer(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            seed=seed,
        )
        candidates.append((abs(parameter_count(model) - target_params), embed_dim))
    if not candidates:
        raise ValueError("no valid standard Transformer embed_dim candidates")
    return min(candidates)[1]


def batch_examples(dataset: ABCDigitsDataset, start: int, batch_size: int) -> dict[str, np.ndarray]:
    examples = [dataset[(start + offset) % len(dataset)] for offset in range(batch_size)]
    keys = examples[0].keys()
    return {key: np.stack([example[key] for example in examples], axis=0) for key in keys}


def evaluate(model: nnx.Module, dataset: ABCDigitsDataset, *, batch_size: int, max_examples: int) -> dict[str, float]:
    totals: dict[str, float] = {}
    batches = 0
    for start in range(0, max_examples, batch_size):
        batch = batch_examples(dataset, start, min(batch_size, max_examples - start))
        metrics = loss_fn(model, batch)
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value)
        batches += 1
    return {key: value / batches for key, value in totals.items()}


def run_spec(
    spec: RunSpec,
    *,
    tokenizer: PratchyaTokenizer,
    train_dataset: ABCDigitsDataset,
    eval_dataset: ABCDigitsDataset,
    args: argparse.Namespace,
) -> dict[str, Any]:
    model = make_model(spec, tokenizer.vocab_size, args.seed)
    params = parameter_count(model)
    output_dir = Path(args.output_dir) / spec.name

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        train_dataset=train_dataset,
        config=TrainerConfig(
            batch_size=args.batch_size,
            max_steps=args.steps,
            logging_steps=args.logging_steps,
            learning_rate=args.learning_rate,
            warmup_ratio=args.warmup_ratio,
            output_dir=str(output_dir),
            num_workers=args.num_workers,
            sharding=None,
        ),
    )
    trainer.train()
    eval_metrics = evaluate(model, eval_dataset, batch_size=args.eval_batch_size, max_examples=args.eval_examples)

    return {
        "spec": asdict(spec),
        "parameters": params,
        "eval": eval_metrics,
    }


def build_specs(args: argparse.Namespace, vocab_size: int) -> list[RunSpec]:
    specs = []
    target_params = None
    if args.match_standard_params:
        target = create_model(
            vocab_size=vocab_size,
            embed_dim=args.embed_dim,
            num_layers=args.layers,
            num_heads=args.heads,
            num_slots=max(args.slots),
            selector_mode=args.selector_modes[0],
            selector_temperature=args.selector_temperatures[0],
            seed=args.seed,
        )
        target_params = parameter_count(target)

    for selector_mode in args.selector_modes:
        for selector_temperature in args.selector_temperatures:
            for slots in args.slots:
                temperature_name = f"-t{selector_temperature:g}" if selector_temperature != 1.0 else ""
                specs.append(
                    RunSpec(
                        name=f"tachia-{selector_mode}-s{slots}{temperature_name}",
                        model_type="tachia",
                        embed_dim=args.embed_dim,
                        num_layers=args.layers,
                        num_heads=args.heads,
                        num_slots=slots,
                        selector_mode=selector_mode,
                        selector_temperature=selector_temperature,
                    )
                )

    if args.include_standard:
        standard_embed_dim = args.standard_embed_dim or args.embed_dim
        if args.match_standard_params:
            assert target_params is not None
            standard_embed_dim = nearest_standard_embed_dim(
                target_params=target_params,
                vocab_size=vocab_size,
                num_layers=args.layers,
                num_heads=args.heads,
                seed=args.seed,
                min_dim=args.heads * 2,
                max_dim=max(args.embed_dim * 4, args.heads * 2),
            )
        specs.append(
            RunSpec(
                name=f"standard-e{standard_embed_dim}",
                model_type="standard",
                embed_dim=standard_embed_dim,
                num_layers=args.layers,
                num_heads=args.heads,
            )
        )

    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Tachia selector modes, slot counts, and a standard baseline.")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--train-examples", type=int, default=1_000)
    parser.add_argument("--eval-examples", type=int, default=128)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--context-equations", type=int, default=96)
    parser.add_argument("--embed-dim", type=int, default=112)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--slots", type=int, nargs="+", default=[16, 32, 64, 128])
    parser.add_argument("--selector-modes", nargs="+", choices=["sigmoid", "softmax"], default=["sigmoid", "softmax"])
    parser.add_argument("--selector-temperatures", type=float, nargs="+", default=[2.0])
    parser.add_argument("--include-standard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--standard-embed-dim", type=int, default=None)
    parser.add_argument("--match-standard-params", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="./output/abcdigits_compare")
    parser.add_argument("--results-path", default="./output/abcdigits_compare/results.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = PratchyaTokenizer.from_pretrained("tokenizer")
    train_dataset = ABCDigitsDataset(
        tokenizer=tokenizer,
        num_examples=args.train_examples,
        sequence_length=args.sequence_length,
        context_equations=args.context_equations,
        seed=args.seed,
    )
    eval_dataset = ABCDigitsDataset(
        tokenizer=tokenizer,
        num_examples=args.eval_examples,
        sequence_length=args.sequence_length,
        context_equations=args.context_equations,
        seed=args.seed + 1_000_000,
    )
    specs = build_specs(args, tokenizer.vocab_size)

    results_path = Path(args.results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", encoding="utf-8") as result_file:
        for spec in specs:
            result = run_spec(
                spec,
                tokenizer=tokenizer,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                args=args,
            )
            line = json.dumps(result, sort_keys=True)
            print(line)
            result_file.write(line + "\n")
            result_file.flush()


if __name__ == "__main__":
    main()
