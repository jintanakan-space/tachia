from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from flax import nnx
from zlynx.trainer import Trainer, TrainerConfig

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tachia import create_model
from tokenizer import PratchyaTokenizer
from train import ABCDigitsDataset, loss_fn


class RepeatDataset:
    def __init__(self, dataset: ABCDigitsDataset, length: int):
        self.dataset = dataset
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        return self.dataset[index % len(self.dataset)]


def parameter_count(model: nnx.Module) -> int:
    return int(sum(variable[...].size for _, variable in nnx.to_flat_state(nnx.state(model, nnx.Param))))


def batch_examples(dataset: ABCDigitsDataset, start: int, batch_size: int) -> dict[str, np.ndarray]:
    examples = [dataset[(start + offset) % len(dataset)] for offset in range(batch_size)]
    return {key: np.stack([example[key] for example in examples], axis=0) for key in examples[0]}


def evaluate(model: nnx.Module, dataset: ABCDigitsDataset, *, batch_size: int) -> dict[str, float]:
    totals: dict[str, float] = {}
    batches = 0
    for start in range(0, len(dataset), batch_size):
        batch = batch_examples(dataset, start, min(batch_size, len(dataset) - start))
        metrics = loss_fn(model, batch)
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value)
        batches += 1
    return {key: value / batches for key, value in totals.items()}


def collect_activation_snapshot(model: nnx.Module, dataset: ABCDigitsDataset, *, batch_size: int) -> dict[str, Any]:
    batch = batch_examples(dataset, 0, min(batch_size, len(dataset)))
    _, stats = model(np.asarray(batch["input_ids"], dtype=np.int32), collect_stats=True)
    snapshot: dict[str, Any] = {}
    for side in ("key", "value"):
        snapshot[side] = {
            "entropy": float(np.asarray(stats[side]["entropy"]).mean()),
            "effective_context": float(np.asarray(stats[side]["effective_context"]).mean()),
            "top_token_overlap": float(np.asarray(stats[side]["top_token_overlap"]).mean()),
            "pairwise_selector_cosine": float(np.asarray(stats[side]["pairwise_selector_cosine"]).mean()),
        }
    return snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overfit check for Tachia before expensive TPU runs.")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--train-examples", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--context-equations", type=int, default=96)
    parser.add_argument("--target-depth", type=float, default=0.5)
    parser.add_argument("--embed-dim", type=int, default=112)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--slots", type=int, default=32)
    parser.add_argument("--selector-mode", choices=["sigmoid", "softmax"], default="sigmoid")
    parser.add_argument("--selector-temperature", type=float, default=4.0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--warmup-ratio", type=float, default=0.02)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="./output/debug_tiny_overfit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = PratchyaTokenizer.from_pretrained("tokenizer")
    model = create_model(
        vocab_size=tokenizer.vocab_size,
        embed_dim=args.embed_dim,
        num_layers=args.layers,
        num_heads=args.heads,
        num_slots=args.slots,
        selector_mode=args.selector_mode,
        selector_temperature=args.selector_temperature,
        seed=args.seed,
    )
    dataset = ABCDigitsDataset(
        tokenizer=tokenizer,
        num_examples=args.train_examples,
        sequence_length=args.sequence_length,
        context_equations=args.context_equations,
        target_depth=args.target_depth,
        seed=args.seed,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    before = evaluate(model, dataset, batch_size=args.batch_size)
    before_activations = collect_activation_snapshot(model, dataset, batch_size=args.batch_size)

    repeated_dataset = RepeatDataset(dataset, max(args.train_examples, args.steps * args.batch_size))
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        train_dataset=repeated_dataset,
        config=TrainerConfig(
            batch_size=args.batch_size,
            max_steps=args.steps,
            logging_steps=args.logging_steps,
            learning_rate=args.learning_rate,
            warmup_ratio=args.warmup_ratio,
            output_dir=str(output_dir / "checkpoint"),
            num_workers=0,
            sharding=None,
        ),
    )
    trainer.train()

    after = evaluate(model, dataset, batch_size=args.batch_size)
    after_activations = collect_activation_snapshot(model, dataset, batch_size=args.batch_size)
    result = {
        "parameters": parameter_count(model),
        "config": vars(args),
        "before": before,
        "after": after,
        "before_activations": before_activations,
        "after_activations": after_activations,
    }
    results_path = output_dir / "result.json"
    results_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    print(f"saved -> {results_path}")


if __name__ == "__main__":
    main()
