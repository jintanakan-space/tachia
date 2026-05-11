from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from flax import nnx
from orbax import checkpoint as ocp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tachia import create_model
from tokenizer import PratchyaTokenizer
from train import ABCDigitsDataset


RUN_RE = re.compile(r"tachia-(?P<selector>[^-]+)-s(?P<slots>\d+)(?:-t(?P<temperature>[0-9.]+))?")
METRICS = ("entropy", "effective_context", "top_token_overlap", "pairwise_selector_cosine")


@dataclass(frozen=True)
class RunInfo:
    name: str
    selector: str
    slots: int
    selector_temperature: float
    vocab_size: int
    embed_dim: int
    num_layers: int
    num_heads: int


def parse_run_name(name: str) -> tuple[str, int, float] | None:
    match = RUN_RE.fullmatch(name)
    if match is None:
        return None
    temperature = float(match.group("temperature") or 1.0)
    return match.group("selector"), int(match.group("slots")), temperature


def checkpoint_steps(run_dir: Path) -> list[int]:
    steps = []
    for child in run_dir.iterdir():
        if child.is_dir() and child.name.isdigit() and (child / "default" / "_METADATA").exists():
            steps.append(int(child.name))
    return sorted(steps)


def _shape_for(metadata: dict[str, Any], needle: str) -> list[int]:
    for key, value in metadata["tree_metadata"].items():
        if needle in key:
            return value["value_metadata"]["write_shape"]
    raise ValueError(f"could not find {needle!r} in checkpoint metadata")


def infer_run_info(run_dir: Path) -> RunInfo:
    parsed = parse_run_name(run_dir.name)
    if parsed is None:
        raise ValueError(f"unsupported run name: {run_dir.name}")
    selector, slots, selector_temperature = parsed

    steps = checkpoint_steps(run_dir)
    if not steps:
        raise ValueError(f"no completed checkpoints found in {run_dir}")
    metadata_path = run_dir / str(steps[-1]) / "default" / "_METADATA"
    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    vocab_size, embed_dim = _shape_for(metadata, "'token_embedding', 'embedding'")[:2]
    num_layers, num_heads, head_dim, inferred_slots = _shape_for(metadata, "'k_slots'")
    if inferred_slots != slots:
        raise ValueError(f"run name says S={slots}, checkpoint says S={inferred_slots}")
    if embed_dim != num_heads * head_dim:
        raise ValueError(f"embed_dim={embed_dim} does not match heads={num_heads}, head_dim={head_dim}")

    return RunInfo(
        name=run_dir.name,
        selector=selector,
        slots=slots,
        selector_temperature=selector_temperature,
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
    )


def batch_examples(dataset: ABCDigitsDataset, start: int, batch_size: int) -> dict[str, np.ndarray]:
    examples = [dataset[(start + offset) % len(dataset)] for offset in range(batch_size)]
    keys = examples[0].keys()
    return {key: np.stack([example[key] for example in examples], axis=0) for key in keys}


def restore_model(info: RunInfo, checkpoint_dir: Path):
    model = create_model(
        vocab_size=info.vocab_size,
        embed_dim=info.embed_dim,
        num_layers=info.num_layers,
        num_heads=info.num_heads,
        num_slots=info.slots,
        selector_mode=info.selector,
        selector_temperature=info.selector_temperature,
    )
    _, state = nnx.split(model)
    restored = ocp.StandardCheckpointer().restore((checkpoint_dir / "default").resolve(), state)
    nnx.update(model, restored)
    return model


def _accumulate_metric(
    totals: dict[str, dict[str, float]],
    per_layer: dict[str, dict[str, np.ndarray]],
    side: str,
    metric: str,
    values,
    answer_mask,
) -> None:
    values = jnp.asarray(values)
    answer_mask = jnp.asarray(answer_mask, dtype=values.dtype)
    key = f"{side}_{metric}"

    if values.ndim == 5:
        mask = answer_mask[None, :, None, None, :]
        reduce_axes = (1, 2, 3, 4)
    elif values.ndim == 4:
        mask = answer_mask[None, :, None, :]
        reduce_axes = (1, 2, 3)
    else:
        raise ValueError(f"unexpected metric rank for {key}: {values.shape}")

    weighted = values * mask
    denom = jnp.sum(jnp.ones_like(values) * mask)
    totals.setdefault(key, {"sum": 0.0, "denom": 0.0})
    totals[key]["sum"] += float(jnp.sum(weighted))
    totals[key]["denom"] += float(denom)

    layer_sum = np.asarray(jnp.sum(weighted, axis=reduce_axes))
    layer_denom = np.asarray(jnp.sum(jnp.ones_like(values) * mask, axis=reduce_axes))
    per_layer.setdefault(key, {"sum": np.zeros_like(layer_sum), "denom": np.zeros_like(layer_denom)})
    per_layer[key]["sum"] += layer_sum
    per_layer[key]["denom"] += layer_denom


def evaluate_activation_metrics(
    model,
    dataset: ABCDigitsDataset,
    *,
    batch_size: int,
    eval_examples: int,
) -> dict[str, Any]:
    totals: dict[str, dict[str, float]] = {}
    per_layer: dict[str, dict[str, np.ndarray]] = {}

    for start in range(0, eval_examples, batch_size):
        batch = batch_examples(dataset, start, min(batch_size, eval_examples - start))
        _, stats = model(jnp.asarray(batch["input_ids"], dtype=jnp.int32), collect_stats=True)
        answer_mask = jnp.asarray(batch["answer_mask"], dtype=jnp.float32)

        for side, side_stats in stats.items():
            for metric in METRICS:
                _accumulate_metric(totals, per_layer, side, metric, side_stats[metric], answer_mask)

    overall = {
        key: value["sum"] / max(value["denom"], 1.0)
        for key, value in sorted(totals.items())
    }
    by_layer = {
        key: np.divide(
            value["sum"],
            np.maximum(value["denom"], 1.0),
        ).tolist()
        for key, value in sorted(per_layer.items())
    }
    return {"overall": overall, "per_layer": by_layer}


def collect_runs(input_dir: Path, selectors: set[str]) -> list[RunInfo]:
    runs = []
    for run_dir in sorted(input_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        parsed = parse_run_name(run_dir.name)
        if parsed is None:
            continue
        selector, _, _ = parsed
        if selector not in selectors:
            continue
        if not checkpoint_steps(run_dir):
            continue
        runs.append(infer_run_info(run_dir))
    return sorted(runs, key=lambda run: (run.selector, run.slots))


def write_plots(results: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_names = sorted({key for item in results for key in item["metrics"]["overall"]})
    for metric in metric_names:
        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
        for item in results:
            run = item["run"]
            checkpoints = item["checkpoints"]
            y = [checkpoint["metrics"]["overall"][metric] for checkpoint in checkpoints]
            ax.plot(
                [checkpoint["step"] for checkpoint in checkpoints],
                y,
                marker="o",
                linewidth=1.8,
                label=f"{run['selector']} S={run['slots']} T={run['selector_temperature']:g}",
            )
        ax.set_title(metric.replace("_", " ").title())
        ax.set_xlabel("Checkpoint step")
        ax.set_ylabel(metric.replace("_", " "))
        ax.grid(True, alpha=0.25)
        ax.legend(ncols=2, fontsize=9)
        fig.savefig(output_dir / f"{metric}.png", dpi=160)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Tachia selector activation metrics at checkpoints.")
    parser.add_argument("--input-dir", default="output/abcdigits_compare")
    parser.add_argument("--output-dir", default="output/abcdigits_compare/activation_metrics")
    parser.add_argument("--selectors", nargs="+", default=["sigmoid"])
    parser.add_argument("--eval-examples", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--context-equations", type=int, default=96)
    parser.add_argument("--seed", type=int, default=1_000_042)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = PratchyaTokenizer.from_pretrained("tokenizer")
    dataset = ABCDigitsDataset(
        tokenizer=tokenizer,
        num_examples=args.eval_examples,
        sequence_length=args.sequence_length,
        context_equations=args.context_equations,
        seed=args.seed,
    )

    runs = collect_runs(input_dir, set(args.selectors))
    if not runs:
        raise SystemExit(f"no completed Tachia checkpoints found under {input_dir}")

    results = []
    for run in runs:
        checkpoints = []
        for step in checkpoint_steps(input_dir / run.name):
            print(f"evaluating {run.name} step {step}", flush=True)
            model = restore_model(run, input_dir / run.name / str(step))
            metrics = evaluate_activation_metrics(
                model,
                dataset,
                batch_size=args.batch_size,
                eval_examples=args.eval_examples,
            )
            checkpoints.append({"step": step, "metrics": metrics})
        results.append(
            {
                "run": {
                    "name": run.name,
                    "selector": run.selector,
                    "slots": run.slots,
                    "selector_temperature": run.selector_temperature,
                    "vocab_size": run.vocab_size,
                    "embed_dim": run.embed_dim,
                    "num_layers": run.num_layers,
                    "num_heads": run.num_heads,
                },
                "metrics": checkpoints[-1]["metrics"],
                "checkpoints": checkpoints,
            }
        )

    metrics_path = output_dir / "activation_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, sort_keys=True)
    write_plots(results, output_dir / "plots")

    print(f"wrote {metrics_path}")
    print(f"wrote plots to {output_dir / 'plots'}")


if __name__ == "__main__":
    main()
