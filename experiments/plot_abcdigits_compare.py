from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


RUN_RE = re.compile(
    r"tachia-(?P<selector>[^-]+)-s(?P<slots>\d+)(?:-t(?P<temperature>[\d.]+))?|standard-e(?P<standard_dim>\d+)"
)


@dataclass(frozen=True)
class RunMetrics:
    name: str
    label: str
    selector: str
    slots: int | None
    steps: list[int]
    metrics: dict[str, list[float]]


def parse_run_name(name: str) -> tuple[str, int | None, str]:
    match = RUN_RE.fullmatch(name)
    if not match:
        return "unknown", None, name
    if match.group("standard_dim") is not None:
        dim = match.group("standard_dim")
        return "standard", None, f"standard e={dim}"
    selector = match.group("selector")
    slots = int(match.group("slots"))
    temperature = match.group("temperature")
    temperature_label = f" T={temperature}" if temperature is not None else ""
    return selector, slots, f"{selector} S={slots}{temperature_label}"


def load_runs(input_dir: Path) -> list[RunMetrics]:
    runs = []
    for metrics_path in sorted(input_dir.glob("*/training_metrics.json")):
        with metrics_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        step_data = data.get("steps") or {}
        if not step_data:
            continue

        name = metrics_path.parent.name
        selector, slots, label = parse_run_name(name)
        steps = sorted(int(step) for step in step_data)
        metric_names = sorted({key for step in step_data.values() for key in step})
        metrics = {
            metric_name: [float(step_data[str(step)][metric_name]) for step in steps if metric_name in step_data[str(step)]]
            for metric_name in metric_names
        }
        runs.append(
            RunMetrics(
                name=name,
                label=label,
                selector=selector,
                slots=slots,
                steps=steps,
                metrics=metrics,
            )
        )

    return sorted(runs, key=lambda run: (run.selector != "sigmoid", run.selector, run.slots or 0, run.name))


def metric_style(run: RunMetrics) -> dict[str, object]:
    marker = "o"
    if run.selector == "softmax":
        marker = "s"
    elif run.selector == "standard":
        marker = "^"
    return {"marker": marker, "linewidth": 1.8, "markersize": 4}


def plot_metric(runs: list[RunMetrics], metric: str, output_path: Path) -> bool:
    usable = [run for run in runs if metric in run.metrics]
    if not usable:
        return False

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for run in usable:
        values = run.metrics[metric]
        ax.plot(run.steps[: len(values)], values, label=run.label, **metric_style(run))

    ax.set_title(metric.replace("_", " ").title())
    ax.set_xlabel("Step")
    ax.set_ylabel(metric.replace("_", " "))
    ax.grid(True, alpha=0.25)
    ax.legend(ncols=2, fontsize=9)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return True


def plot_summary(runs: list[RunMetrics], metrics: list[str], output_path: Path) -> None:
    available = [metric for metric in metrics if any(metric in run.metrics for run in runs)]
    if not available:
        return

    rows = (len(available) + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(14, 4.5 * rows), constrained_layout=True)
    axes_list = list(axes.flat if hasattr(axes, "flat") else [axes])

    for ax, metric in zip(axes_list, available):
        for run in runs:
            if metric not in run.metrics:
                continue
            values = run.metrics[metric]
            ax.plot(run.steps[: len(values)], values, label=run.label, **metric_style(run))
        ax.set_title(metric.replace("_", " ").title())
        ax.set_xlabel("Step")
        ax.set_ylabel(metric.replace("_", " "))
        ax.grid(True, alpha=0.25)

    for ax in axes_list[len(available) :]:
        ax.axis("off")

    handles, labels = axes_list[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=4, fontsize=9)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot ABCDigits comparison training metrics.")
    parser.add_argument("--input-dir", default="output/abcdigits_compare")
    parser.add_argument("--output-dir", default="output/abcdigits_compare/plots")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=[
            "loss",
            "lm_loss",
            "prompt_lm_loss",
            "answer_lm_loss",
            "accuracy",
            "answer_accuracy",
            "exact_answer_accuracy",
            "grad_norm",
            "steps_per_sec",
            "slot_diversity_loss",
            "learning_rate",
        ],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = load_runs(input_dir)
    if not runs:
        raise SystemExit(f"no training_metrics.json files found under {input_dir}")

    written = []
    for metric in args.metrics:
        output_path = output_dir / f"{metric}.png"
        if plot_metric(runs, metric, output_path):
            written.append(output_path)

    summary_path = output_dir / "summary.png"
    plot_summary(runs, args.metrics, summary_path)
    if summary_path.exists():
        written.insert(0, summary_path)

    print("loaded runs:")
    for run in runs:
        print(f"  {run.label}: {len(run.steps)} checkpoints, last_step={run.steps[-1]}")
    print("wrote plots:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
