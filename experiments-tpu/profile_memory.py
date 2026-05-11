from __future__ import annotations

import argparse
import math
from dataclasses import dataclass


DTYPE_BYTES = {
    "bf16": 2,
    "f16": 2,
    "f32": 4,
}


@dataclass(frozen=True)
class TensorEstimate:
    module: str
    name: str
    shape: tuple[int, ...]
    dtype: str = "f32"
    copies: int = 1

    @property
    def bytes(self) -> int:
        return math.prod(self.shape) * DTYPE_BYTES[self.dtype] * self.copies


def fmt_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    amount = float(value)
    for unit in units:
        if amount < 1024.0 or unit == units[-1]:
            return f"{amount:.2f} {unit}"
        amount /= 1024.0
    raise AssertionError("unreachable")


def parameter_estimates(args: argparse.Namespace) -> list[TensorEstimate]:
    e = args.embed_dim
    h = args.heads
    d = e // h
    s = args.slots
    m = args.mlp_hidden_dim or e * 4
    l = args.layers
    vocab = args.vocab_size

    return [
        TensorEstimate("embedding", "token_embedding", (vocab, e), args.param_dtype),
        TensorEstimate("attention/layer", "qkv_out_proj kernels", (4, e, e), args.param_dtype, copies=l),
        TensorEstimate("attention/layer", "k/v slot params", (2, h, d, s), args.param_dtype, copies=l),
        TensorEstimate("mlp/layer", "gate/up/down kernels", (2 * e * m + m * e,), args.param_dtype, copies=l),
        TensorEstimate("norm/layer", "rms scales", (2, e), args.param_dtype, copies=l),
        TensorEstimate("norm/final", "rms scale", (e,), args.param_dtype),
    ]


def active_activation_estimates(args: argparse.Namespace) -> list[TensorEstimate]:
    b = args.batch_size
    t = args.sequence_length
    e = args.embed_dim
    h = args.heads
    d = e // h
    s = args.slots
    l = args.layers if args.include_layer_axis else 1
    m = args.mlp_hidden_dim or e * 4
    dtype = args.activation_dtype

    prefix = (l,) if args.include_layer_axis else ()
    return [
        TensorEstimate("embedding", "hidden state", prefix + (b, t, e), dtype),
        TensorEstimate("attention/layer", "q/k/v projections", prefix + (b, t, h, d), dtype, copies=3),
        TensorEstimate("attention/layer", "selector logits k/v", prefix + (b, h, s, t), "f32", copies=2),
        TensorEstimate("attention/layer", "FUSED scan cumulative k/v", prefix + (b, h, s, d), "f32", copies=2),
        TensorEstimate("attention/layer", "FUSED scan slot logits", prefix + (b, h, s), "f32"),
        TensorEstimate("attention/layer", "NEW slot attention output", prefix + (b, t, h, d), dtype),
        TensorEstimate("mlp/layer", "gate/up activations", prefix + (b, t, m), dtype, copies=2),
        TensorEstimate("mlp/layer", "gated hidden", prefix + (b, t, m), dtype),
        TensorEstimate("lm_head", "logits", (b, t, args.vocab_size), "f32"),
    ]


def old_unfused_reference_estimates(args: argparse.Namespace) -> list[TensorEstimate]:
    b = args.batch_size
    t = args.sequence_length
    e = args.embed_dim
    h = args.heads
    d = e // h
    s = args.slots
    l = args.layers if args.include_layer_axis else 1
    prefix = (l,) if args.include_layer_axis else ()
    return [
        TensorEstimate("attention/layer", "OLD prefix compressed k/v slots", prefix + (b, t, s, h, d), "f32", copies=2),
        TensorEstimate("attention/layer", "OLD nnx dpa broadcast temp", prefix + (b, h, s, t, d), "f32", copies=4),
        TensorEstimate("attention/layer", "unfused slot attention logits", prefix + (b, t, h, s), "f32"),
    ]


def print_table(title: str, estimates: list[TensorEstimate]) -> None:
    print(title)
    print("-" * len(title))
    total = 0
    for item in estimates:
        total += item.bytes
        shape = "x".join(str(dim) for dim in item.shape)
        copies = f" x{item.copies}" if item.copies != 1 else ""
        print(f"{item.module:18} {item.name:32} {item.dtype:4} {shape:34} {fmt_bytes(item.bytes)}{copies}")
    print(f"{'total':18} {'':32} {'':4} {'':34} {fmt_bytes(total)}")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate Tachia TPU memory by module tensor shapes.")
    parser.add_argument("--vocab-size", type=int, default=82369)
    parser.add_argument("--embed-dim", type=int, default=1408)
    parser.add_argument("--layers", type=int, default=24)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--slots", type=int, default=128)
    parser.add_argument("--mlp-hidden-dim", type=int, default=None)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--param-dtype", choices=sorted(DTYPE_BYTES), default="bf16")
    parser.add_argument("--activation-dtype", choices=sorted(DTYPE_BYTES), default="bf16")
    parser.add_argument(
        "--include-layer-axis",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Match XLA reports from scanned/vmapped blocks that show a leading layer axis.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.embed_dim % args.heads != 0:
        raise SystemExit("--embed-dim must be divisible by --heads")

    print(
        "config: "
        f"L={args.layers}, B={args.batch_size}, T={args.sequence_length}, "
        f"E={args.embed_dim}, H={args.heads}, D={args.embed_dim // args.heads}, S={args.slots}"
    )
    print()
    print_table("Parameter Estimates", parameter_estimates(args))
    active = active_activation_estimates(args)
    old = old_unfused_reference_estimates(args)
    print_table("Active Fused Training Path Estimates", active)
    print_table("Old Unfused Reference Estimates", old)
    old_dpa = next(item for item in old if item.name == "OLD nnx dpa broadcast temp")
    old_prefix = next(item for item in old if item.name == "OLD prefix compressed k/v slots")
    fused_state = next(item for item in active if item.name == "FUSED scan cumulative k/v")
    fused_logits = next(item for item in active if item.name == "FUSED scan slot logits")
    print(f"old prefix compressed k/v temp : {fmt_bytes(old_prefix.bytes)}")
    print(f"old final attention broadcast temp: {fmt_bytes(old_dpa.bytes)}")
    print(f"fused scan cumulative k/v temp : {fmt_bytes(fused_state.bytes)}")
    print(f"fused scan slot logits temp     : {fmt_bytes(fused_logits.bytes)}")
    print(f"final attention estimated reduction: {old_dpa.bytes / max(fused_logits.bytes, 1):.1f}x")
    print(f"prefix state estimated reduction   : {old_prefix.bytes / max(fused_state.bytes, 1):.1f}x")


if __name__ == "__main__":
    main()
