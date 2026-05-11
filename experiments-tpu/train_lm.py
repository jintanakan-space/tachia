from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Iterator

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from zlynx.trainer import Trainer, TrainerConfig

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tachia import create_model
from tokenizer import PratchyaTokenizer
from train import SLOT_DIVERSITY_WEIGHT, slot_parameter_diversity_loss


IGNORE_INDEX = -100


class TokenLMDataset:
    def __init__(
        self,
        tokens: np.ndarray,
        *,
        sequence_length: int,
        num_examples: int | None = None,
        seed: int = 42,
        shuffle_offsets: bool = True,
    ):
        if tokens.ndim != 1:
            raise ValueError("tokens must be a 1D array")
        if tokens.shape[0] <= sequence_length:
            raise ValueError("not enough tokens for the requested sequence_length")

        self.tokens = tokens.astype(np.int32, copy=False)
        self.sequence_length = sequence_length
        self.max_offset = tokens.shape[0] - sequence_length - 1
        natural_examples = max(1, self.max_offset // sequence_length)
        self.num_examples = num_examples or natural_examples

        if shuffle_offsets:
            rng = np.random.default_rng(seed)
            self.offsets = rng.integers(0, self.max_offset + 1, size=self.num_examples, dtype=np.int64)
        else:
            self.offsets = (np.arange(self.num_examples, dtype=np.int64) * sequence_length) % (self.max_offset + 1)

    def __len__(self) -> int:
        return self.num_examples

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        offset = int(self.offsets[index % self.num_examples])
        input_ids = self.tokens[offset : offset + self.sequence_length]
        labels = self.tokens[offset + 1 : offset + self.sequence_length + 1]
        return {
            "input_ids": input_ids.astype(np.int32, copy=False),
            "labels": labels.astype(np.int32, copy=False),
            "loss_mask": np.ones((self.sequence_length,), dtype=np.float32),
            "answer_mask": np.zeros((self.sequence_length,), dtype=np.float32),
        }


def iter_text_records(paths: list[Path], text_field: str | None) -> Iterable[str]:
    for path in paths:
        if path.suffix == ".jsonl":
            field = text_field or "text"
            with path.open("r", encoding="utf-8") as file:
                for line in file:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    value = record.get(field)
                    if isinstance(value, str) and value:
                        yield value
            continue

        with path.open("r", encoding="utf-8") as file:
            yield file.read()


def iter_hf_text_records(
    *,
    repo: str,
    config: str | None,
    split: str,
    text_field: str | None,
    streaming: bool,
) -> Iterator[str]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Hugging Face datasets support requires `datasets`. Install it before using --hf-repo.") from exc

    kwargs = {"path": repo, "split": split, "streaming": streaming}
    if config is not None:
        kwargs["name"] = config
    dataset = load_dataset(**kwargs)
    field = text_field or "text"
    for record in dataset:
        value = record.get(field)
        if isinstance(value, str) and value:
            yield value


def load_or_build_tokens(
    *,
    tokenizer: PratchyaTokenizer,
    data_paths: list[Path] | None,
    hf_repo: str | None,
    hf_config: str | None,
    hf_split: str,
    hf_streaming: bool,
    token_cache: Path | None,
    text_field: str | None,
    max_records: int | None,
) -> np.ndarray:
    if token_cache is not None and token_cache.exists():
        return np.load(token_cache, mmap_mode="r")

    if data_paths is not None and len(data_paths) == 1 and data_paths[0].suffix == ".npy":
        tokens = np.load(data_paths[0], mmap_mode="r")
        if token_cache is not None and token_cache != data_paths[0]:
            np.save(token_cache, np.asarray(tokens, dtype=np.int32))
        return tokens

    if hf_repo is not None:
        text_records = iter_hf_text_records(
            repo=hf_repo,
            config=hf_config,
            split=hf_split,
            text_field=text_field,
            streaming=hf_streaming,
        )
    elif data_paths is not None:
        text_records = iter_text_records(data_paths, text_field)
    else:
        raise ValueError("provide either --data-path or --hf-repo")

    all_tokens: list[int] = []
    eos = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
    for index, text in enumerate(text_records):
        if max_records is not None and index >= max_records:
            break
        all_tokens.extend(tokenizer.encode(text, add_special_tokens=False))
        all_tokens.extend(eos)

    tokens = np.asarray(all_tokens, dtype=np.int32)
    if token_cache is not None:
        token_cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(token_cache, tokens)
    return tokens


def lm_loss_fn(model, batch: dict[str, jax.Array]) -> dict[str, jax.Array]:
    input_ids = jnp.asarray(batch["input_ids"], dtype=jnp.int32)
    labels = jnp.asarray(batch["labels"], dtype=jnp.int32)
    loss_mask = jnp.asarray(batch["loss_mask"], dtype=jnp.float32)

    logits = model(input_ids)
    safe_labels = jnp.where(labels == IGNORE_INDEX, 0, labels)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    token_loss = -jnp.take_along_axis(log_probs, safe_labels[..., None], axis=-1)[..., 0]

    denom = jnp.maximum(loss_mask.sum(), 1.0)
    lm_loss = (token_loss * loss_mask).sum() / denom
    slot_diversity_loss = slot_parameter_diversity_loss(model)
    loss = lm_loss + SLOT_DIVERSITY_WEIGHT * slot_diversity_loss

    predictions = jnp.argmax(logits, axis=-1)
    accuracy = ((predictions == safe_labels) * loss_mask).sum() / denom
    return {
        "loss": loss,
        "lm_loss": lm_loss,
        "slot_diversity_loss": slot_diversity_loss,
        "accuracy": accuracy,
    }


def parameter_count(model: nnx.Module) -> int:
    return int(sum(variable[...].size for _, variable in nnx.to_flat_state(nnx.state(model, nnx.Param))))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TPU-oriented Tachia language-model training.")
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
    parser.add_argument("--embed-dim", type=int, default=1536)
    parser.add_argument("--layers", type=int, default=24)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--slots", type=int, default=128)
    parser.add_argument("--selector-temperature", type=float, default=4.0)
    parser.add_argument("--selector-mode", choices=["sigmoid", "softmax"], default="sigmoid")
    parser.add_argument("--steps", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="./output/tpu_lm")
    parser.add_argument("--shuffle-offsets", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sharding", choices=["ddp", "no"], default="no")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.text_field is None and args.jsonl_field is not None:
        args.text_field = args.jsonl_field
    tokenizer = PratchyaTokenizer.from_pretrained("tokenizer")
    tokens = load_or_build_tokens(
        tokenizer=tokenizer,
        data_paths=args.data_path,
        hf_repo=args.hf_repo,
        hf_config=args.hf_config,
        hf_split=args.hf_split,
        hf_streaming=args.hf_streaming,
        token_cache=args.token_cache,
        text_field=args.text_field,
        max_records=args.max_records,
    )
    dataset = TokenLMDataset(
        tokens,
        sequence_length=args.sequence_length,
        num_examples=args.train_examples,
        seed=args.seed,
        shuffle_offsets=args.shuffle_offsets,
    )
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

    print(
        json.dumps(
            {
                "tokens": int(tokens.shape[0]),
                "examples": len(dataset),
                "parameters": parameter_count(model),
                "config": {
                    "embed_dim": args.embed_dim,
                    "layers": args.layers,
                    "heads": args.heads,
                    "slots": args.slots,
                    "selector_mode": args.selector_mode,
                    "selector_temperature": args.selector_temperature,
                    "sequence_length": args.sequence_length,
                    "batch_size": args.batch_size,
                    "steps": args.steps,
                },
            },
            sort_keys=True,
        )
    )

    trainer = Trainer(
        model=model,
        loss_fn=lm_loss_fn,
        train_dataset=dataset,
        config=TrainerConfig(
            batch_size=args.batch_size,
            max_steps=args.steps,
            logging_steps=args.logging_steps,
            learning_rate=args.learning_rate,
            warmup_ratio=args.warmup_ratio,
            output_dir=args.output_dir,
            num_workers=args.num_workers,
            sharding="ddp" if args.sharding == "ddp" else None,
        ),
    )
    trainer.train()


if __name__ == "__main__":
    main()
