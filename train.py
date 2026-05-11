from __future__ import annotations

import math
import string
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from zlynx.trainer import Trainer, TrainerConfig

from tachia import create_model
from tokenizer import PratchyaTokenizer


IGNORE_INDEX = -100
SLOT_DIVERSITY_WEIGHT = 1e-3
PROMPT_LOSS_WEIGHT = 1.0
ANSWER_LOSS_WEIGHT = 1.0


@dataclass(frozen=True)
class ABCDigitsDataset:
    tokenizer: PratchyaTokenizer
    num_examples: int = 10_000
    sequence_length: int = 256
    context_equations: int = 96
    target_depth: float = 0.5
    seed: int = 42

    def __len__(self) -> int:
        return self.num_examples

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(self.seed + index)
        letters = np.array(list(string.ascii_uppercase))
        target = str(rng.choice(letters))

        values = rng.choice(np.arange(100_000, 1_000_000), size=26, replace=False)
        mapping = {letter: f"{value:06d}" for letter, value in zip(letters, values)}

        non_target = [letter for letter in letters if letter != target]
        rng.shuffle(non_target)

        equations = [f"{letter}={mapping[letter]}" for letter in non_target]

        weights = np.power(2.0, np.arange(len(non_target), dtype=np.float64))
        rng.shuffle(weights)
        weights = weights / weights.sum()
        extra_count = max(0, self.context_equations - len(equations) - 1)
        extra_letters = rng.choice(non_target, size=extra_count, replace=True, p=weights)
        equations.extend(f"{letter}={mapping[str(letter)]}" for letter in extra_letters)
        rng.shuffle(equations)

        insert_at = int(np.clip(round(self.target_depth * len(equations)), 0, len(equations)))
        equations.insert(insert_at, f"{target}={mapping[target]}")
        prompt = "\n".join(equations) + f"\n{target}="
        answer = mapping[target]

        input_ids, labels, loss_mask, answer_mask = self._encode_for_completion(prompt, answer)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "loss_mask": loss_mask,
            "answer_mask": answer_mask,
        }

    def _encode_for_completion(
        self,
        prompt: str,
        answer: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        bos = [self.tokenizer.bos_token_id] if self.tokenizer.bos_token_id is not None else []
        eos = [self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id is not None else []
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        answer_ids = self.tokenizer.encode(answer, add_special_tokens=False) + eos

        full = np.asarray(bos + prompt_ids + answer_ids, dtype=np.int32)
        if full.shape[0] > self.sequence_length + 1:
            keep_prompt = self.sequence_length + 1 - len(bos) - len(answer_ids)
            if keep_prompt <= 0:
                raise ValueError("sequence_length is too small for one ABCDigits answer")
            prompt_ids = prompt_ids[-keep_prompt:]
            full = np.asarray(bos + prompt_ids + answer_ids, dtype=np.int32)

        input_ids = np.full((self.sequence_length,), self.tokenizer.pad_token_id, dtype=np.int32)
        labels = np.full((self.sequence_length,), IGNORE_INDEX, dtype=np.int32)
        loss_mask = np.zeros((self.sequence_length,), dtype=np.float32)
        answer_mask = np.zeros((self.sequence_length,), dtype=np.float32)

        seq_len = min(self.sequence_length, full.shape[0] - 1)
        input_ids[:seq_len] = full[:seq_len]
        labels[:seq_len] = full[1 : seq_len + 1]
        loss_mask[:seq_len] = 1.0

        answer_start = len(bos) + len(prompt_ids)
        label_positions = np.arange(1, seq_len + 1)
        answer_mask[:seq_len] = (label_positions >= answer_start).astype(np.float32)
        labels[loss_mask == 0] = IGNORE_INDEX
        return input_ids, labels, loss_mask, answer_mask


def loss_fn(model, batch: dict[str, jax.Array]) -> dict[str, jax.Array]:
    input_ids = jnp.asarray(batch["input_ids"], dtype=jnp.int32)
    labels = jnp.asarray(batch["labels"], dtype=jnp.int32)
    loss_mask = jnp.asarray(batch["loss_mask"], dtype=jnp.float32)
    answer_mask = jnp.asarray(batch["answer_mask"], dtype=jnp.float32)

    logits = model(input_ids)
    safe_labels = jnp.where(labels == IGNORE_INDEX, 0, labels)
    token_loss = optax_softmax_cross_entropy_with_integer_labels(logits, safe_labels)
    loss_weights = (PROMPT_LOSS_WEIGHT * loss_mask) + (
        (ANSWER_LOSS_WEIGHT - PROMPT_LOSS_WEIGHT) * answer_mask
    )
    denom = jnp.maximum(loss_weights.sum(), 1.0)
    lm_loss = (token_loss * loss_weights).sum() / denom
    slot_diversity_loss = slot_parameter_diversity_loss(model)
    loss = lm_loss + SLOT_DIVERSITY_WEIGHT * slot_diversity_loss

    predictions = jnp.argmax(logits, axis=-1)
    loss_mask_denom = jnp.maximum(loss_mask.sum(), 1.0)
    accuracy = ((predictions == safe_labels) * loss_mask).sum() / loss_mask_denom
    answer_denom = jnp.maximum(answer_mask.sum(), 1.0)
    answer_accuracy = ((predictions == safe_labels) * answer_mask).sum() / answer_denom
    answer_lm_loss = ((token_loss * answer_mask).sum()) / answer_denom
    prompt_mask = jnp.maximum(loss_mask - answer_mask, 0.0)
    prompt_denom = jnp.maximum(prompt_mask.sum(), 1.0)
    prompt_lm_loss = ((token_loss * prompt_mask).sum()) / prompt_denom
    answer_token_correct = jnp.logical_or(answer_mask == 0.0, predictions == safe_labels)
    has_answer = answer_mask.sum(axis=-1) > 0.0
    exact_answer = jnp.logical_and(jnp.all(answer_token_correct, axis=-1), has_answer)
    exact_answer_accuracy = exact_answer.sum() / jnp.maximum(has_answer.sum(), 1)
    return {
        "loss": loss,
        "lm_loss": lm_loss,
        "prompt_lm_loss": prompt_lm_loss,
        "answer_lm_loss": answer_lm_loss,
        "slot_diversity_loss": slot_diversity_loss,
        "accuracy": accuracy,
        "answer_accuracy": answer_accuracy,
        "exact_answer_accuracy": exact_answer_accuracy,
    }


def optax_softmax_cross_entropy_with_integer_labels(logits: jax.Array, labels: jax.Array) -> jax.Array:
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.take_along_axis(log_probs, labels[..., None], axis=-1)[..., 0]


def slot_parameter_diversity_loss(model) -> jax.Array:
    losses = []
    for path, variable in nnx.to_flat_state(nnx.state(model, nnx.Param)):
        if path[-1] in {"k_slots", "v_slots"}:
            losses.append(_slot_parameter_diversity_loss(variable[...]))
    if not losses:
        return jnp.asarray(0.0, dtype=jnp.float32)
    return jnp.mean(jnp.stack(losses))


def _slot_parameter_diversity_loss(slots: jax.Array) -> jax.Array:
    num_slots = slots.shape[-1]
    if num_slots < 2:
        return jnp.asarray(0.0, dtype=slots.dtype)

    vectors = jnp.moveaxis(slots, -1, -2)
    vectors = vectors / (jnp.linalg.norm(vectors, axis=-1, keepdims=True) + 1e-6)
    similarity = jnp.einsum("...sd,...td->...st", vectors, vectors)
    off_diagonal = similarity - jnp.eye(num_slots, dtype=similarity.dtype)
    return jnp.sum(jnp.square(off_diagonal)) / (
        math.prod(similarity.shape[:-2]) * num_slots * (num_slots - 1)
    )


def main() -> None:
    tokenizer = PratchyaTokenizer.from_pretrained("tokenizer")
    model = create_model(vocab_size=tokenizer.vocab_size)
    dataset = ABCDigitsDataset(
        tokenizer=tokenizer,
        num_examples=1_000,
        sequence_length=256,
        context_equations=96,
        target_depth=0.5,
        seed=42,
    )

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        train_dataset=dataset,
        config=TrainerConfig(
            batch_size=4,
            max_steps=200,
            logging_steps=10,
            learning_rate=3e-4,
            warmup_ratio=0.05,
            output_dir="./output/abcdigits",
        ),
    )
    trainer.train()


if __name__ == "__main__":
    main()
