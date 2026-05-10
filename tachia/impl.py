from __future__ import annotations

import math
from collections.abc import Sequence

import jax
import jax.numpy as jnp
from flax import nnx
from zlynx.module.block import create_block, call_block

from tachia.config import TachiaConfig


Array = jax.Array


def _normalize_sigmoid(logits: Array, axis: int, eps: float) -> Array:
    weights = jax.nn.sigmoid(logits)
    return weights / (jnp.sum(weights, axis=axis, keepdims=True) + eps)


def apply_rope(x: Array, positions: Array | None = None, theta: float = 10_000.0) -> Array:
    """Apply rotary position embeddings to the final dimension of `x`.

    `x` is expected to be shaped `[batch, length, ..., dim]`.
    """

    dim = x.shape[-1]
    if dim % 2 != 0:
        raise ValueError("RoPE requires an even embedding dimension")

    length = x.shape[1]
    if positions is None:
        positions = jnp.arange(length, dtype=jnp.float32)
    else:
        positions = positions.astype(jnp.float32)

    half_dim = dim // 2
    inv_freq = theta ** (-jnp.arange(0, half_dim, dtype=jnp.float32) / half_dim)
    angles = positions[:, None] * inv_freq[None, :]
    broadcast_shape = (1, length, *(1 for _ in range(x.ndim - 3)), half_dim)
    cos = jnp.cos(angles).reshape(broadcast_shape)
    sin = jnp.sin(angles).reshape(broadcast_shape)

    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    rotated = jnp.stack(
        (x_even * cos - x_odd * sin, x_even * sin + x_odd * cos),
        axis=-1,
    )
    return rotated.reshape(x.shape)


class RMSNorm(nnx.Module):
    def __init__(self, dim: int, *, eps: float = 1e-6, rngs: nnx.Rngs):
        self.eps = eps
        self.scale = nnx.Param(jnp.ones((dim,)))

    def __call__(self, x: Array) -> Array:
        dtype = x.dtype
        x_f32 = x.astype(jnp.float32)
        variance = jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True)
        normalized = x_f32 * jax.lax.rsqrt(variance + self.eps)
        return (normalized * self.scale.value).astype(dtype)


class TachiaAttention(nnx.Module):
    """Slot-compressed sigmoid attention from the README."""

    def __init__(self, config: TachiaConfig, *, rngs: nnx.Rngs):
        self.config = config
        dim = config.embed_dim
        slots = config.num_slots

        self.q_proj = nnx.Linear(dim, dim, use_bias=config.use_bias, rngs=rngs)
        self.k_proj = nnx.Linear(dim, dim, use_bias=config.use_bias, rngs=rngs)
        self.v_proj = nnx.Linear(dim, dim, use_bias=config.use_bias, rngs=rngs)
        self.out_proj = nnx.Linear(dim, dim, use_bias=config.use_bias, rngs=rngs)

        head_dim = dim // config.num_heads
        slot_scale = head_dim**-0.5
        slot_shape = (config.num_heads, head_dim, slots)
        self.k_slots = nnx.Param(jax.random.normal(rngs.params(), slot_shape) * slot_scale)
        self.v_slots = nnx.Param(jax.random.normal(rngs.params(), slot_shape) * slot_scale)

    def _split_heads(self, x: Array) -> Array:
        batch, length, _ = x.shape
        return x.reshape(batch, length, self.config.num_heads, -1)

    def _merge_heads(self, x: Array) -> Array:
        batch, length, _, _ = x.shape
        return x.reshape(batch, length, self.config.embed_dim)

    def __call__(
        self,
        x: Array,
        *,
        positions: Array | None = None,
        deterministic: bool = True,
    ) -> Array:
        cfg = self.config

        q = apply_rope(self._split_heads(self.q_proj(x)), positions, cfg.rope_theta)
        selector_source = self._split_heads(x)

        key_selector_logits = jnp.einsum("hds,blhd->bhsl", self.k_slots.value, selector_source)
        key_selector = _normalize_sigmoid(key_selector_logits, axis=-1, eps=cfg.eps)

        value_selector_logits = jnp.einsum("hds,blhd->bhsl", self.v_slots.value, selector_source)
        value_selector = _normalize_sigmoid(value_selector_logits, axis=-1, eps=cfg.eps)

        k_l = apply_rope(self._split_heads(self.k_proj(x)), positions, cfg.rope_theta)
        k_s = jnp.einsum("bhsl,blhd->bshd", key_selector, k_l)
        v_s = jnp.einsum("bhsl,blhd->bshd", value_selector, self._split_heads(self.v_proj(x)))

        y = nnx.dot_product_attention(
            q,
            k_s,
            v_s,
            dropout_rate=0.0,
            deterministic=True,
        )
        return self.out_proj(self._merge_heads(y))


class GateMLP(nnx.Module):
    """SiGLU-style gated MLP."""

    def __init__(self, config: TachiaConfig, *, rngs: nnx.Rngs):
        dim = config.embed_dim
        hidden_dim = config.resolved_mlp_hidden_dim
        self.gate_proj = nnx.Linear(dim, hidden_dim, use_bias=config.use_bias, rngs=rngs)
        self.up_proj = nnx.Linear(dim, hidden_dim, use_bias=config.use_bias, rngs=rngs)
        self.down_proj = nnx.Linear(hidden_dim, dim, use_bias=config.use_bias, rngs=rngs)

    def __call__(self, x: Array, *, deterministic: bool = True) -> Array:
        x = jax.nn.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.down_proj(x)


class TachiaBlock(nnx.Module):
    def __init__(self, config: TachiaConfig, *, rngs: nnx.Rngs):
        self.attn_norm = RMSNorm(config.embed_dim, eps=config.eps, rngs=rngs)
        self.attn = TachiaAttention(config, rngs=rngs)
        self.mlp_norm = RMSNorm(config.embed_dim, eps=config.eps, rngs=rngs)
        self.mlp = GateMLP(config, rngs=rngs)

    def __call__(
        self,
        x: Array,
        *,
        positions: Array | None = None,
        deterministic: bool = True,
    ) -> Array:
        x = x + self.attn(self.attn_norm(x), positions=positions, deterministic=deterministic)
        x = x + self.mlp(self.mlp_norm(x), deterministic=deterministic)
        return x


class TachiaModel(nnx.Module):
    def __init__(self, config: TachiaConfig, *, rngs: nnx.Rngs):
        self.config = config
        self.token_embedding = nnx.Embed(config.vocab_size, config.embed_dim, rngs=rngs)
        self.blocks = create_block(config.num_layers, TachiaBlock, rngs=rngs, module_args=(config,))
        self.norm = RMSNorm(config.embed_dim, eps=config.eps, rngs=rngs)

    def __call__(
        self,
        input_ids: Array,
        *,
        positions: Array | None = None,
        deterministic: bool = True,
    ) -> Array:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, length]")
        if input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError(
                f"sequence length {input_ids.shape[1]} exceeds max_seq_len "
                f"{self.config.max_seq_len}"
            )

        x = self.token_embedding(input_ids)
        x = x * math.sqrt(self.config.embed_dim)
        x = call_block(
            self.blocks,
            x,
            positions=positions,
            deterministic=deterministic,
            in_axes=(0, nnx.Carry, None, None),
        )

        return self.norm(x)


class TachiaLM(nnx.Module):
    def __init__(self, config: TachiaConfig, *, rngs: nnx.Rngs):
        self.config = config
        self.model = TachiaModel(config, rngs=rngs)
        if not config.tie_word_embeddings:
            self.lm_head = nnx.Linear(
                config.embed_dim,
                config.vocab_size,
                use_bias=False,
                rngs=rngs,
            )

    def __call__(
        self,
        input_ids: Array,
        *,
        positions: Array | None = None,
        deterministic: bool = True,
    ) -> Array:
        hidden = self.model(input_ids, positions=positions, deterministic=deterministic)
        if self.config.tie_word_embeddings:
            return self.model.token_embedding.attend(hidden)
        return self.lm_head(hidden)

    def loss(
        self,
        input_ids: Array,
        labels: Array,
        *,
        positions: Array | None = None,
        deterministic: bool = True,
    ) -> Array:
        logits = self(input_ids, positions=positions, deterministic=deterministic)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        token_log_probs = jnp.take_along_axis(log_probs, labels[..., None], axis=-1)[..., 0]
        return -jnp.mean(token_log_probs)

    def generate(
        self,
        input_ids: Array,
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        rng: Array | None = None,
    ) -> Array:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, length]")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")

        tokens = input_ids
        if rng is None:
            rng = jax.random.PRNGKey(0)

        for _ in range(max_new_tokens):
            if tokens.shape[1] > self.config.max_seq_len:
                context = tokens[:, -self.config.max_seq_len :]
            else:
                context = tokens
            logits = self(context, deterministic=True)[:, -1, :] / temperature
            rng, step_rng = jax.random.split(rng)
            next_token = jax.random.categorical(step_rng, logits, axis=-1)[:, None]
            tokens = jnp.concatenate([tokens, next_token], axis=1)

        return tokens


def create_model(
    *,
    vocab_size: int,
    max_seq_len: int,
    embed_dim: int = 256,
    num_layers: int = 4,
    num_heads: int = 8,
    num_slots: int = 64,
    mlp_hidden_dim: int | None = None,
    seed: int = 0,
) -> TachiaLM:
    config = TachiaConfig(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        num_slots=num_slots,
        mlp_hidden_dim=mlp_hidden_dim,
    )
    return TachiaLM(config, rngs=nnx.Rngs(seed))


__all__: Sequence[str] = (
    "GateMLP",
    "RMSNorm",
    "TachiaAttention",
    "TachiaBlock",
    "TachiaLM",
    "TachiaModel",
    "apply_rope",
    "create_model",
)
