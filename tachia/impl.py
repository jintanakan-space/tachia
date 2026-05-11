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


def _prefix_sigmoid_compress(logits: Array, values: Array, eps: float) -> Array:
    weights = jax.nn.sigmoid(logits)
    values = jnp.swapaxes(values, 1, 2)
    weighted_values = weights[..., None] * values[:, :, None, :, :]
    prefix_values = jnp.cumsum(weighted_values, axis=-2)
    prefix_weights = jnp.cumsum(weights, axis=-1)[..., None]
    compressed = prefix_values / (prefix_weights + eps)
    return jnp.transpose(compressed, (0, 3, 2, 1, 4))


def _full_sigmoid_compress(logits: Array, values: Array, eps: float) -> Array:
    weights = _normalize_sigmoid(logits, axis=-1, eps=eps)
    return jnp.einsum("bhsl,blhd->bshd", weights, values)


def _prefix_softmax_compress(logits: Array, values: Array) -> Array:
    length = values.shape[1]
    causal_mask = jnp.tril(jnp.ones((length, length), dtype=bool))
    masked_logits = jnp.where(causal_mask[None, None, None, :, :], logits[:, :, :, None, :], -jnp.inf)
    weights = jax.nn.softmax(masked_logits, axis=-1)
    return jnp.einsum("bhstl,blhd->btshd", weights, values)


def _full_softmax_compress(logits: Array, values: Array) -> Array:
    weights = jax.nn.softmax(logits, axis=-1)
    return jnp.einsum("bhsl,blhd->bshd", weights, values)


def _slot_dot_product_attention(q: Array, k: Array, v: Array) -> Array:
    """Attention from each token query to its compressed slots.

    Shapes:
      q: [B, T, H, D]
      k: [B, T, S, H, D]
      v: [B, T, S, H, D]

    `nnx.dot_product_attention` treats the per-token `T` dimension as a batch
    axis here, and XLA materializes a large broadcast shaped roughly
    [layers, B, H, S, T, D] on TPU. The explicit einsums keep the largest
    attention tensor at [B, T, H, S].
    """

    scale = q.shape[-1] ** -0.5
    logits = jnp.einsum("bthd,btshd->bths", q, k) * scale
    weights = jax.nn.softmax(logits, axis=-1)
    return jnp.einsum("bths,btshd->bthd", weights, v)


def _causal_sigmoid_slot_attention(
    q: Array,
    key_selector_logits: Array,
    value_selector_logits: Array,
    keys: Array,
    values: Array,
    eps: float,
) -> Array:
    """Fused causal sigmoid slot compression and attention.

    This is algebraically the same as:
      1. causal prefix sigmoid compression for K/V
      2. attention from each token query to its prefix slots

    but it scans over sequence positions and keeps only cumulative slot
    numerators/denominators in memory.
    """

    batch, _, heads, head_dim = q.shape
    slots = key_selector_logits.shape[2]
    dtype = q.dtype

    q_time = jnp.swapaxes(q, 0, 1)
    key_time = jnp.swapaxes(keys, 0, 1)
    value_time = jnp.swapaxes(values, 0, 1)
    key_logits_time = jnp.moveaxis(key_selector_logits, -1, 0)
    value_logits_time = jnp.moveaxis(value_selector_logits, -1, 0)

    zeros_num = jnp.zeros((batch, heads, slots, head_dim), dtype=dtype)
    zeros_den = jnp.zeros((batch, heads, slots), dtype=dtype)
    init = (zeros_num, zeros_den, zeros_num, zeros_den)
    scale = head_dim**-0.5

    def step(carry, inputs):
        key_num, key_den, value_num, value_den = carry
        q_t, key_t, value_t, key_logits_t, value_logits_t = inputs

        key_weights_t = jax.nn.sigmoid(key_logits_t).astype(dtype)
        value_weights_t = jax.nn.sigmoid(value_logits_t).astype(dtype)

        key_num = key_num + key_weights_t[..., None] * key_t[:, :, None, :]
        key_den = key_den + key_weights_t
        value_num = value_num + value_weights_t[..., None] * value_t[:, :, None, :]
        value_den = value_den + value_weights_t

        key_slots = key_num / (key_den[..., None] + eps)
        value_slots = value_num / (value_den[..., None] + eps)
        logits = jnp.einsum("bhd,bhsd->bhs", q_t, key_slots) * scale
        weights = jax.nn.softmax(logits, axis=-1)
        y_t = jnp.einsum("bhs,bhsd->bhd", weights, value_slots)
        return (key_num, key_den, value_num, value_den), y_t

    _, y_time = jax.lax.scan(
        step,
        init,
        (q_time, key_time, value_time, key_logits_time, value_logits_time),
    )
    return jnp.swapaxes(y_time, 0, 1)


def _selector_weights(logits: Array, mode: str, causal: bool, eps: float) -> Array:
    if causal:
        length = logits.shape[-1]
        causal_mask = jnp.tril(jnp.ones((length, length), dtype=bool))
        masked_logits = jnp.where(
            causal_mask[None, None, None, :, :],
            logits[:, :, :, None, :],
            -jnp.inf,
        )
        if mode == "sigmoid":
            weights = jax.nn.sigmoid(masked_logits)
            weights = weights * causal_mask[None, None, None, :, :]
            weights = weights / (jnp.sum(weights, axis=-1, keepdims=True) + eps)
            return weights
        return jax.nn.softmax(masked_logits, axis=-1)

    if mode == "sigmoid":
        weights = _normalize_sigmoid(logits, axis=-1, eps=eps)
    else:
        weights = jax.nn.softmax(logits, axis=-1)
    return weights[:, :, :, None, :]


def selector_activation_metrics(weights: Array, eps: float = 1e-6) -> dict[str, Array]:
    """Compute selector usage metrics from weights shaped [B, H, S, T, L]."""

    entropy = -jnp.sum(weights * jnp.log(weights + eps), axis=-1)
    effective_context = jnp.exp(entropy)
    top_tokens = jnp.argmax(weights, axis=-1)

    slots = weights.shape[2]
    if slots < 2:
        zero = jnp.zeros(weights.shape[:2] + weights.shape[3:4], dtype=weights.dtype)
        return {
            "entropy": entropy,
            "effective_context": effective_context,
            "top_token_overlap": zero,
            "pairwise_selector_cosine": zero,
        }

    token_counts = jnp.sum(jax.nn.one_hot(top_tokens, weights.shape[-1], dtype=weights.dtype), axis=2)
    top_overlap = (jnp.sum(jnp.square(token_counts), axis=-1) - slots) / (slots * (slots - 1))

    normed = weights / (jnp.linalg.norm(weights, axis=-1, keepdims=True) + eps)
    summed_normed = jnp.sum(normed, axis=2)
    off_diagonal = (jnp.sum(jnp.square(summed_normed), axis=-1) - slots) / (slots * (slots - 1))

    return {
        "entropy": entropy,
        "effective_context": effective_context,
        "top_token_overlap": top_overlap,
        "pairwise_selector_cosine": off_diagonal,
    }


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
        return (normalized * self.scale[...]).astype(dtype)


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
        collect_stats: bool = False,
    ) -> Array | tuple[Array, dict[str, dict[str, Array]]]:
        cfg = self.config

        q = apply_rope(self._split_heads(self.q_proj(x)), positions, cfg.rope_theta)
        selector_source = self._split_heads(x)

        key_selector_logits = (
            jnp.einsum("hds,blhd->bhsl", self.k_slots[...], selector_source)
            * cfg.selector_temperature
        )
        value_selector_logits = (
            jnp.einsum("hds,blhd->bhsl", self.v_slots[...], selector_source)
            * cfg.selector_temperature
        )

        k_l = apply_rope(self._split_heads(self.k_proj(x)), positions, cfg.rope_theta)
        v_l = self._split_heads(self.v_proj(x))

        if cfg.causal and cfg.selector_mode == "sigmoid" and not collect_stats:
            y = _causal_sigmoid_slot_attention(
                q,
                key_selector_logits,
                value_selector_logits,
                k_l,
                v_l,
                cfg.eps,
            )
        elif cfg.causal:
            if cfg.selector_mode == "sigmoid":
                k_s = _prefix_sigmoid_compress(key_selector_logits, k_l, cfg.eps)
                v_s = _prefix_sigmoid_compress(value_selector_logits, v_l, cfg.eps)
            else:
                k_s = _prefix_softmax_compress(key_selector_logits, k_l)
                v_s = _prefix_softmax_compress(value_selector_logits, v_l)
            y = _slot_dot_product_attention(q, k_s, v_s)
        else:
            if cfg.selector_mode == "sigmoid":
                k_s = _full_sigmoid_compress(key_selector_logits, k_l, cfg.eps)
                v_s = _full_sigmoid_compress(value_selector_logits, v_l, cfg.eps)
            else:
                k_s = _full_softmax_compress(key_selector_logits, k_l)
                v_s = _full_softmax_compress(value_selector_logits, v_l)
            k_s = jnp.broadcast_to(k_s[:, None], (x.shape[0], x.shape[1], *k_s.shape[1:]))
            v_s = jnp.broadcast_to(v_s[:, None], (x.shape[0], x.shape[1], *v_s.shape[1:]))
            y = _slot_dot_product_attention(q, k_s, v_s)
        y = self.out_proj(self._merge_heads(y))
        if not collect_stats:
            return y

        key_weights = _selector_weights(key_selector_logits, cfg.selector_mode, cfg.causal, cfg.eps)
        value_weights = _selector_weights(value_selector_logits, cfg.selector_mode, cfg.causal, cfg.eps)
        return y, {
            "key": selector_activation_metrics(key_weights, cfg.eps),
            "value": selector_activation_metrics(value_weights, cfg.eps),
        }


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
        collect_stats: bool = False,
    ) -> Array | tuple[Array, dict[str, dict[str, Array]]]:
        attn_outputs = self.attn(
            self.attn_norm(x),
            positions=positions,
            deterministic=deterministic,
            collect_stats=collect_stats,
        )
        if collect_stats:
            attn_y, stats = attn_outputs
        else:
            attn_y = attn_outputs
            stats = None

        x = x + attn_y
        x = x + self.mlp(self.mlp_norm(x), deterministic=deterministic)
        if collect_stats:
            return x, stats
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
        collect_stats: bool = False,
    ) -> Array | tuple[Array, dict[str, dict[str, Array]]]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, length]")

        x = self.token_embedding(input_ids)
        x = x * math.sqrt(self.config.embed_dim)
        outputs = call_block(
            self.blocks,
            x,
            positions=positions,
            deterministic=deterministic,
            module_kwargs={"collect_stats": collect_stats},
            in_axes=(0, nnx.Carry, None, None),
            return_aux=collect_stats,
        )

        if collect_stats:
            x, stats = outputs
            return self.norm(x), stats
        return self.norm(outputs)


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
        collect_stats: bool = False,
    ) -> Array | tuple[Array, dict[str, dict[str, Array]]]:
        outputs = self.model(
            input_ids,
            positions=positions,
            deterministic=deterministic,
            collect_stats=collect_stats,
        )
        if collect_stats:
            hidden, stats = outputs
        else:
            hidden = outputs
            stats = None

        if self.config.tie_word_embeddings:
            logits = self.model.token_embedding.attend(hidden)
        else:
            logits = self.lm_head(hidden)
        if collect_stats:
            return logits, stats
        return logits

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
            logits = self(tokens, deterministic=True)[:, -1, :] / temperature
            rng, step_rng = jax.random.split(rng)
            next_token = jax.random.categorical(step_rng, logits, axis=-1)[:, None]
            tokens = jnp.concatenate([tokens, next_token], axis=1)

        return tokens


class StandardSelfAttention(nnx.Module):
    def __init__(self, config: TachiaConfig, *, rngs: nnx.Rngs):
        self.config = config
        dim = config.embed_dim
        self.q_proj = nnx.Linear(dim, dim, use_bias=config.use_bias, rngs=rngs)
        self.k_proj = nnx.Linear(dim, dim, use_bias=config.use_bias, rngs=rngs)
        self.v_proj = nnx.Linear(dim, dim, use_bias=config.use_bias, rngs=rngs)
        self.out_proj = nnx.Linear(dim, dim, use_bias=config.use_bias, rngs=rngs)

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
        q = apply_rope(self._split_heads(self.q_proj(x)), positions, self.config.rope_theta)
        k = apply_rope(self._split_heads(self.k_proj(x)), positions, self.config.rope_theta)
        v = self._split_heads(self.v_proj(x))
        y = nnx.dot_product_attention(
            q,
            k,
            v,
            dropout_rate=0.0,
            deterministic=True,
            is_causal=self.config.causal,
        )
        return self.out_proj(self._merge_heads(y))


class StandardTransformerBlock(nnx.Module):
    def __init__(self, config: TachiaConfig, *, rngs: nnx.Rngs):
        self.attn_norm = RMSNorm(config.embed_dim, eps=config.eps, rngs=rngs)
        self.attn = StandardSelfAttention(config, rngs=rngs)
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


class StandardTransformerModel(nnx.Module):
    def __init__(self, config: TachiaConfig, *, rngs: nnx.Rngs):
        self.config = config
        self.token_embedding = nnx.Embed(config.vocab_size, config.embed_dim, rngs=rngs)
        self.blocks = create_block(
            config.num_layers,
            StandardTransformerBlock,
            rngs=rngs,
            module_args=(config,),
        )
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


class StandardTransformerLM(nnx.Module):
    def __init__(self, config: TachiaConfig, *, rngs: nnx.Rngs):
        self.config = config
        self.model = StandardTransformerModel(config, rngs=rngs)
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


def create_model(
    *,
    vocab_size: int,
    embed_dim: int = 112,
    num_layers: int = 4,
    num_heads: int = 8,
    num_slots: int = 32,
    mlp_hidden_dim: int | None = None,
    causal: bool = True,
    selector_mode: str = "sigmoid",
    selector_temperature: float = 2.0,
    seed: int = 0,
) -> TachiaLM:
    config = TachiaConfig(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        num_slots=num_slots,
        mlp_hidden_dim=mlp_hidden_dim,
        causal=causal,
        selector_mode=selector_mode,
        selector_temperature=selector_temperature,
    )
    return TachiaLM(config, rngs=nnx.Rngs(seed))


def create_standard_transformer(
    *,
    vocab_size: int,
    embed_dim: int = 112,
    num_layers: int = 4,
    num_heads: int = 8,
    mlp_hidden_dim: int | None = None,
    causal: bool = True,
    seed: int = 0,
) -> StandardTransformerLM:
    config = TachiaConfig(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        num_slots=1,
        mlp_hidden_dim=mlp_hidden_dim,
        causal=causal,
    )
    return StandardTransformerLM(config, rngs=nnx.Rngs(seed))


__all__: Sequence[str] = (
    "GateMLP",
    "RMSNorm",
    "StandardSelfAttention",
    "StandardTransformerBlock",
    "StandardTransformerLM",
    "StandardTransformerModel",
    "TachiaAttention",
    "TachiaBlock",
    "TachiaLM",
    "TachiaModel",
    "apply_rope",
    "create_model",
    "create_standard_transformer",
    "selector_activation_metrics",
)
