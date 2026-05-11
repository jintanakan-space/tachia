from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from tachia import TachiaAttention, TachiaConfig, TachiaLM, create_model, create_standard_transformer
from tachia.impl import (
    _causal_sigmoid_slot_attention,
    _prefix_sigmoid_compress,
    _prefix_softmax_compress,
    _slot_dot_product_attention,
)
from train import loss_fn, slot_parameter_diversity_loss


class TachiaModelTest(unittest.TestCase):
    def test_config_has_no_max_seq_len(self) -> None:
        config = TachiaConfig(vocab_size=32, embed_dim=16, num_layers=1, num_heads=4, num_slots=2)
        self.assertFalse(hasattr(config, "max_seq_len"))
        model = create_model(vocab_size=32, embed_dim=16, num_layers=1, num_heads=4, num_slots=2)
        logits = model(jnp.ones((1, 7), dtype=jnp.int32))
        self.assertEqual(logits.shape, (1, 7, 32))

    def test_attention_shapes_for_causal_and_noncausal(self) -> None:
        x = jax.random.normal(jax.random.PRNGKey(0), (2, 5, 16))
        for causal in (True, False):
            for selector_mode in ("sigmoid", "softmax"):
                config = TachiaConfig(
                    vocab_size=32,
                    embed_dim=16,
                    num_layers=1,
                    num_heads=4,
                    num_slots=3,
                    causal=causal,
                    selector_mode=selector_mode,
                )
                attention = TachiaAttention(config, rngs=nnx.Rngs(1))
                y = attention(x)
                self.assertEqual(y.shape, x.shape)

    def test_standard_transformer_shapes_and_no_slot_diversity(self) -> None:
        model = create_standard_transformer(vocab_size=32, embed_dim=16, num_layers=1, num_heads=4)
        logits = model(jnp.ones((2, 6), dtype=jnp.int32))
        self.assertEqual(logits.shape, (2, 6, 32))
        self.assertEqual(float(slot_parameter_diversity_loss(model)), 0.0)

    def test_config_rejects_invalid_selector_mode(self) -> None:
        with self.assertRaises(ValueError):
            config = TachiaConfig(
                vocab_size=32,
                embed_dim=16,
                num_layers=1,
                num_heads=4,
                num_slots=3,
                selector_mode="bad",
            )

    def test_config_rejects_invalid_selector_temperature(self) -> None:
        with self.assertRaises(ValueError):
            TachiaConfig(vocab_size=32, selector_temperature=0.0)

    def test_prefix_sigmoid_compress_matches_explicit_causal_compress(self) -> None:
        logits = jnp.asarray(
            [[[[0.1, -0.4, 0.7, 1.2], [0.5, -0.2, 0.3, -0.8]]]],
            dtype=jnp.float32,
        )
        values = jnp.arange(1 * 4 * 1 * 2, dtype=jnp.float32).reshape(1, 4, 1, 2)
        actual = _prefix_sigmoid_compress(logits, values, eps=1e-6)

        expected_steps = []
        weights = jax.nn.sigmoid(logits)
        for t in range(values.shape[1]):
            prefix_weights = weights[..., : t + 1]
            prefix_values = values[:, : t + 1, :, :]
            compressed = jnp.einsum("bhsl,blhd->bshd", prefix_weights, prefix_values)
            denom = jnp.transpose(prefix_weights.sum(axis=-1), (0, 2, 1))[..., None]
            compressed = compressed / (denom + 1e-6)
            expected_steps.append(compressed)
        expected = jnp.stack(expected_steps, axis=1)

        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    def test_prefix_softmax_compress_matches_explicit_causal_compress(self) -> None:
        logits = jnp.asarray(
            [[[[0.1, -0.4, 0.7, 1.2], [0.5, -0.2, 0.3, -0.8]]]],
            dtype=jnp.float32,
        )
        values = jnp.arange(1 * 4 * 1 * 2, dtype=jnp.float32).reshape(1, 4, 1, 2)
        actual = _prefix_softmax_compress(logits, values)

        expected_steps = []
        for t in range(values.shape[1]):
            weights = jax.nn.softmax(logits[..., : t + 1], axis=-1)
            prefix_values = values[:, : t + 1, :, :]
            expected_steps.append(jnp.einsum("bhsl,blhd->bshd", weights, prefix_values))
        expected = jnp.stack(expected_steps, axis=1)

        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    def test_slot_dot_product_attention_matches_manual_softmax(self) -> None:
        q = jax.random.normal(jax.random.PRNGKey(0), (2, 5, 4, 6))
        k = jax.random.normal(jax.random.PRNGKey(1), (2, 5, 3, 4, 6))
        v = jax.random.normal(jax.random.PRNGKey(2), (2, 5, 3, 4, 6))

        actual = _slot_dot_product_attention(q, k, v)
        logits = jnp.sum(q[:, :, None, :, :] * k, axis=-1) * (q.shape[-1] ** -0.5)
        weights = jax.nn.softmax(logits, axis=2)
        expected = jnp.sum(weights[..., None] * v, axis=2)

        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    def test_fused_causal_sigmoid_slot_attention_matches_unfused_path(self) -> None:
        q = jax.random.normal(jax.random.PRNGKey(0), (2, 5, 4, 6))
        k = jax.random.normal(jax.random.PRNGKey(1), (2, 5, 4, 6))
        v = jax.random.normal(jax.random.PRNGKey(2), (2, 5, 4, 6))
        key_logits = jax.random.normal(jax.random.PRNGKey(3), (2, 4, 3, 5))
        value_logits = jax.random.normal(jax.random.PRNGKey(4), (2, 4, 3, 5))

        actual = _causal_sigmoid_slot_attention(q, key_logits, value_logits, k, v, eps=1e-6)
        k_slots = _prefix_sigmoid_compress(key_logits, k, eps=1e-6)
        v_slots = _prefix_sigmoid_compress(value_logits, v, eps=1e-6)
        expected = _slot_dot_product_attention(q, k_slots, v_slots)

        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

    def test_causal_outputs_do_not_change_when_future_tokens_change(self) -> None:
        config = TachiaConfig(vocab_size=64, embed_dim=16, num_layers=2, num_heads=4, num_slots=3)
        model = TachiaLM(config, rngs=nnx.Rngs(2))
        left = jnp.asarray([[1, 2, 3, 4, 5, 6]], dtype=jnp.int32)
        right = jnp.asarray([[1, 2, 3, 40, 50, 60]], dtype=jnp.int32)

        left_logits = model(left)
        right_logits = model(right)
        np.testing.assert_allclose(left_logits[:, :3], right_logits[:, :3], rtol=1e-5, atol=1e-5)

    def test_train_loss_includes_slot_diversity_metric(self) -> None:
        model = create_model(vocab_size=32, embed_dim=16, num_layers=1, num_heads=4, num_slots=3)
        batch = {
            "input_ids": jnp.asarray([[1, 2, 3, 4]], dtype=jnp.int32),
            "labels": jnp.asarray([[2, 3, 4, 5]], dtype=jnp.int32),
            "loss_mask": jnp.ones((1, 4), dtype=jnp.float32),
            "answer_mask": jnp.ones((1, 4), dtype=jnp.float32),
        }
        metrics = loss_fn(model, batch)
        self.assertEqual(metrics["loss"].shape, ())
        self.assertEqual(metrics["lm_loss"].shape, ())
        self.assertEqual(metrics["prompt_lm_loss"].shape, ())
        self.assertEqual(metrics["answer_lm_loss"].shape, ())
        self.assertEqual(metrics["slot_diversity_loss"].shape, ())
        self.assertEqual(metrics["exact_answer_accuracy"].shape, ())
        self.assertGreaterEqual(float(metrics["slot_diversity_loss"]), 0.0)

    def test_collect_stats_returns_selector_activation_metrics(self) -> None:
        model = create_model(vocab_size=32, embed_dim=16, num_layers=2, num_heads=4, num_slots=3)
        logits, stats = model(jnp.ones((2, 5), dtype=jnp.int32), collect_stats=True)
        self.assertEqual(logits.shape, (2, 5, 32))
        for side in ("key", "value"):
            self.assertEqual(stats[side]["entropy"].shape, (2, 2, 4, 3, 5))
            self.assertEqual(stats[side]["effective_context"].shape, (2, 2, 4, 3, 5))
            self.assertEqual(stats[side]["top_token_overlap"].shape, (2, 2, 4, 5))
            self.assertEqual(stats[side]["pairwise_selector_cosine"].shape, (2, 2, 4, 5))


if __name__ == "__main__":
    unittest.main()
