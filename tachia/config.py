from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TachiaConfig:
    """Configuration for the Tachia NNX language model."""

    vocab_size: int
    embed_dim: int = 112
    num_layers: int = 4
    num_heads: int = 8
    num_slots: int = 32
    mlp_hidden_dim: int | None = None
    eps: float = 1e-6
    rope_theta: float = 10_000.0
    causal: bool = True
    selector_mode: str = "sigmoid"
    selector_temperature: float = 2.0
    tie_word_embeddings: bool = True
    use_bias: bool = False

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.embed_dim <= 0:
            raise ValueError("embed_dim must be positive")
        if self.embed_dim % 2 != 0:
            raise ValueError("embed_dim must be even for RoPE")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.embed_dim % self.num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        if (self.embed_dim // self.num_heads) % 2 != 0:
            raise ValueError("head dimension must be even for RoPE")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.num_slots <= 0:
            raise ValueError("num_slots must be positive")
        if self.eps <= 0.0:
            raise ValueError("eps must be positive")
        if self.selector_mode not in {"sigmoid", "softmax"}:
            raise ValueError("selector_mode must be 'sigmoid' or 'softmax'")
        if self.selector_temperature <= 0.0:
            raise ValueError("selector_temperature must be positive")

    @property
    def resolved_mlp_hidden_dim(self) -> int:
        return self.mlp_hidden_dim or self.embed_dim * 4
