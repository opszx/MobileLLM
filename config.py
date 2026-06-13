# Model configuration (tiny / 350M / 1B)
"""
PhantomLM Configuration
Phone-native hybrid Mamba-Transformer architecture
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PhantomLMConfig:
    # ── Vocabulary ──────────────────────────────────────────
    vocab_size: int = 32000
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    # ── Model dimensions ────────────────────────────────────
    d_model: int = 1024          # hidden dimension
    n_layers: int = 24           # total layers
    max_seq_len: int = 2048      # max context length

    # ── Hybrid layer placement ───────────────────────────────
    # Zone 1: layers 0-5   → pure Mamba     (fast, cheap)
    # Zone 2: layers 6-17  → mixed          (every 3rd is Attention)
    # Zone 3: layers 18-23 → Attention-heavy (alternating)
    mamba_zone_end: int = 6
    mixed_zone_end: int = 18
    # attention_layer_indices are computed automatically in model.py

    # ── Mamba SSM parameters ────────────────────────────────
    d_state: int = 16            # SSM state dimension
    d_conv: int = 4              # local conv width
    mamba_expand: int = 2        # inner dim = expand * d_model

    # ── Attention (GQA) parameters ──────────────────────────
    n_heads: int = 8             # query heads
    n_kv_heads: int = 2          # key-value heads (GQA: n_heads / n_kv_heads = 4)
    head_dim: int = 128          # dimension per head (d_model / n_heads)
    dropout: float = 0.0

    # ── Sparse MoE parameters ───────────────────────────────
    moe_every_n_layers: int = 4  # insert MoE every 4th layer
    n_experts: int = 8           # total experts
    n_experts_active: int = 2    # top-k experts per token
    expert_intermediate: int = 2048   # FFN intermediate dim
    moe_aux_loss_coef: float = 0.01   # load balancing loss weight

    # ── Precision per layer type ─────────────────────────────
    # Mamba layers  → 1.58-bit (ternary) weights
    # Attention     → 4-bit weights
    # Output head   → 8-bit weights
    # These are simulated during training via STE (Straight-Through Estimator)

    # ── Training ─────────────────────────────────────────────
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 2000
    max_steps: int = 100000
    batch_size: int = 8
    grad_clip: float = 1.0
    dtype: str = "float32"       # use bfloat16 on A100/T4

    # ── Parameter count targets ───────────────────────────────
    # 350M prototype (fits Kaggle T4 16GB)
    # 1B full model  (fits Kaggle A100 or colab pro)

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, \
            "d_model must be divisible by n_heads"
        assert self.n_heads % self.n_kv_heads == 0, \
            "n_heads must be divisible by n_kv_heads"
        assert self.head_dim == self.d_model // self.n_heads, \
            "head_dim must equal d_model // n_heads"

    @classmethod
    def phantom_350m(cls):
        """350M parameter prototype — fits Kaggle T4 16GB"""
        return cls(
            d_model=1024, n_layers=24, n_heads=8, n_kv_heads=2,
            head_dim=128, expert_intermediate=2048, n_experts=8
        )

    @classmethod
    def phantom_1b(cls):
        """1B parameter model — fits Kaggle A100 or Colab Pro"""
        return cls(
            d_model=2048, n_layers=32, n_heads=16, n_kv_heads=4,
            head_dim=128, expert_intermediate=4096, n_experts=8
        )

    @classmethod
    def phantom_medium(cls):
        """~50M parameter model — sweet spot for Kaggle T4 (16GB)
        Enough capacity to learn coherent English from TinyStories.
        Trains in ~3-4 hours on T4 with 50M tokens.
        """
        return cls(
            d_model=512, n_layers=12, n_heads=8, n_kv_heads=2,
            head_dim=64, d_state=16, expert_intermediate=1024,
            n_experts=4, n_experts_active=2,
            mamba_zone_end=3, mixed_zone_end=9,
            max_seq_len=512
        )

    @classmethod
    def phantom_tiny(cls):
        """Tiny model for quick testing — CPU friendly"""
        return cls(
            d_model=256, n_layers=8, n_heads=4, n_kv_heads=1,
            head_dim=64, d_state=8, expert_intermediate=512,
            n_experts=4, mamba_zone_end=2, mixed_zone_end=6,
            max_seq_len=512
        )