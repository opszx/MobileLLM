# BitLinear 1.58-bit + Linear4bit + STE
"""
BitLinear — 1.58-bit weight quantization
Weights constrained to ternary {-1, 0, +1}
Activations quantized to 8-bit integers

Based on: "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits"
Novel contribution: applied selectively only to Mamba layers in PhantomLM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Straight-Through Estimator ────────────────────────────────────────────────
# During forward pass  → use quantized weights (efficient inference)
# During backward pass → treat as if weights were full precision (stable gradients)

class STEFunction(torch.autograd.Function):
    """Straight-Through Estimator for quantization."""
    @staticmethod
    def forward(ctx, x, x_quantized):
        return x_quantized

    @staticmethod
    def backward(ctx, grad_output):
        # Pass gradient through unchanged — STE trick
        return grad_output, None


def ste_round(x):
    """Round with straight-through gradient."""
    return STEFunction.apply(x, torch.round(x))


def ste_clamp(x, min_val, max_val):
    """Clamp with straight-through gradient."""
    return STEFunction.apply(x, torch.clamp(x, min_val, max_val))


# ── Ternary weight quantization ───────────────────────────────────────────────

def quantize_weights_ternary(weight: torch.Tensor):
    """
    Quantize weights to ternary {-1, 0, +1}.
    
    Steps:
      1. Compute scale = mean(|W|)  (absmean scaling)
      2. W_scaled = W / scale
      3. W_ternary = RoundClip(W_scaled, -1, 1)
    
    Returns: (W_ternary, scale)
    """
    scale = weight.abs().mean() + 1e-8
    w_normalized = weight / scale
    w_ternary = ste_clamp(ste_round(w_normalized), -1.0, 1.0)
    return w_ternary, scale


def quantize_activations_8bit(x: torch.Tensor):
    """
    Quantize activations to 8-bit integers.
    
    x_quant = Clamp(Round(x * 127 / max(|x|)), -128, 127)
    
    Returns: (x_quantized, scale)
    """
    scale = x.abs().max() + 1e-8
    x_scaled = x / scale * 127.0
    x_quant = ste_clamp(ste_round(x_scaled), -128.0, 127.0)
    return x_quant, scale


# ── BitLinear Layer ───────────────────────────────────────────────────────────

class BitLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear with 1.58-bit weights.
    
    Forward pass:
      1. LayerNorm input (stabilizes ternary computation)
      2. Quantize weights to ternary {-1, 0, +1}
      3. Quantize activations to 8-bit
      4. Compute W̃ · x̃ (becomes additions, no multiplications)
      5. Rescale output

    Used in: all Mamba blocks and MoE expert FFNs in PhantomLM
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Full precision weights — quantized during forward pass via STE
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features)
        )
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        # Pre-norm before quantized computation (critical for stability)
        self.norm = nn.LayerNorm(in_features, elementwise_affine=False)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values — important for ternary quantization."""
        nn.init.trunc_normal_(self.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, in_features)
        returns: (batch, seq_len, out_features)
        """
        # Step 1 — normalize input
        x_norm = self.norm(x)

        # Step 2 — quantize weights to ternary
        w_ternary, w_scale = quantize_weights_ternary(self.weight)

        # Step 3 — quantize activations to 8-bit
        x_quant, x_scale = quantize_activations_8bit(x_norm)

        # Step 4 — linear projection (ternary weights make this very cheap on device)
        # In training: uses full precision via STE
        # On device: becomes integer additions
        out = F.linear(x_quant / 127.0, w_ternary)

        # Step 5 — rescale back to original magnitude
        out = out * w_scale * x_scale

        if self.bias is not None:
            out = out + self.bias

        return out

    def extra_repr(self):
        return (f"in={self.in_features}, out={self.out_features}, "
                f"precision=1.58-bit (ternary)")


# ── 4-bit Linear for Attention layers ────────────────────────────────────────

def quantize_weights_4bit(weight: torch.Tensor):
    """
    Quantize weights to 4-bit integers [-8, 7].
    Used for attention layers which need more precision than ternary.
    """
    scale = weight.abs().max() + 1e-8
    w_scaled = weight / scale * 7.0
    w_quant = ste_clamp(ste_round(w_scaled), -8.0, 7.0)
    return w_quant, scale


class Linear4bit(nn.Module):
    """
    4-bit weight quantization for attention Q, K, V, O projections.
    More precise than BitLinear — attention scores need more range.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.norm = nn.LayerNorm(in_features, elementwise_affine=False)
        nn.init.trunc_normal_(self.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm(x)
        w_quant, w_scale = quantize_weights_4bit(self.weight)
        out = F.linear(x_norm, w_quant / 7.0 * w_scale)
        if self.bias is not None:
            out = out + self.bias
        return out

    def extra_repr(self):
        return f"in={self.in_features}, out={self.out_features}, precision=4-bit"