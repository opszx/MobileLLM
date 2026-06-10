# Grouped Query Attention + RoPE
"""
Grouped Query Attention (GQA) with Rotary Position Embeddings (RoPE)
Used in ~25% of PhantomLM layers (Zone 2 every 3rd, Zone 3 alternating)

GQA: multiple query heads share fewer KV heads
  n_heads=8, n_kv_heads=2 → each KV head serves 4 query heads
  Saves memory + compute while preserving quality

RoPE: encodes position via rotation of Q/K vectors
  No learned position embeddings needed

All Q/K/V/O projections use Linear4bit (4-bit weights)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

from bitlinear import Linear4bit


# ── Rotary Position Embedding (RoPE) ─────────────────────────────────────────

def precompute_freqs_cis(head_dim: int, max_seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Precompute complex exponentials for RoPE.
    
    For each position t and dimension pair d:
      freq = 1 / (theta^(2d/head_dim))
      freqs_cis[t, d] = exp(i * t * freq) = cos(t*freq) + i*sin(t*freq)
    
    Returns: (max_seq_len, head_dim // 2) complex tensor
    """
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)  # (max_seq_len, head_dim // 2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple:
    """
    Apply RoPE to query and key tensors.
    
    xq, xk: (B, n_heads, L, head_dim)
    freqs_cis: (max_seq_len, head_dim // 2) — sliced to L
    
    Pairs adjacent dims → complex, multiplies by rotation, unpairs.
    """
    # Reshape to complex: (B, n_heads, L, head_dim) → (B, n_heads, L, head_dim//2, 2) → complex
    xq_r = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_r = xk.float().reshape(*xk.shape[:-1], -1, 2)
    xq_c = torch.view_as_complex(xq_r)
    xk_c = torch.view_as_complex(xk_r)

    # Slice freqs_cis to sequence length and reshape for broadcasting
    L = xq.shape[2]
    freqs = freqs_cis[:L]  # (L, head_dim // 2)
    freqs = freqs.unsqueeze(0).unsqueeze(0)  # (1, 1, L, head_dim // 2)

    # Apply rotation
    xq_out = torch.view_as_real(xq_c * freqs).flatten(-2)
    xk_out = torch.view_as_real(xk_c * freqs).flatten(-2)

    return xq_out.type_as(xq), xk_out.type_as(xk)


# ── Causal Mask ──────────────────────────────────────────────────────────────

def make_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Create causal attention mask.
    
    Returns: (1, 1, L, L) boolean mask where True = masked (cannot attend).
    Upper triangle is True (future positions masked).
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, L, L)


# ── Grouped Query Attention ──────────────────────────────────────────────────

class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention with RoPE.
    
    Architecture:
      Input x (B, L, D)
        → pre-norm
        → Q projection → split into n_heads
        → K, V projection → split into n_kv_heads
        → K, V repeated to match n_heads (GQA grouping)
        → RoPE on Q, K
        → scaled dot-product attention with causal mask
        → concatenate heads → O projection
        → residual connection
    
    All projections use Linear4bit (4-bit quantized weights).
    """

    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.n_rep = self.n_heads // self.n_kv_heads  # how many Q heads per KV head
        self.d_model = config.d_model
        self.dropout = config.dropout

        # ── Q/K/V/O projections (4-bit quantized)
        self.wq = Linear4bit(self.d_model, self.n_heads * self.head_dim)
        self.wk = Linear4bit(self.d_model, self.n_kv_heads * self.head_dim)
        self.wv = Linear4bit(self.d_model, self.n_kv_heads * self.head_dim)
        self.wo = Linear4bit(self.n_heads * self.head_dim, self.d_model)

        # ── Pre-norm
        self.norm = nn.LayerNorm(self.d_model)

        # ── Attention dropout
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: (B, L, D)
        freqs_cis: precomputed RoPE frequencies
        mask: causal mask (1, 1, L, L) where True = masked
        returns: (B, L, D) with residual connection
        """
        B, L, D = x.shape
        residual = x

        # Pre-norm
        x = self.norm(x)

        # ── Project to Q, K, V
        q = self.wq(x)  # (B, L, n_heads * head_dim)
        k = self.wk(x)  # (B, L, n_kv_heads * head_dim)
        v = self.wv(x)  # (B, L, n_kv_heads * head_dim)

        # ── Reshape to (B, n_heads, L, head_dim)
        q = q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # ── Apply RoPE to Q and K
        if freqs_cis is not None:
            q, k = apply_rotary_emb(q, k, freqs_cis)

        # ── GQA: repeat K, V to match number of query heads
        # (B, n_kv_heads, L, head_dim) → (B, n_heads, L, head_dim)
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # ── Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, n_heads, L, L)

        # Apply causal mask
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # ── Attend to values
        attn_output = torch.matmul(attn_weights, v)  # (B, n_heads, L, head_dim)

        # ── Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, -1)

        # ── Output projection
        out = self.wo(attn_output)
        out = self.resid_dropout(out)

        # Residual connection
        return out + residual