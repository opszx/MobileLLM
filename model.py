"""
PhantomLM — Full Model
Phone-native hybrid Mamba-Transformer architecture

Layer placement strategy (the key architectural novelty):
  Zone 1 (layers 0-5)   → pure Mamba only
  Zone 2 (layers 6-17)  → alternating: Mamba, Mamba, Attention, Mamba, Mamba, Attention...
  Zone 3 (layers 18-23) → attention-heavy: Mamba, Attention, Mamba, Attention, Mamba, Attention

MoE replaces FFN every 4th layer across all zones.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

from config import PhantomLMConfig
from mamba_block import MambaBlock
from attention import GroupedQueryAttention, precompute_freqs_cis, make_causal_mask
from moe import SparseMoE


# ── Standard FFN (for non-MoE layers) ─────────────────────────────────────────

class FeedForward(nn.Module):
    """Standard SwiGLU FFN used in non-MoE layers."""

    def __init__(self, config: PhantomLMConfig):
        super().__init__()
        from bitlinear import BitLinear
        d_ff = config.expert_intermediate
        self.gate = BitLinear(config.d_model, d_ff)
        self.up = BitLinear(config.d_model, d_ff)
        self.down = BitLinear(d_ff, config.d_model)
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, x):
        x_norm = self.norm(x)
        return self.down(F.silu(self.gate(x_norm)) * self.up(x_norm)) + x


# ── Hybrid Layer ──────────────────────────────────────────────────────────────

def compute_layer_types(config: PhantomLMConfig):
    """
    Compute the type of each layer based on the zone placement strategy.

    Zone 1 (0 → mamba_zone_end):     pure Mamba
    Zone 2 (mamba_zone_end → mixed_zone_end): every 3rd layer is Attention
    Zone 3 (mixed_zone_end → n_layers): alternating Mamba/Attention

    Returns list of 'mamba' or 'attention' for each layer index.
    """
    layer_types = []
    for l in range(config.n_layers):
        if l < config.mamba_zone_end:
            # Zone 1: all Mamba
            layer_types.append('mamba')
        elif l < config.mixed_zone_end:
            # Zone 2: every 3rd layer is Attention
            pos_in_zone = l - config.mamba_zone_end
            if pos_in_zone % 3 == 2:
                layer_types.append('attention')
            else:
                layer_types.append('mamba')
        else:
            # Zone 3: alternating (attention-heavy)
            pos_in_zone = l - config.mixed_zone_end
            if pos_in_zone % 2 == 1:
                layer_types.append('attention')
            else:
                layer_types.append('mamba')

    return layer_types


class HybridLayer(nn.Module):
    """
    One layer of PhantomLM.
    Can be either:
      - Mamba block + (FFN or MoE)
      - Attention block + (FFN or MoE)

    The FFN is replaced by MoE every config.moe_every_n_layers layers.
    """

    def __init__(self, config: PhantomLMConfig, layer_idx: int, layer_type: str):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = layer_type
        self.use_moe = (layer_idx % config.moe_every_n_layers == 0)

        # ── Core block: Mamba or Attention
        if layer_type == 'mamba':
            self.core = MambaBlock(config)
        else:
            self.core = GroupedQueryAttention(config)

        # ── FFN or MoE after core
        if self.use_moe:
            self.ffn = SparseMoE(config)
        else:
            self.ffn = FeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        use_checkpoint: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        from torch.utils.checkpoint import checkpoint as grad_ckpt

        def _core(x_):
            if self.layer_type == 'mamba':
                return self.core(x_)
            else:
                return self.core(x_, freqs_cis=freqs_cis, mask=mask)

        if use_checkpoint and self.training:
            x = grad_ckpt(_core, x, use_reentrant=False)
        else:
            x = _core(x)

        if use_checkpoint and self.training:
            x = grad_ckpt(self.ffn, x, use_reentrant=False)
        else:
            x = self.ffn(x)

        aux_loss = self.ffn.aux_loss if self.use_moe else None
        return x, aux_loss


# ── Full PhantomLM Model ──────────────────────────────────────────────────────

class PhantomLM(nn.Module):
    """
    PhantomLM: Phone-native hybrid 1.58-bit Mamba-Transformer.

    Architecture:
      Token embedding (8-bit)
        ↓
      24 HybridLayers (Mamba + Attention, selective precision, sparse MoE)
        ↓
      RMS norm
        ↓
      Output head (8-bit, tied to embedding)
        ↓
      Logits over vocabulary

    Parameter count:
      phantom_tiny:  ~30M  (testing)
      phantom_350m:  ~350M (Kaggle T4)
      phantom_1b:    ~1B   (Kaggle A100 / Colab Pro)
    """

    def __init__(self, config: PhantomLMConfig):
        super().__init__()
        self.config = config

        # ── Token embedding (8-bit in deployment, full precision during training)
        self.embed = nn.Embedding(config.vocab_size, config.d_model)

        # ── Compute which layers are Mamba vs Attention
        self.layer_types = compute_layer_types(config)

        # ── Build all layers
        self.layers = nn.ModuleList([
            HybridLayer(config, layer_idx=i, layer_type=self.layer_types[i])
            for i in range(config.n_layers)
        ])

        # ── Final normalization
        self.norm_f = nn.LayerNorm(config.d_model)

        # ── Output head (shared weights with embedding — saves ~200M params)
        # This is the "shared embedding with projection" architectural choice
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # Tie weights
        self.lm_head.weight = self.embed.weight

        # ── Precompute RoPE frequencies
        self.register_buffer(
            'freqs_cis',
            precompute_freqs_cis(config.head_dim, config.max_seq_len),
            persistent=False
        )

        # ── Initialize weights
        self.apply(self._init_weights)

        # Log architecture summary
        self._log_architecture()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, std=0.02)

    def _log_architecture(self):
        """Log layer placement for transparency."""
        mamba_count = self.layer_types.count('mamba')
        attn_count = self.layer_types.count('attention')
        moe_count = sum(1 for i in range(self.config.n_layers)
                       if i % self.config.moe_every_n_layers == 0)
        print(f"\n{'='*50}")
        print(f"PhantomLM Architecture")
        print(f"{'='*50}")
        print(f"Total layers  : {self.config.n_layers}")
        print(f"Mamba layers  : {mamba_count} ({mamba_count/self.config.n_layers*100:.0f}%)")
        print(f"Attn layers   : {attn_count} ({attn_count/self.config.n_layers*100:.0f}%)")
        print(f"MoE layers    : {moe_count}")
        print(f"d_model       : {self.config.d_model}")
        print(f"Parameters    : {self.count_parameters():,}")
        print(f"{'='*50}\n")
        print("Layer placement:")
        for i, lt in enumerate(self.layer_types):
            moe = " + MoE" if i % self.config.moe_every_n_layers == 0 else ""
            zone = ("Zone1" if i < self.config.mamba_zone_end
                    else "Zone2" if i < self.config.mixed_zone_end
                    else "Zone3")
            print(f"  Layer {i:2d} [{zone}]: {lt:9s}{moe}")
        print()

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        return_loss: bool = True,
        use_checkpoint: bool = False,
    ):
        B, L   = input_ids.shape
        device = input_ids.device

        x    = self.embed(input_ids)
        mask = make_causal_mask(L, device)

        total_aux_loss = torch.tensor(0.0, device=device)

        for layer in self.layers:
            x, aux_loss = layer(
                x, freqs_cis=self.freqs_cis, mask=mask,
                use_checkpoint=use_checkpoint
            )
            if aux_loss is not None:
                total_aux_loss = total_aux_loss + aux_loss

        x      = self.norm_f(x)
        logits = self.lm_head(x)

        if targets is None or not return_loss:
            return logits

        shift_logits  = logits[:, :-1, :].contiguous()
        shift_targets = targets[:, 1:].contiguous()

        lm_loss = F.cross_entropy(
            shift_logits.view(-1, self.config.vocab_size),
            shift_targets.view(-1),
            ignore_index=self.config.pad_token_id
        )

        total_loss = lm_loss + total_aux_loss

        return total_loss, logits, lm_loss, total_aux_loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_p: float = 0.9,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation with temperature and nucleus sampling.

        Args:
            input_ids: prompt token ids (B, L)
            max_new_tokens: how many tokens to generate
            temperature: > 1 more random, < 1 more focused
            top_p: nucleus sampling probability threshold
            eos_token_id: stop when this token is generated
        """
        self.eval()
        eos = eos_token_id or self.config.eos_token_id

        for _ in range(max_new_tokens):
            # Truncate context if too long
            idx_cond = input_ids[:, -self.config.max_seq_len:]

            # Forward pass
            logits = self.forward(idx_cond, return_loss=False)
            logits = logits[:, -1, :]   # last token only (B, vocab_size)

            # Temperature scaling
            logits = logits / temperature

            # Nucleus (top-p) sampling
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens above threshold
            remove_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p
            sorted_logits[remove_mask] = float('-inf')

            # Scatter back
            logits_filtered = torch.scatter(
                torch.full_like(logits, float('-inf')),
                1, sorted_idx, sorted_logits
            )

            # Sample
            probs = F.softmax(logits_filtered, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Stop at EOS
            if (next_token == eos).any():
                break

        return input_ids