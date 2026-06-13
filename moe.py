# Sparse Mixture of Experts + load balancing
"""
Sparse Mixture of Experts (MoE)
Inserted every 4th layer in PhantomLM

Design:
  - 8 experts total, only top-2 activate per token
  - Router: small linear network that learns which expert handles what
  - Load balancing loss prevents expert collapse
  - All expert FFNs use BitLinear (1.58-bit weights)

Why MoE enables generality:
  Different experts naturally specialize during training:
    Expert 1-2 → reasoning/math
    Expert 3-4 → language/writing
    Expert 5-6 → code
    Expert 7-8 → factual recall
  The router selects the right ones per input — automatically.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from bitlinear import BitLinear


class ExpertFFN(nn.Module):
    """
    Single expert — a standard FFN with BitLinear weights.
    Uses SwiGLU activation (better than ReLU for language models).
    """

    def __init__(self, d_model: int, d_intermediate: int):
        super().__init__()
        # SwiGLU: gate_proj and up_proj together → element-wise multiply
        self.gate_proj = BitLinear(d_model, d_intermediate)
        self.up_proj = BitLinear(d_model, d_intermediate)
        self.down_proj = BitLinear(d_intermediate, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: swish(gate) * up
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class SparseMoE(nn.Module):
    """
    Sparse Mixture of Experts layer.

    Forward pass:
      1. Router scores all 8 experts for this token
      2. Select top-2 experts by score
      3. Route token to those 2 experts
      4. Weighted sum of expert outputs
      5. Compute load balancing auxiliary loss

    Key equations:
      r(x)    = Softmax(x · W_router)           — expert scores
      G(x)    = Top2(r(x)) / ΣTop2(r(x))        — normalized gates
      y       = Σ_{i∈top2} G_i(x) · FFN_i(x)   — expert combination
      L_aux   = N_e · Σ_i f_i · P_i             — load balance loss
    """

    def __init__(self, config):
        super().__init__()
        self.n_experts = config.n_experts
        self.n_active = config.n_experts_active   # top-k (=2)
        self.d_model = config.d_model
        self.aux_loss_coef = config.moe_aux_loss_coef

        # ── Router: small linear, 8-bit precision (routing decisions matter)
        self.router = nn.Linear(config.d_model, config.n_experts, bias=False)

        # ── Expert pool
        self.experts = nn.ModuleList([
            ExpertFFN(config.d_model, config.expert_intermediate)
            for _ in range(config.n_experts)
        ])

        # ── Pre-norm
        self.norm = nn.LayerNorm(config.d_model)

        # Storage for aux loss (accessed during training loop)
        self.aux_loss = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, D)
        returns: (B, L, D) with self.aux_loss set for training
        """
        B, L, D = x.shape
        residual = x
        x = self.norm(x)

        # Flatten to (B*L, D) for routing
        x_flat = x.view(-1, D)   # (T, D) where T = B*L

        # ── Step 1: Router scores
        router_logits = self.router(x_flat)   # (T, n_experts)
        router_probs = F.softmax(router_logits, dim=-1)

        # ── Step 2: Select top-k experts
        top_k_probs, top_k_idx = torch.topk(router_probs, self.n_active, dim=-1)
        # (T, n_active) for both

        # ── Step 3: Normalize gates (so they sum to 1)
        gates = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # ── Step 4: Route tokens to experts
        output = torch.zeros_like(x_flat)   # (T, D)

        for i in range(self.n_active):
            expert_idx = top_k_idx[:, i]    # (T,) — which expert for each token
            gate_weight = gates[:, i]        # (T,) — weight for this expert

            # Process each expert
            for e_id in range(self.n_experts):
                # Find tokens assigned to expert e_id at position i
                token_mask = (expert_idx == e_id)
                if not token_mask.any():
                    continue

                expert_input = x_flat[token_mask]          # (n_tokens, D)
                expert_output = self.experts[e_id](expert_input)  # (n_tokens, D)

                # Weight by gate score
                output[token_mask] += gate_weight[token_mask].unsqueeze(-1) * expert_output

        # ── Compute load balancing auxiliary loss
        self.aux_loss = self._compute_aux_loss(router_probs, top_k_idx, B * L)

        # Reshape back and add residual
        output = output.view(B, L, D)
        return output + residual

    def _compute_aux_loss(
        self,
        router_probs: torch.Tensor,  # (T, n_experts)
        top_k_idx: torch.Tensor,     # (T, n_active)
        total_tokens: int
    ) -> torch.Tensor:
        """
        Load balancing loss — prevents all tokens routing to same experts.

        L_aux = N_e · Σ_i f_i · P_i
          f_i = fraction of tokens dispatched to expert i
          P_i = mean router probability for expert i
        """
        # f_i: fraction of tokens going to each expert
        # Create one-hot from top-k selections
        expert_counts = torch.zeros(
            self.n_experts, device=router_probs.device
        )
        for k in range(self.n_active):
            for e_id in range(self.n_experts):
                expert_counts[e_id] += (top_k_idx[:, k] == e_id).sum()

        f = expert_counts / (total_tokens * self.n_active)  # normalize

        # P_i: mean router probability for expert i
        P = router_probs.mean(dim=0)   # (n_experts,)

        # Auxiliary loss
        aux_loss = self.n_experts * (f * P).sum()
        return self.aux_loss_coef * aux_loss

    def extra_repr(self):
        return (f"n_experts={self.n_experts}, n_active={self.n_active}, "
                f"expert_precision=1.58-bit, router_precision=8-bit")