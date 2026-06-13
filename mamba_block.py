"""
Mamba Block — Selective State Space Model
Memory-efficient version for Kaggle T4 training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat

from bitlinear import BitLinear


def selective_scan(x, dt, A, B, C, D_skip):
    """
    Selective scan — uses PyTorch autograd for correct gradients.

    The previous custom autograd.Function had an incomplete backward pass
    (grad_dt and grad_B were always zero). This version lets PyTorch
    compute all gradients correctly through the recurrence.

    Memory cost: O(B * L * D_inner * N) for the computation graph.
    Acceptable for models up to ~350M on T4 (16GB).

    Args:
        x:      (B, L, D_inner)  — input after conv + activation
        dt:     (B, L, D_inner)  — learned time step
        A:      (D_inner, N)     — state transition (negative, log-parameterized)
        B:      (B, L, N)        — input-to-state projection
        C:      (B, L, N)        — state-to-output projection
        D_skip: (D_inner,)       — skip connection weight
    Returns:
        (B, L, D_inner)
    """
    B_size, L, D_inner = x.shape
    N = A.shape[1]

    # Discretize continuous parameters
    dt_A = dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)   # (B, L, D, N)
    A_bar = torch.exp(dt_A)                                  # (B, L, D, N)
    B_bar = dt.unsqueeze(-1) * B.unsqueeze(2)                # (B, L, D, N)

    # Sequential scan — autograd builds the graph through the loop
    h = torch.zeros(B_size, D_inner, N, device=x.device, dtype=x.dtype)
    ys = []
    for t in range(L):
        h = A_bar[:, t] * h + B_bar[:, t] * x[:, t].unsqueeze(-1)
        y_t = (h * C[:, t].unsqueeze(1)).sum(-1)
        ys.append(y_t)

    ys = torch.stack(ys, dim=1)                              # (B, L, D_inner)
    return ys + x * D_skip.unsqueeze(0).unsqueeze(0)         # skip connection


class MambaBlock(nn.Module):
    """
    Mamba selective SSM block.
    All projections use BitLinear (1.58-bit weights).
    """

    def __init__(self, config):
        super().__init__()
        self.d_model  = config.d_model
        self.d_state  = config.d_state
        self.d_conv   = config.d_conv
        self.d_inner  = config.mamba_expand * config.d_model

        self.in_proj  = BitLinear(self.d_model, self.d_inner * 2)

        self.conv1d   = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=self.d_conv,
            padding=self.d_conv - 1,
            groups=self.d_inner, bias=True
        )

        self.dt_rank  = max(1, self.d_inner // 16)
        self.x_proj   = BitLinear(self.d_inner, self.d_state * 2 + self.dt_rank)
        self.dt_proj  = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32),
            'n -> d n', d=self.d_inner
        )
        self.A_log    = nn.Parameter(torch.log(A))
        self.D        = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = BitLinear(self.d_inner, self.d_model)
        self.norm     = nn.LayerNorm(self.d_model)
        self._init_weights()

    def _init_weights(self):
        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001))
            + math.log(0.001)
        )
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B_size, L, D = x.shape
        residual = x
        x        = self.norm(x)

        xz               = self.in_proj(x)
        x_inner, z       = xz.chunk(2, dim=-1)

        x_inner          = rearrange(x_inner, 'b l d -> b d l')
        x_inner          = self.conv1d(x_inner)[..., :L]
        x_inner          = rearrange(x_inner, 'b d l -> b l d')
        x_inner          = F.silu(x_inner)

        x_proj_out       = self.x_proj(x_inner)
        B_proj, C_proj, dt_raw = x_proj_out.split(
            [self.d_state, self.d_state, self.dt_rank], dim=-1
        )
        dt               = F.softplus(self.dt_proj(dt_raw))
        A                = -torch.exp(self.A_log.float())

        y = selective_scan(
            x_inner.float(), dt.float(), A,
            B_proj.float(), C_proj.float(), self.D.float()
        ).to(x.dtype)

        y    = y * F.silu(z)
        out  = self.out_proj(y)
        return out + residual