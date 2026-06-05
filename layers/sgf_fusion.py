import torch
import torch.nn as nn
import torch.nn.functional as F


class Fusion(nn.Module):
    def __init__(self, d_series, d_core):
        super().__init__()

        self.norm = nn.LayerNorm(d_series)

        self.gen1 = nn.Linear(d_series, d_series)
        self.gen2 = nn.Linear(d_series, d_core)

        self.gate = nn.Linear(d_core, d_series)

        self.proj = nn.Sequential(
            nn.Linear(d_series, d_series),
            nn.GELU(),
            nn.Linear(d_series, d_series)
        )

    def forward(self, x):

        residual = x

        x = self.norm(x)

        # channel encoding
        core = F.gelu(self.gen1(x))
        core = self.gen2(core)

        # differentiable stochastic pooling
        weight = F.gumbel_softmax(
            core,
            tau=1.0,
            dim=1,
            hard=False
        )

        global_feat = torch.sum(
            core * weight,
            dim=1,
            keepdim=True
        )

        # adaptive gating
        gate = torch.sigmoid(
            self.gate(global_feat)
        )

        out = x * gate

        out = self.proj(out)

        return residual + out