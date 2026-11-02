from __future__ import annotations

import torch
import torch.nn as nn

from .features import OBS_DIM, ORACLE_EXTRA


class ResBlock(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.ln = nn.LayerNorm(w)
        self.l1 = nn.Linear(w, w * 2)
        self.l2 = nn.Linear(w * 2, w)

    def forward(self, x):
        return x + self.l2(torch.relu(self.l1(self.ln(x))))


class MLP(nn.Module):
    def __init__(self, din, dout, width=512, depth=4):
        super().__init__()
        self.inp = nn.Linear(din, width)
        self.blocks = nn.Sequential(*[ResBlock(width) for _ in range(depth)])
        self.ln = nn.LayerNorm(width)
        self.out = nn.Linear(width, dout)

    def forward(self, x):
        return self.out(self.ln(self.blocks(self.inp(x))))


class ActorCritic(nn.Module):
    def __init__(self, width=512, depth=4, cwidth=512, cdepth=4):
        super().__init__()
        self.actor = MLP(OBS_DIM, 52, width, depth)
        self.critic = MLP(OBS_DIM + ORACLE_EXTRA, 1, cwidth, cdepth)
        nn.init.zeros_(self.actor.out.weight)
        nn.init.zeros_(self.actor.out.bias)

    def logits(self, obs, legal):
        lg = self.actor(obs)
        return lg.masked_fill(~legal, -1e9)

    def value(self, oobs):
        return self.critic(oobs).squeeze(-1)
