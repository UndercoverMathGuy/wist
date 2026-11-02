"""Duplicate-match evaluation.

A policy is a callable env -> (B,) actions for the current player of every game.
`play` routes each game's current seat to the policy assigned to that seat.
"""
from __future__ import annotations

import math
from typing import Callable, Sequence

import torch

from .engine import NC, WhistBatch

Policy = Callable[[WhistBatch], torch.Tensor]


@torch.no_grad()
def play(env: WhistBatch, seat_policies: Sequence[Policy]) -> torch.Tensor:
    """Play full games. Returns tricks (B,2)."""
    B = env.B
    while not env.finished:
        cur = env.cur
        act = torch.empty(B, dtype=torch.long, device=env.device)
        # call each distinct policy once on the whole batch, take what we need
        uniq = {}
        for s, pol in enumerate(seat_policies):
            uniq.setdefault(id(pol), (pol, []))[1].append(s)
        for pol, seats in uniq.values():
            m = torch.zeros(B, dtype=torch.bool, device=env.device)
            for s in seats:
                m |= cur == s
            if m.any():
                a = pol(env, m) if getattr(pol, "wants_mask", False) else pol(env)
                act = torch.where(m, a, act)
        env.step(act)
    return env.tricks.clone()


def duplicate(polA: Sequence[Policy] | Policy, polB: Sequence[Policy] | Policy, n_deals: int,
              B: int = 4096, device="cpu", seed: int = 0):
    """Team A (pair of policies for (seat0, seat2)) vs team B. Every deal is played twice
    with the hands swapped between teams. Returns mean tricks/game for A minus 6.5, and stderr."""
    if callable(polA):
        polA = (polA, polA)
    if callable(polB):
        polB = (polB, polB)
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    diffs = []
    done = 0
    while done < n_deals:
        b = min(B, n_deals - done)
        env = WhistBatch(b, device, gen=g)
        env.reset()
        deal, leader = env.owner.clone(), env.leader.clone()
        t1 = play(env, [polA[0], polB[0], polA[1], polB[1]])[:, 0]
        env2 = WhistBatch(b, device, gen=g)
        env2.reset(leader=leader, deal=deal)
        t2 = play(env2, [polB[0], polA[0], polB[1], polA[1]])[:, 1]
        diffs.append(((t1 + t2).float() / 2 - 6.5).cpu())
        done += b
    d = torch.cat(diffs)
    return d.mean().item(), d.std().item() / math.sqrt(len(d))
