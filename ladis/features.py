"""Observation encoding from the acting player's point of view (relative seats).

rel 0 = me, 1 = left-hand opponent, 2 = partner, 3 = right-hand opponent.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .engine import NC, NS, NR, WhistBatch

OBS_DIM = 52 + 3 * 4 * 52 + 3 * 52 + 4 + 4 + 12 + 52 + 52 + 1 + 4 + 13 + 2 + 52
ORACLE_EXTRA = 3 * 52
HIST_DIM = NR * NS * NC + NR * NS + NR * NS


def obs_dim(hist: bool) -> int:
    return OBS_DIM + (HIST_DIM if hist else 0)


def _rel_gather(x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """x: (B,4,...) by absolute seat -> (B,4,...) by relative seat from p."""
    B = x.shape[0]
    idx = (p.unsqueeze(1) + torch.arange(NS, device=x.device).unsqueeze(0)) % NS  # (B,4)
    ar = torch.arange(B, device=x.device).unsqueeze(1)
    return x[ar, idx]


def encode(env: WhistBatch, oracle: bool = False, hist: bool = False) -> torch.Tensor:
    B, dev = env.B, env.device
    p = env.cur
    ar = env.ar
    hand = env.hands[ar, p].float()
    played = _rel_gather(env.played_by, p).float().reshape(B, -1)
    led = _rel_gather(env.led_by, p).float().reshape(B, -1)
    disc = _rel_gather(env.discard_by, p).float().reshape(B, -1)
    tc = _rel_gather(env.trick_cards, p)[:, 1:]  # (B,3)
    tco = F.one_hot(tc.clamp(min=0), NC).float() * (tc >= 0).unsqueeze(-1).float()
    tco = tco.reshape(B, -1)
    ls = env.lead_suit
    lso = F.one_hot(ls.clamp(min=0), NS).float() * (ls >= 0).unsqueeze(-1).float()
    pos = F.one_hot(env.n_in_trick, NS).float()
    void = _rel_gather(env.void, p)[:, 1:].float().reshape(B, -1)
    all_played = env.played_by.any(1)
    unseen = (~env.hands[ar, p]) & (~all_played)
    # winning card so far
    nonempty = env.n_in_trick > 0
    ws = env.winning_seat()
    wcard = env.trick_cards[ar, ws].clamp(min=0)
    wco = F.one_hot(wcard, NC).float() * nonempty.unsqueeze(-1).float()
    wrel = (ws - p) % NS
    partner_win = (nonempty & (wrel == 2)).float().unsqueeze(1)
    wrelo = F.one_hot(wrel, NS).float() * nonempty.unsqueeze(-1).float()
    tno = F.one_hot(torch.full((B,), min(env.trick_no, NR - 1), device=dev), NR).float()
    myteam = p % 2
    us = env.tricks[ar, myteam].float() / 13.0
    them = env.tricks[ar, 1 - myteam].float() / 13.0
    # master: card not played and no unplayed higher card in its suit
    remaining = (~all_played).reshape(B, NS, NR)
    higher = torch.flip(torch.cumsum(torch.flip(remaining.int(), [2]), 2), [2]) - remaining.int()
    master = (remaining & (higher == 0)).reshape(B, NC).float()
    obs = torch.cat(
        [hand, played, led, disc, tco, lso, pos, void, unseen.float(), wco, partner_win, wrelo, tno,
         us.unsqueeze(1), them.unsqueeze(1), master], 1)
    if hist:
        hc = env.hist_card.view(B, NR, NS)  # play order within trick
        hs = env.hist_seat.view(B, NR, NS)
        valid = hc >= 0
        rel = (hs - p.view(B, 1, 1)) % NS
        # slot (trick, rel seat) -> card one-hot
        h = torch.zeros(B, NR, NS, NC, device=dev)
        h.scatter_(3, torch.where(valid, hc, 0).unsqueeze(3), valid.float().unsqueeze(3))
        # h currently indexed by play position; reorder by relative seat
        h2 = torch.zeros(B, NR, NS, NC, device=dev)
        h2.scatter_add_(2, rel.unsqueeze(3).expand(B, NR, NS, NC), h)
        lead_rel = F.one_hot(rel[:, :, 0], NS).float() * valid[:, :, 0:1].float()
        tw = env.trick_winner_hist
        win_rel = F.one_hot(((tw - p.unsqueeze(1)) % NS), NS).float() * (tw >= 0).unsqueeze(2).float()
        obs = torch.cat([obs, h2.reshape(B, -1), lead_rel.reshape(B, -1), win_rel.reshape(B, -1)], 1)
    if oracle:
        others = _rel_gather(env.hands, p)[:, 1:].float().reshape(B, -1)
        obs = torch.cat([obs, others], 1)
    return obs
