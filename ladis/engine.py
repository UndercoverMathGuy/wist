"""Batched, GPU-friendly Whist ("Ladis") engine.

Conventions
-----------
* Card index c in [0, 52): suit = c // 13, rank = c % 13 (0 = deuce ... 12 = ace).
* Trump is ALWAYS suit 0. Trump is chosen uniformly at random in the real game and
  carries no information, so canonicalising it to suit 0 loses nothing.
* Seats 0..3, teams = seat % 2 (0&2 vs 1&3).
* Every game is exactly 52 decisions long, so a batch of games advances in lockstep.
* "Hand" in the user's terminology == "trick" here.
"""
from __future__ import annotations

import torch

NC = 52
NS = 4
NR = 13
SUIT = torch.arange(NC) // NR
RANK = torch.arange(NC) % NR


class WhistBatch:
    def __init__(self, B: int, device="cpu", gen: torch.Generator | None = None):
        self.B = B
        self.device = torch.device(device)
        self.gen = gen
        self.suit = SUIT.to(self.device)
        self.rank = RANK.to(self.device)
        self.ar = torch.arange(B, device=self.device)

    # ------------------------------------------------------------------ setup
    def reset(self, leader: torch.Tensor | None = None, deal: torch.Tensor | None = None):
        B, dev = self.B, self.device
        if deal is None:
            perm = torch.argsort(torch.rand(B, NC, device=dev, generator=self.gen), dim=1)
            owner = torch.empty(B, NC, dtype=torch.long, device=dev)
            owner.scatter_(1, perm, (torch.arange(NC, device=dev) // NR).expand(B, NC))
        else:
            owner = deal.to(dev).long()  # (B, 52) seat that holds each card
        self.owner = owner
        self.hands = torch.zeros(B, NS, NC, dtype=torch.bool, device=dev)
        self.hands.scatter_(1, owner.unsqueeze(1), True)
        self.played_by = torch.zeros(B, NS, NC, dtype=torch.bool, device=dev)
        self.led_by = torch.zeros(B, NS, NC, dtype=torch.bool, device=dev)
        self.discard_by = torch.zeros(B, NS, NC, dtype=torch.bool, device=dev)  # off-suit plays (incl. ruffs)
        self.void = torch.zeros(B, NS, NS, dtype=torch.bool, device=dev)  # publicly known voids
        self.trick_cards = torch.full((B, NS), -1, dtype=torch.long, device=dev)
        if leader is None:
            leader = torch.randint(0, NS, (B,), device=dev, generator=self.gen)
        self.leader = leader.to(dev).long().clone()
        self.first_leader = self.leader.clone()
        self.n_in_trick = torch.zeros(B, dtype=torch.long, device=dev)
        self.trick_no = 0
        self.step_no = 0
        self.lead_suit = torch.full((B,), -1, dtype=torch.long, device=dev)
        self.tricks = torch.zeros(B, 2, dtype=torch.long, device=dev)
        self.hist_card = torch.full((B, NC), -1, dtype=torch.long, device=dev)
        self.hist_seat = torch.full((B, NC), -1, dtype=torch.long, device=dev)
        self.trick_winner_hist = torch.full((B, NR), -1, dtype=torch.long, device=dev)
        return self

    # --------------------------------------------------------------- queries
    @property
    def cur(self) -> torch.Tensor:
        return (self.leader + self.n_in_trick) % NS

    def legal_mask(self) -> torch.Tensor:
        hand = self.hands[self.ar, self.cur]  # (B,52)
        leading = self.n_in_trick == 0
        ls = self.lead_suit.clamp(min=0)
        in_suit = hand & (self.suit.unsqueeze(0) == ls.unsqueeze(1))
        has = in_suit.any(1)
        must = (~leading) & has
        return torch.where(must.unsqueeze(1), in_suit, hand)

    def winning_seat(self) -> torch.Tensor:
        """Seat currently winning the (partial) trick. Undefined if trick empty."""
        tc = self.trick_cards
        valid = tc >= 0
        tcc = tc.clamp(min=0)
        s = self.suit[tcc]
        r = self.rank[tcc]
        val = torch.where(s == 0, 100 + r, torch.where(s == self.lead_suit.unsqueeze(1), r, -1))
        val = torch.where(valid, val, -2)
        return val.argmax(1)

    # ------------------------------------------------------------------ step
    def step(self, action: torch.Tensor):
        """Play `action` (B,) card for the current player. Returns (trick_done, winner_team or -1)."""
        B = self.B
        ar = self.ar
        p = self.cur
        a = action.long()
        # legality check (cheap, keep it on)
        lm = self.legal_mask()
        if not bool(lm[ar, a].all()):
            bad = (~lm[ar, a]).nonzero()[:5].flatten().tolist()
            raise ValueError(f"illegal action in games {bad}")
        leading = self.n_in_trick == 0
        self.lead_suit = torch.where(leading, self.suit[a], self.lead_suit)
        # void inference: failed to follow suit
        off = (~leading) & (self.suit[a] != self.lead_suit)
        self.void[ar, p, self.lead_suit.clamp(min=0)] |= off
        self.hands[ar, p, a] = False
        self.played_by[ar, p, a] = True
        self.led_by[ar, p, a] |= leading
        self.discard_by[ar, p, a] |= off
        self.trick_cards[ar, p] = a
        self.hist_card[:, self.step_no] = a
        self.hist_seat[:, self.step_no] = p
        self.step_no += 1
        self.n_in_trick += 1
        done = bool((self.n_in_trick == 4).all())
        if done:
            w = self.winning_seat()
            team = w % 2
            self.tricks[ar, team] += 1
            self.trick_winner_hist[:, self.trick_no] = w
            self.trick_no += 1
            self.leader = w
            self.n_in_trick.zero_()
            self.trick_cards.fill_(-1)
            self.lead_suit.fill_(-1)
            return True, team
        return False, None

    @property
    def finished(self) -> bool:
        return self.step_no >= NC


def select(env: WhistBatch, idx: torch.Tensor) -> WhistBatch:
    """New batch made of games env[idx] (deep copy of per-game state)."""
    out = WhistBatch(len(idx), env.device, gen=env.gen)
    for k, v in env.__dict__.items():
        if isinstance(v, torch.Tensor) and v.dim() >= 1 and v.shape[0] == env.B and k not in ("suit", "rank", "ar"):
            setattr(out, k, v[idx].clone())
    out.trick_no = env.trick_no
    out.step_no = env.step_no
    return out
