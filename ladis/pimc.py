"""Perfect-Information Monte Carlo (PIMC) baseline using the DDS double-dummy solver.

For each decision: sample K deals of the unseen cards consistent with hand sizes and
publicly-known voids, solve each double-dummy, play the card with the best average.
This is the classic strong benchmark for trick-taking card play.
"""
from __future__ import annotations

import os
import random
from concurrent.futures import ProcessPoolExecutor

import torch

from .engine import NC, NS, NR, WhistBatch

_RANKS = "23456789TJQKA"


def _pbn(hands):
    # hands: list of 4 lists of card ids (our seat order 0..3 -> N E S W), our suit 0 -> spades
    out = []
    for h in hands:
        out.append(".".join("".join(_RANKS[c % 13] for c in sorted([c for c in h if c // 13 == s], reverse=True))
                            for s in range(4)))
    return "N:" + " ".join(out)


def sample_world(rng, me, my_hand, unseen, counts, void):
    """Assign unseen cards to the 3 other seats. counts[s]: cards seat s holds. void[s][suit]."""
    others = [s for s in range(4) if s != me]
    for _ in range(200):
        cards = list(unseen)
        rng.shuffle(cards)
        # place the most constrained cards first
        cards.sort(key=lambda c: sum(void[s][c // 13] for s in others), reverse=True)
        cap = {s: counts[s] for s in others}
        hands = {s: [] for s in others}
        ok = True
        for c in cards:
            cand = [s for s in others if cap[s] > 0 and not void[s][c // 13]]
            if not cand:
                ok = False
                break
            w = [cap[s] for s in cand]
            s = rng.choices(cand, weights=w)[0]
            hands[s].append(c)
            cap[s] -= 1
        if ok:
            hands[me] = list(my_hand)
            return [hands[s] for s in range(4)]
    return None


def decide(args):
    """Pure-python single decision (runs in worker processes)."""
    (me, my_hand, unseen, counts, void, trick, leader, legal, K, seed) = args
    from endplay.types import Deal, Denom, Player, Card
    from endplay.dds import solve_board
    rng = random.Random(seed)
    if len(legal) == 1:
        return legal[0]
    score = {c: 0.0 for c in legal}
    n = 0
    for _ in range(K):
        hands = sample_world(rng, me, my_hand, unseen, counts, void)
        if hands is None:
            continue
        # put current trick cards back into their owners' hands, then replay them
        for s, c in trick:
            hands[s].append(c)
        d = Deal.from_pbn(_pbn(hands))
        d.trump = Denom.spades
        d.first = Player(leader)  # Player.north == 0 ... west == 3 (same order as our seats)
        for s, c in trick:
            d.play(Card(f"{'SHDC'[c // 13]}{_RANKS[c % 13]}"))
        res = solve_board(d)
        got = {}
        for card, tr in res:
            cid = int(card.suit) * 13 + (int(card.rank).bit_length() - 3)
            got[cid] = tr
        for c in legal:
            score[c] += got.get(c, 0)
        n += 1
    best = max(legal, key=lambda c: (score[c], -(c % 13)))
    return best


class PIMC:
    """Arena-compatible policy. Call pimc(env, mask) -> actions."""

    def __init__(self, K=32, workers=None, seed=0):
        self.K = K
        self.seed = seed
        self.pool = ProcessPoolExecutor(workers or os.cpu_count())
        self.calls = 0
        self.wants_mask = True

    def __call__(self, env: WhistBatch, mask: torch.Tensor | None = None):
        B = env.B
        if mask is None:
            mask = torch.ones(B, dtype=torch.bool)
        legal = env.legal_mask().cpu()
        hands = env.hands.cpu()
        played = env.played_by.cpu()
        void = env.void.cpu()
        cur = env.cur.cpu()
        tc = env.trick_cards.cpu()
        leader = env.leader.cpu()
        n_in = env.n_in_trick.cpu()
        jobs, idx = [], []
        for b in mask.cpu().nonzero().flatten().tolist():
            me = int(cur[b])
            my_hand = hands[b, me].nonzero().flatten().tolist()
            seen = set(my_hand) | set(played[b].any(0).nonzero().flatten().tolist())
            unseen = [c for c in range(NC) if c not in seen]
            counts = [int(hands[b, s].sum()) for s in range(4)]
            vd = void[b].tolist()
            ld = int(leader[b])
            trick = [((ld + i) % 4, int(tc[b, (ld + i) % 4])) for i in range(int(n_in[b]))]
            # cards in the current trick were removed from hands & are in played: fine
            lg = legal[b].nonzero().flatten().tolist()
            self.calls += 1
            jobs.append((me, my_hand, unseen, counts, vd, trick, ld, lg, self.K,
                         hash((self.seed, self.calls)) & 0xFFFFFFFF))
            idx.append(b)
        out = torch.zeros(B, dtype=torch.long)
        if jobs:
            res = list(self.pool.map(decide, jobs, chunksize=max(1, len(jobs) // (4 * os.cpu_count()))))
            out[idx] = torch.tensor(res)
        return out.to(env.device)
