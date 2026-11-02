"""Test-time search on top of a trained policy.

For the player to act:
 1. sample N worlds (hidden-card layouts) consistent with hand sizes + known voids;
 2. replay the whole game in every world and score how likely the *observed* plays of the
    other three players were under the policy (Bayesian card reading), tempered by beta
    and mixed with a uniform floor so non-policy opponents don't zero out worlds;
 3. resample M worlds by those weights;
 4. for every legal card: play it in every world and roll the game out with the policy
    for all four seats (each acting on its own information) -> expected team tricks;
 5. play the best card.
"""
from __future__ import annotations

import random

import torch
import torch.nn.functional as F

from .engine import NC, NS, WhistBatch, select
from .features import encode
from .pimc import sample_world


class SearchPolicy:
    wants_mask = True

    def __init__(self, model, N=128, M=32, beta=0.5, floor=0.1, seed=0, max_batch=32768):
        self.model = model
        self.N, self.M, self.beta, self.floor = N, M, beta, floor
        self.rng = random.Random(seed)
        self.max_batch = max_batch

    @torch.no_grad()
    def _logits(self, W):
        obs = encode(W, hist=self.model.hist)
        legal = W.legal_mask()
        dev = W.device
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            lg = self.model.logits(obs, legal).float()
        return lg, legal

    @torch.no_grad()
    def __call__(self, env: WhistBatch, mask=None):
        dev = env.device
        B = env.B
        if mask is None:
            mask = torch.ones(B, dtype=torch.bool, device=dev)
        games = mask.nonzero().flatten()
        out = torch.zeros(B, dtype=torch.long, device=dev)
        # chunk games so that rollout batch stays bounded
        per = max(1, self.max_batch // (self.M * 13))
        q = torch.full((B, NC), float("nan"), device=dev)
        for i in range(0, len(games), per):
            g = games[i:i + per]
            out[g] = self._search(env, g)
            _, cand, sc, sv = self.last
            qq = torch.full((len(g), NC), float("nan"), device=dev)
            qq.scatter_(1, torch.where(sv, cand, cand[:, :1]), torch.where(sv, sc, sc[:, :1]))
            q[g] = qq
        self.q = q  # (B,52) expected final team tricks per legal card (nan = not evaluated)
        return out

    @torch.no_grad()
    def _search(self, env, games):
        dev = env.device
        G, N, M = len(games), self.N, self.M
        cur = env.cur[games].cpu()
        hands = env.hands[games].cpu()
        void = env.void[games].cpu()
        owner_real = env.owner[games].cpu()
        played_any = env.played_by[games].any(1).cpu()
        owners = torch.empty(G * N, NC, dtype=torch.long)
        failed = torch.zeros(G * N, dtype=torch.bool)
        for gi in range(G):
            me = int(cur[gi])
            my_hand = hands[gi, me].nonzero().flatten().tolist()
            unseen = [c for c in range(NC) if not played_any[gi, c] and not hands[gi, me, c]]
            counts = [int(hands[gi, s].sum()) for s in range(NS)]
            vd = void[gi].tolist()
            base = owner_real[gi].clone()  # played cards keep their true owner; my cards are mine
            for n in range(N):
                w = sample_world(self.rng, me, my_hand, unseen, counts, vd)
                o = base.clone()
                if w is None:  # never fall back to the true layout
                    failed[gi * N + n] = True
                    o[[c for c in unseen]] = (me + 1) % NS
                else:
                    for s in range(NS):
                        if s != me:
                            o[w[s]] = s
                owners[gi * N + n] = o
        rep = torch.arange(G, device=dev).repeat_interleave(N)
        W = WhistBatch(G * N, dev).reset(leader=env.first_leader[games][rep], deal=owners.to(dev))
        me_rep = env.cur[games][rep]
        loglik = torch.zeros(G * N, device=dev)
        for t in range(env.step_no):
            a = env.hist_card[games, t][rep]
            if self.beta > 0:
                lg, legal = self._logits(W)
                p = F.softmax(lg, 1)
                nl = legal.sum(1, keepdim=True).float()
                p = (1 - self.floor) * p + self.floor * legal.float() / nl
                lp = torch.log(p.gather(1, a.unsqueeze(1)).squeeze(1) + 1e-12)
                loglik += torch.where(W.cur != me_rep, lp, torch.zeros_like(lp))
            W.step(a)
        loglik = loglik.masked_fill(failed.to(dev), -1e9)
        wts = F.softmax((self.beta * loglik).view(G, N), 1)
        pick = torch.multinomial(wts, M, replacement=True)  # (G,M)
        R = select(W, (pick + torch.arange(G, device=dev).unsqueeze(1) * N).flatten())
        # candidate actions = my legal cards, padded to 13 slots
        legal_g = env.legal_mask()[games]  # (G,52)
        order = torch.where(legal_g, torch.arange(NC, device=dev), NC + torch.arange(NC, device=dev)).argsort(1)[:, :13]
        nleg = legal_g.sum(1)
        slot_valid = torch.arange(13, device=dev).unsqueeze(0) < nleg.unsqueeze(1)
        cand = torch.where(slot_valid, order, order[:, :1])  # (G,13)
        idx = torch.arange(G * M, device=dev).repeat_interleave(13)
        E = select(R, idx)
        acts = cand.unsqueeze(1).expand(G, M, 13).reshape(-1)
        my_team = (env.cur[games] % 2).repeat_interleave(M * 13)
        E.step(acts)
        while not E.finished:
            lg, _ = self._logits(E)
            E.step(lg.argmax(1))
        sc = E.tricks.gather(1, my_team.unsqueeze(1)).squeeze(1).float().view(G, M, 13).mean(1)
        sc = torch.where(slot_valid, sc, torch.full_like(sc, -1e9))
        best = sc.argmax(1)
        self.last = (games, cand, sc, slot_valid)  # exposed for expert-iteration targets
        return cand.gather(1, best.unsqueeze(1)).squeeze(1)
