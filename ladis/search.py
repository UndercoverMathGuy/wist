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

import math
import random

import torch
import torch.nn.functional as F

from .engine import NC, NS, WhistBatch, select
from .pimc import sample_world


def sample_world_weighted(rng, me, my_hand, unseen, counts, void, prob, tries=200):
    """Like pimc.sample_world, but each unseen card goes to a seat with probability proportional to
    the belief head's P(seat holds card) x remaining capacity (respecting voids and hand sizes).
    Returns (hands, logw): logw = log target - log proposal, where the target is the product of the
    belief marginals restricted to consistent layouts. prob[s][c] for absolute seats s != me."""
    others = [s for s in range(4) if s != me]
    for _ in range(tries):
        cards = list(unseen)
        rng.shuffle(cards)
        # order: cards with a single feasible seat first, then the cards the belief is most sure about,
        # so the proposal follows the beliefs instead of fighting them (keeps importance weights flat)
        def key(c):
            feas = [s for s in others if not void[s][c // 13]]
            return (len(feas) > 1, -max(prob[s][c] for s in feas) if feas else 0.0)
        cards.sort(key=key)
        cap = {s: counts[s] for s in others}
        hands = {s: [] for s in others}
        logw, ok = 0.0, True
        for c in cards:
            cand = [s for s in others if cap[s] > 0 and not void[s][c // 13]]
            if not cand:
                ok = False
                break
            q = [max(prob[s][c], 1e-6) * cap[s] for s in cand]
            z = sum(q)
            s = rng.choices(cand, weights=q)[0]
            logw += math.log(max(prob[s][c], 1e-6)) - math.log(q[cand.index(s)] / z)
            hands[s].append(c)
            cap[s] -= 1
        if ok:
            hands[me] = list(my_hand)
            return [hands[s] for s in range(4)], logw
    return None, 0.0


class SearchPolicy:
    wants_mask = True

    def __init__(self, model, N=128, M=32, beta=0.5, floor=0.1, seed=0, max_batch=32768, z=2.0, margin=0.0,
                 mode="rollout", prior="uniform"):
        self.model = model
        self.N, self.M, self.beta, self.floor = N, M, beta, floor
        self.rng = random.Random(seed)
        self.max_batch = max_batch
        self.z, self.margin = z, margin
        self.n_dev = 0
        self.n_dec = 0
        self.ess_log = []
        self.mode = mode  # "rollout": play every world out with the policy; "critic": 1-step + value net
        # "uniform": hidden cards dealt uniformly at random (PIMC-style, constraints only);
        # "belief": dealt from the network's belief head (who holds which card), importance-weighted
        self.prior = prior

    @torch.no_grad()
    def _logits(self, W):
        obs = self.model.observe(W)
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
        per = max(1, self.max_batch // (self.N * 13))
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
        logw = torch.zeros(G * N)
        bel_abs = None
        if self.prior == "belief":  # belief marginals for the player to act, mapped to absolute seats
            Es = select(env, games)
            ob = self.model.observe(Es)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                _, bel = self.model.logits(ob, Es.legal_mask(), return_belief=True)
            p_rel = F.softmax(bel.float().view(G, 3, NC), 1).cpu()  # rel seats 1..3 = LHO, partner, RHO
            bel_abs = torch.zeros(G, NS, NC)
            for r in range(3):
                bel_abs[torch.arange(G), (cur + r + 1) % NS] = p_rel[:, r]
            bel_abs = bel_abs.tolist()
        for gi in range(G):
            me = int(cur[gi])
            my_hand = hands[gi, me].nonzero().flatten().tolist()
            unseen = [c for c in range(NC) if not played_any[gi, c] and not hands[gi, me, c]]
            counts = [int(hands[gi, s].sum()) for s in range(NS)]
            vd = void[gi].tolist()
            base = owner_real[gi].clone()  # played cards keep their true owner; my cards are mine
            for n in range(N):
                if bel_abs is not None:
                    w, lw = sample_world_weighted(self.rng, me, my_hand, unseen, counts, vd, bel_abs[gi])
                    logw[gi * N + n] = lw
                else:
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
        # importance weights over ALL sampled worlds (no resampling -> no duplicate worlds);
        # failed samples are masked AFTER scaling so they get zero weight even when beta == 0
        score_w = (self.beta * loglik + logw.to(dev)).masked_fill(failed.to(dev), -1e9)
        wts = F.softmax(score_w.view(G, N), 1)  # (G,N)
        M = N
        R = W
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
        if self.mode == "critic":
            # Q = trick margin banked so far + critic's estimate of the remaining margin, seen from
            # my team (critic values are from the next actor's team, so flip sign for opponents)
            opp_team = 1 - my_team
            banked = (E.tricks.gather(1, my_team.unsqueeze(1)) - E.tricks.gather(1, opp_team.unsqueeze(1))).squeeze(1).float()
            if E.finished:
                fut = torch.zeros_like(banked)
            else:
                ob = self.model.observe(E, oracle=True)
                with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    v = self.model.value(ob).float()
                fut = torch.where((E.cur % 2) == my_team, v, -v)
            R = (banked + fut).view(G, M, 13) / 2.0  # margin/2 ~ "tricks" units
        else:
            while not E.finished:
                lg, _ = self._logits(E)
                E.step(lg.argmax(1))
            R = E.tricks.gather(1, my_team.unsqueeze(1)).squeeze(1).float().view(G, M, 13)
        sc = torch.where(slot_valid, (wts.unsqueeze(2) * R).sum(1), torch.full_like(R[:, 0], -1e9))
        # default = the network's own greedy card; deviate only on a significant paired gain
        lg0, _ = self._logits(select(env, games))
        a0 = lg0.argmax(1)
        slot0 = (cand == a0.unsqueeze(1)).float().argmax(1)
        diff = R - R.gather(2, slot0.view(G, 1, 1).expand(G, M, 1))
        w = wts.unsqueeze(2)  # (G,M,1)
        mu = (w * diff).sum(1)
        var = (w * (diff - mu.unsqueeze(1)) ** 2).sum(1)
        ess = 1.0 / (wts ** 2).sum(1, keepdim=True)  # effective number of worlds
        se = (var / ess.clamp(min=1.0)).sqrt()
        self.ess_log.append(float(ess.mean()))
        ok = slot_valid & (mu > self.z * se + self.margin) & (mu > 0)
        score = torch.where(ok, mu, torch.full_like(mu, -1e9))
        dev_best = score.argmax(1)
        any_ok = ok.any(1)
        best = torch.where(any_ok, dev_best, slot0)
        self.n_dev += int(any_ok.sum()); self.n_dec += G
        self.last = (games, cand, sc, slot_valid)  # exposed for expert-iteration targets
        return cand.gather(1, best.unsqueeze(1)).squeeze(1)
