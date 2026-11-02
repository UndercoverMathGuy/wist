"""PPO self-play league for Ladis/Whist.

Each rollout plays B games in lockstep. Every game gets a seat->policy assignment:
  mode "self"     : learner in all four seats
  mode "team"     : learner pair vs an opponent pair (rule bot or frozen snapshot)
  mode "adhoc"    : learner + rule-bot partner vs opponent pair (rule / snapshot / learner)
Only learner seats produce training samples. Critic is an oracle (sees all hands).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import time

import torch
import torch.nn.functional as F

from .arena import duplicate
from .bots import random_bot, rule_bot
from .engine import NC, NS, NR, WhistBatch
from .features import obs_dim, encode
from .model import ActorCritic


def nn_policy(model, greedy=True):
    @torch.no_grad()
    def pol(env):
        obs = encode(env, hist=model.hist)
        legal = env.legal_mask()
        with torch.autocast(env.device.type, dtype=torch.bfloat16, enabled=env.device.type == "cuda"):
            lg = model.logits(obs, legal).float()
        if greedy:
            return lg.argmax(1)
        return torch.distributions.Categorical(logits=lg).sample()
    return pol


class League:
    def __init__(self, max_snaps=12):
        self.snaps: list[torch.nn.Module] = []
        self.max_snaps = max_snaps

    def add(self, model):
        m = copy.deepcopy(model).eval()
        for p in m.parameters():
            p.requires_grad_(False)
        self.snaps.append(m)
        if len(self.snaps) > self.max_snaps:
            # keep the first snapshot (diversity), drop the second-oldest
            self.snaps.pop(1)


@torch.no_grad()
def rollout(model, league: League, B, device, mix, gen):
    env = WhistBatch(B, device, gen=gen).reset()
    ar = env.ar
    # ---- assign policies. pid: 0 learner, 1 rule, 2+i snapshot i
    n_snap = len(league.snaps)
    u = torch.rand(B, device=device, generator=gen)
    mode = torch.where(u < mix[0], 0, torch.where(u < mix[0] + mix[1], 1, 2))
    # opponent id: rule or snapshot
    if n_snap > 0:
        opp = torch.where(torch.rand(B, device=device, generator=gen) < 0.4, 1,
                          2 + torch.randint(0, n_snap, (B,), device=device, generator=gen))
    else:
        opp = torch.ones(B, dtype=torch.long, device=device)
    # in adhoc mode the opponent may also be the learner itself
    opp_adhoc = torch.where(torch.rand(B, device=device, generator=gen) < 0.3, 0, opp)
    lteam = torch.randint(0, 2, (B,), device=device, generator=gen)
    lseat = lteam + 2 * torch.randint(0, 2, (B,), device=device, generator=gen)  # learner seat in adhoc
    pids = torch.zeros(B, NS, dtype=torch.long, device=device)
    seats = torch.arange(NS, device=device).unsqueeze(0)
    team_of = seats % 2
    is_lteam = team_of == lteam.unsqueeze(1)
    m1 = (mode == 1).unsqueeze(1)
    pids = torch.where(m1 & ~is_lteam, opp.unsqueeze(1), pids)
    m2 = (mode == 2).unsqueeze(1)
    pids = torch.where(m2 & ~is_lteam, opp_adhoc.unsqueeze(1), pids)
    pids = torch.where(m2 & is_lteam & (seats != lseat.unsqueeze(1)), torch.ones_like(pids), pids)

    D = obs_dim(model.hist)
    obs_buf = torch.zeros(NR, NS, B, D, dtype=torch.float16, device=device)
    oth_buf = torch.zeros(NR, NS, B, 3 * NC, dtype=torch.bool, device=device)
    leg_buf = torch.zeros(NR, NS, B, NC, dtype=torch.bool, device=device)
    act_buf = torch.zeros(NR, NS, B, dtype=torch.long, device=device)
    logp_buf = torch.zeros(NR, NS, B, device=device)
    val_buf = torch.zeros(NR, NS, B, device=device)
    rule_cache = None
    for t in range(NC):
        k = t // 4
        s = env.cur
        pid_cur = pids[ar, s]
        oobs = encode(env, oracle=True, hist=model.hist)
        obs = oobs[:, :D]
        legal = env.legal_mask()
        act = torch.zeros(B, dtype=torch.long, device=device)
        # learner
        m0 = pid_cur == 0
        if m0.any():
            idx = m0.nonzero().squeeze(1)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                lg = model.logits(obs[idx], legal[idx]).float()
                v = model.value(oobs[idx]).float()
            dist = torch.distributions.Categorical(logits=lg)
            a = dist.sample()
            act[idx] = a
            si = s[idx]
            obs_buf[k, si, idx] = obs[idx].half()
            oth_buf[k, si, idx] = oobs[idx, D:].bool()
            leg_buf[k, si, idx] = legal[idx]
            act_buf[k, si, idx] = a
            logp_buf[k, si, idx] = dist.log_prob(a)
            val_buf[k, si, idx] = v
        m1_ = pid_cur == 1
        if m1_.any():
            act = torch.where(m1_, rule_bot(env), act)
        for i, snap in enumerate(league.snaps):
            mi = pid_cur == 2 + i
            if mi.any():
                idx = mi.nonzero().squeeze(1)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    lg = snap.logits(obs[idx], legal[idx]).float()
                act[idx] = torch.distributions.Categorical(logits=lg).sample()
        env.step(act)
    # rewards: trick k outcome for seat s's team
    w = env.trick_winner_hist.T  # (13,B)
    rew = torch.where((w.unsqueeze(1) % 2) == (torch.arange(NS, device=device).view(1, NS, 1) % 2), 1.0, -1.0)
    lmask = (pids == 0).T.unsqueeze(0).expand(NR, NS, B)  # (13,4,B)
    stats = {
        "self_team_tricks": None,
    }
    # learner team result in mode 1 (vs rule/snapshot) as a cheap training signal
    lt = env.tricks[ar, lteam].float()
    for mval, name in ((1, "team"), (2, "adhoc")):
        mm = mode == mval
        if mm.any():
            stats[f"{name}_tricks"] = lt[mm].mean().item()
    return dict(obs=obs_buf, oth=oth_buf, leg=leg_buf, act=act_buf, logp=logp_buf, val=val_buf,
                rew=rew, mask=lmask), stats


def gae(val, rew, lam, gamma=1.0):
    NRk = val.shape[0]
    adv = torch.zeros_like(val)
    last = torch.zeros_like(val[0])
    for k in reversed(range(NRk)):
        nv = val[k + 1] if k + 1 < NRk else torch.zeros_like(val[0])
        delta = rew[k] + gamma * nv - val[k]
        last = delta + gamma * lam * last
        adv[k] = last
    return adv, adv + val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/dev")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--B", type=int, default=4096)
    ap.add_argument("--iters", type=int, default=100000)
    ap.add_argument("--hours", type=float, default=1e9)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--mb", type=int, default=16384)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--ent_final", type=float, default=0.002)
    ap.add_argument("--ent_iters", type=int, default=3000)
    ap.add_argument("--vcoef", type=float, default=0.5)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--mix", type=str, default="0.5,0.2,0.3")
    ap.add_argument("--snap_every", type=int, default=100)
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--eval_deals", type=int, default=8192)
    ap.add_argument("--resume", default="")
    ap.add_argument("--hist", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.run, exist_ok=True)
    dev = torch.device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    model = ActorCritic(args.width, args.depth, hist=args.hist).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
    league = League()
    it0 = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=dev)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        it0 = ck["iter"]
    league.add(model)
    mix = [float(x) for x in args.mix.split(",")]
    gen = None
    logf = open(os.path.join(args.run, "log.jsonl"), "a")
    t_start = time.time()
    n_samples = 0
    for it in range(it0, args.iters):
        t0 = time.time()
        model.eval()
        buf, st = rollout(model, league, args.B, dev, mix, gen)
        adv, ret = gae(buf["val"], buf["rew"], args.lam)
        m = buf["mask"]
        obs = buf["obs"][m]
        oth = buf["oth"][m]
        leg = buf["leg"][m]
        act = buf["act"][m]
        old_logp = buf["logp"][m]
        adv = adv[m]
        ret = ret[m]
        del buf
        N = obs.shape[0]
        n_samples += N
        adv_n = (adv - adv.mean()) / (adv.std() + 1e-8)
        frac = min(1.0, it / max(1, args.ent_iters))
        ent_coef = args.ent + (args.ent_final - args.ent) * frac
        model.train()
        stats = dict(pl=0.0, vl=0.0, ent=0.0, kl=0.0, cf=0.0, n=0)
        for ep in range(args.epochs):
            perm = torch.randperm(N, device=dev)
            for i in range(0, N, args.mb):
                j = perm[i:i + args.mb]
                o = obs[j].float()
                oo = torch.cat([o, oth[j].float()], 1)
                with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    lg = model.logits(o, leg[j]).float()
                    v = model.value(oo).float()
                logp_all = F.log_softmax(lg, 1)
                lp = logp_all.gather(1, act[j].unsqueeze(1)).squeeze(1)
                p_all = logp_all.exp()
                ent = -(p_all * torch.where(leg[j], logp_all, torch.zeros_like(logp_all))).sum(1).mean()
                ratio = (lp - old_logp[j]).exp()
                a = adv_n[j]
                pl = -torch.min(ratio * a, ratio.clamp(1 - args.clip, 1 + args.clip) * a).mean()
                vl = F.smooth_l1_loss(v, ret[j])
                loss = pl + args.vcoef * vl - ent_coef * ent
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                with torch.no_grad():
                    stats["pl"] += pl.item(); stats["vl"] += vl.item(); stats["ent"] += ent.item()
                    stats["kl"] += (old_logp[j] - lp).mean().item()
                    stats["cf"] += ((ratio - 1).abs() > args.clip).float().mean().item()
                    stats["n"] += 1
        nb = stats.pop("n")
        rec = {k: v / nb for k, v in stats.items()}
        rec.update(it=it, N=N, sps=N / (time.time() - t0), dt=time.time() - t0,
                   samples=n_samples, hours=(time.time() - t_start) / 3600, ent_coef=ent_coef, **st)
        if (it + 1) % args.snap_every == 0:
            league.add(model)
        if (it + 1) % args.eval_every == 0 or it == it0:
            model.eval()
            g = nn_policy(model, greedy=True)
            rule = rule_bot
            rec["ev_team_vs_rule"] = duplicate(g, rule, args.eval_deals, B=args.B, device=dev, seed=123)[0]
            rec["ev_adhoc_vs_rule"] = duplicate((g, rule), rule, args.eval_deals, B=args.B, device=dev, seed=123)[0]
            rec["ev_vs_first"] = duplicate(g, nn_policy(league.snaps[0]), args.eval_deals, B=args.B,
                                           device=dev, seed=123)[0]
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "iter": it + 1,
                        "args": vars(args)}, os.path.join(args.run, "ckpt.pt"))
            if (it + 1) % (args.eval_every * 10) == 0:
                torch.save({"model": model.state_dict(), "iter": it + 1, "args": vars(args)},
                           os.path.join(args.run, f"ckpt_{it + 1}.pt"))
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
        logf.write(json.dumps(rec) + "\n")
        logf.flush()
        if (time.time() - t_start) / 3600 > args.hours:
            break


if __name__ == "__main__":
    main()
