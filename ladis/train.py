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
from .bots import rule_bot
from .engine import NC, NS, NR, WhistBatch
from .model import ActorCritic, obs_index


def nn_policy(model, greedy=True):
    @torch.no_grad()
    def pol(env):
        obs = model.observe(env)
        legal = env.legal_mask()
        with torch.autocast(env.device.type, dtype=torch.bfloat16, enabled=env.device.type == "cuda"):
            lg = model.logits(obs, legal).float()
        if greedy:
            return lg.argmax(1)
        return torch.distributions.Categorical(logits=lg).sample()
    return pol


class ExitReplay:
    """Replay of search decisions written by `ladis.exit gen --follow` workers."""

    def __init__(self, path, cap, device):
        self.path, self.cap, self.device = path, cap, device
        self.obs, self.act, self.legal = None, None, None
        self.total = 0
        os.makedirs(path, exist_ok=True)

    def poll(self):
        import glob
        for f in sorted(glob.glob(os.path.join(self.path, "shard_*.pt"))):
            try:
                d = torch.load(f, map_location=self.device)
            except Exception:
                continue
            os.remove(f)
            keep = d["legal"].sum(1) > 1  # forced moves teach nothing
            obs = {k: v[keep] for k, v in d["obs"].items()}
            act = d["act"][keep]
            legal = d["legal"][keep]
            if self.obs is None:
                self.obs, self.act, self.legal = obs, act, legal
            else:
                self.obs = {k: torch.cat([self.obs[k], obs[k]])[-self.cap:] for k in obs}
                self.act = torch.cat([self.act, act])[-self.cap:]
                self.legal = torch.cat([self.legal, legal])[-self.cap:]
            self.total += len(act)

    def __len__(self):
        return 0 if self.act is None else len(self.act)

    def sample(self, n):
        j = torch.randint(0, len(self), (n,), device=self.device)
        return {k: v[j] for k, v in self.obs.items()}, self.act[j], self.legal[j]


class League:
    def __init__(self, max_snaps=12, save_dir=""):
        self.snaps: list[torch.nn.Module] = []
        self.max_snaps = max_snaps
        self.save_dir = save_dir

    def load_saved(self, template):
        import glob
        if not self.save_dir:
            return
        for f in sorted(glob.glob(os.path.join(self.save_dir, "snap_*.pt")))[-self.max_snaps:]:
            m = copy.deepcopy(template)
            m.load_state_dict(torch.load(f, map_location=next(template.parameters()).device))
            self.add(m, save=False)

    def add(self, model, save=True, tag=None):
        if save and self.save_dir and tag is not None:
            os.makedirs(self.save_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(self.save_dir, f"snap_{tag:07d}.pt"))
        m = copy.deepcopy(model).eval()
        for attr in ("_core_compiled", "_core_graph"):  # deepcopy keeps the *learner's* compiled fns: drop
            if getattr(getattr(m, "actor", None), attr, None) is not None:
                setattr(m.actor, attr, None)
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

    obs_buf = None  # dict of (13,4,B,...) buffers, allocated on first observation
    leg_buf = torch.zeros(NR, NS, B, NC, dtype=torch.bool, device=device)
    act_buf = torch.zeros(NR, NS, B, dtype=torch.long, device=device)
    logp_buf = torch.zeros(NR, NS, B, device=device)
    val_buf = torch.zeros(NR, NS, B, device=device)
    for t in range(NC):
        k = t // 4
        s = env.cur
        pid_cur = pids[ar, s]
        obs = model.observe(env, oracle=True)
        if obs_buf is None:
            obs_buf = {k: torch.zeros((NR, NS) + v.shape, dtype=v.dtype, device=device) for k, v in obs.items()}
        legal = env.legal_mask()
        act = torch.zeros(B, dtype=torch.long, device=device)
        # learner
        m0 = pid_cur == 0
        if m0.any():
            idx = m0.nonzero().squeeze(1)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                oi = obs_index(obs, idx)
                lg = model.logits(oi, legal[idx]).float()
                v = model.value(oi).float()
            dist = torch.distributions.Categorical(logits=lg)
            a = dist.sample()
            act[idx] = a
            si = s[idx]
            for kk, vv in oi.items():
                obs_buf[kk][k, si, idx] = vv
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
                    lg = snap.logits(obs_index(obs, idx), legal[idx]).float()
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
    return dict(obs=obs_buf, leg=leg_buf, act=act_buf, logp=logp_buf, val=val_buf,
                rew=rew, mask=lmask), stats


def micro_chunks(jb, obs, micro, len_sort):
    """Split minibatch indices into micro-batches. With len_sort (transformer), sort by history
    length and cut chunks to a fixed *token* budget, so each chunk is truncated to its own longest
    history (exactly the same gradient as unsorted; just no compute spent on padding)."""
    if not len_sort or "hcard" not in obs:
        return [jb[c:c + micro] for c in range(0, len(jb), micro)]
    from .features import PAD_CARD
    lens = (obs["hcard"][jb] != PAD_CARD).sum(1)
    order = lens.argsort()
    jb, lens = jb[order], lens[order].cpu().numpy()
    padded = 14 + ((lens + 3) // 4) * 4  # CLS + 13 hand + history rounded to whole tricks
    budget = micro * (14 + NC)
    out, st, n = [], 0, len(jb)
    while st < n:
        lo, hi = st + 1, n  # largest e with (e-st)*padded[e-1] <= budget
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if (mid - st) * padded[mid - 1] <= budget:
                lo = mid
            else:
                hi = mid - 1
        out.append(jb[st:lo])
        st = lo
    return out


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
    ap.add_argument("--lr_end", type=float, default=-1.0, help="linear decay to this by --hours (default: constant)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--mb", type=int, default=16384)
    ap.add_argument("--micro", type=int, default=0, help="micro-batch for grad accumulation (0 = mb)")
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
    ap.add_argument("--arch", default="mlp", choices=["mlp", "tf"])
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--len_sort", type=int, default=1, help="length-sorted token-budget micro-batches")
    ap.add_argument("--compile", type=int, default=0, help="torch.compile the transformer core")
    ap.add_argument("--cudagraph", type=int, default=0, help="CUDA-graph the inference (rollout) path")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exit_dir", default="", help="dir where search workers drop shards")
    ap.add_argument("--exit_coef", type=float, default=1.0)
    ap.add_argument("--exit_cap", type=int, default=400000)
    ap.add_argument("--exit_min", type=int, default=4096)
    ap.add_argument("--target_kl", type=float, default=0.0, help="stop epochs once minibatch KL exceeds this")
    ap.add_argument("--exit_start", type=int, default=0, help="iteration from which the imitation loss is on")
    ap.add_argument("--latest_every", type=int, default=10, help="save latest.pt for search workers")
    args = ap.parse_args()

    os.makedirs(args.run, exist_ok=True)
    dev = torch.device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    model = ActorCritic(args.width, args.depth, hist=args.hist, arch=args.arch, d=args.d,
                        layers=args.layers, heads=args.heads).to(dev)
    if args.compile and args.arch == "tf":
        model.actor.compile_core()
    if args.cudagraph and args.arch == "tf" and dev.type == "cuda":
        model.actor.enable_cudagraphs()
    import logging  # silence harmless symbolic-shape warnings from the compiler
    logging.getLogger("torch.utils._sympy.interp").setLevel(logging.ERROR)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5, fused=dev.type == "cuda")
    league = League(save_dir=os.path.join(args.run, "snaps"))
    it0 = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=dev)
        model.load_state_dict(ck["model"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        it0 = ck["iter"]
    league.load_saved(model)
    if not league.snaps:
        league.add(model, tag=it0)
    replay = ExitReplay(args.exit_dir, args.exit_cap, dev) if args.exit_dir else None

    def save_latest(it_):
        tmp = os.path.join(args.run, "latest.pt.tmp")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "iter": it_, "args": vars(args)}, tmp)
        os.replace(tmp, os.path.join(args.run, "latest.pt"))
    save_latest(it0)
    mix = [float(x) for x in args.mix.split(",")]
    gen = None
    logf = open(os.path.join(args.run, "log.jsonl"), "a")
    t_start = time.time()
    n_samples = 0
    import signal
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    for it in range(it0, args.iters):
        frac_t = min(1.0, (time.time() - t_start) / 3600 / max(1e-6, args.hours))
        lr_now = args.lr if args.lr_end < 0 else args.lr + (args.lr_end - args.lr) * frac_t
        for g in opt.param_groups:  # also overrides the lr stored in a resumed optimizer state
            g["lr"] = lr_now
        if stop["flag"]:
            save_latest(it)
            print(json.dumps({"stopped": "SIGTERM", "saved_iter": it}), flush=True)
            break
        t0 = time.time()
        model.eval()
        buf, st = rollout(model, league, args.B, dev, mix, gen)
        adv, ret = gae(buf["val"], buf["rew"], args.lam)
        m = buf["mask"]
        obs = {k: v[m] for k, v in buf["obs"].items()}
        leg = buf["leg"][m]
        act = buf["act"][m]
        old_logp = buf["logp"][m]
        adv = adv[m]
        ret = ret[m]
        del buf
        N = act.shape[0]
        n_samples += N
        adv_n = (adv - adv.mean()) / (adv.std() + 1e-8)
        frac = min(1.0, it / max(1, args.ent_iters))
        ent_coef = args.ent + (args.ent_final - args.ent) * frac
        model.train()
        stats = dict(pl=0.0, vl=0.0, ent=0.0, kl=0.0, cf=0.0, xl=0.0, xa=0.0, n=0)
        if replay is not None:
            replay.poll()
        use_x = replay is not None and len(replay) >= args.exit_min and it >= args.exit_start
        early_stop = False
        for ep in range(args.epochs):
            if early_stop:
                break
            perm = torch.randperm(N, device=dev)
            for i in range(0, N, args.mb):
                if early_stop:
                    break
                kl_mb, n_mb = 0.0, 0
                jb = perm[i:i + args.mb]
                opt.zero_grad(set_to_none=True)
                micro = args.micro or args.mb
                for j in micro_chunks(jb, obs, micro, args.len_sort):  # gradient accumulation (memory)
                    w = len(j) / len(jb)
                    o = obs_index(obs, j)
                    with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                        lg = model.logits(o, leg[j]).float()
                        v = model.value(o).float()
                    logp_all = F.log_softmax(lg, 1)
                    lp = logp_all.gather(1, act[j].unsqueeze(1)).squeeze(1)
                    p_all = logp_all.exp()
                    ent = -(p_all * torch.where(leg[j], logp_all, torch.zeros_like(logp_all))).sum(1).mean()
                    ratio = (lp - old_logp[j]).exp()
                    a = adv_n[j]
                    pl = -torch.min(ratio * a, ratio.clamp(1 - args.clip, 1 + args.clip) * a).mean()
                    vl = F.smooth_l1_loss(v, ret[j])
                    loss = pl + args.vcoef * vl - ent_coef * ent
                    if use_x:  # imitate the search's chosen card (expert iteration)
                        xo, xa, xleg = replay.sample(max(1024, len(j) // 2))
                        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                            xlg = model.logits(xo, xleg).float()
                        xl = F.cross_entropy(xlg, xa)
                        loss = loss + args.exit_coef * xl
                        with torch.no_grad():
                            stats["xl"] += w * xl.item(); stats["xa"] += w * (xlg.argmax(1) == xa).float().mean().item()
                    (loss * w).backward()
                    with torch.no_grad():  # sample-weighted (chunks differ in size under len_sort)
                        stats["pl"] += w * pl.item(); stats["vl"] += w * vl.item(); stats["ent"] += w * ent.item()
                        kl_c = (old_logp[j] - lp).mean().item()
                        stats["kl"] += w * kl_c
                        kl_mb += w * kl_c; n_mb += w
                        stats["cf"] += w * ((ratio - 1).abs() > args.clip).float().mean().item()
                stats["n"] += 1
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                if args.target_kl > 0 and kl_mb / max(1, n_mb) > args.target_kl:
                    early_stop = True  # standard PPO safeguard: stop updating on this batch
        nb = stats.pop("n")
        rec = {k: v / max(1, nb) for k, v in stats.items()}
        rec["early_stop"] = early_stop
        if replay is not None:
            rec.update(x_buf=len(replay), x_total=replay.total)
        if (it + 1) % args.latest_every == 0:
            save_latest(it + 1)
        rec["lr"] = lr_now
        rec.update(it=it, N=N, sps=N / (time.time() - t0), dt=time.time() - t0,
                   samples=n_samples, hours=(time.time() - t_start) / 3600, ent_coef=ent_coef, **st)
        if (it + 1) % args.snap_every == 0:
            league.add(model, tag=it + 1)
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
