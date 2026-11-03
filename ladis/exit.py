"""Expert iteration: search plays, network imitates the search's card values.

Generation: games where every seat uses SearchPolicy (on top of the current net). For each
decision we store the acting player's observation, legal mask and the search's Q-values.
Training: actor gets cross-entropy to softmax(Q/tau) over legal cards (plus a little of
its own entropy); critic is left alone. Then PPO can continue from the distilled net.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import torch
import torch.nn.functional as F

from .engine import WhistBatch
from .eval import load
from .search import SearchPolicy


@torch.no_grad()
def generate(model, games, out_dir, N, M, beta, device, shard_games=64, seed=0, mode="rollout",
             follow_ckpt=""):
    """follow_ckpt: reload this checkpoint before every shard (async expert iteration)."""
    os.makedirs(out_dir, exist_ok=True)
    sp = SearchPolicy(model, N=N, M=M, beta=beta, seed=seed, mode=mode)
    mtime = None
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    done = 0
    k = len(glob.glob(os.path.join(out_dir, "shard_*.pt")))
    while done < games:
        if follow_ckpt:
            while not os.path.exists(follow_ckpt):
                time.sleep(10)
            mt = os.path.getmtime(follow_ckpt)
            if mt != mtime:
                try:
                    sp.model = load(follow_ckpt, device)
                    mtime = mt
                except Exception as e:  # checkpoint mid-write; retry next shard
                    print(json.dumps({"reload_error": str(e)[:200]}), flush=True)
        t = time.time()
        env = WhistBatch(shard_games, device, gen=g).reset()
        O, L, Q, A = [], [], [], []
        while not env.finished:
            obs = model.observe(env)
            legal = env.legal_mask()
            a = sp(env)
            O.append({k: v.cpu() for k, v in obs.items()}); L.append(legal.cpu()); Q.append(sp.q.half().cpu()); A.append(a.cpu())
            env.step(a)
        print(json.dumps({"deviations": sp.n_dev, "decisions": sp.n_dec}), flush=True)
        path = os.path.join(out_dir, f"shard_{seed}_{k:05d}.pt")
        torch.save({"obs": {k2: torch.cat([o[k2] for o in O]) for k2 in O[0]}, "legal": torch.cat(L),
                    "q": torch.cat(Q), "act": torch.cat(A)}, path + ".tmp")
        os.replace(path + ".tmp", path)  # atomic: readers never see half-written shards
        k += 1
        done += shard_games
        print(json.dumps({"gen_games": done, "secs": round(time.time() - t, 1)}), flush=True)


def distill(ckpt, data_dir, out, device, tau=0.25, epochs=2, lr=1e-4, mb=8192, target_kind="action"):
    ck = torch.load(ckpt, map_location=device)
    model = load(ckpt, device).train()
    shards = sorted(glob.glob(os.path.join(data_dir, "shard_*.pt")))
    data = [torch.load(s) for s in shards]
    obs = {k: torch.cat([d["obs"][k] for d in data]).to(device) for k in data[0]["obs"]}
    legal = torch.cat([d["legal"] for d in data]).to(device)
    q = torch.cat([d["q"] for d in data]).float().to(device)
    act = torch.cat([d["act"] for d in data]).to(device)
    keep = legal.sum(1) > 1  # forced moves teach nothing
    obs = {k: v[keep] for k, v in obs.items()}
    legal, q, act = legal[keep], q[keep], act[keep]
    q = torch.where(legal & ~q.isnan(), q, torch.full_like(q, -1e4))
    if target_kind == "action":  # imitate the (significance-gated) card the search played
        target = F.one_hot(act, 52).float()
    else:  # soft target from the search's card values
        target = F.softmax(q / tau, 1)
    opt = torch.optim.Adam(model.actor.parameters(), lr=lr)
    n = len(act)
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        tot, agree = 0.0, 0.0
        for i in range(0, n, mb):
            j = perm[i:i + mb]
            lg = model.logits({k: v[j] for k, v in obs.items()}, legal[j])
            lp = F.log_softmax(lg, 1)
            loss = -(target[j] * torch.where(legal[j], lp, torch.zeros_like(lp))).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(j)
            agree += (lg.argmax(1) == act[j]).float().sum().item()
        print(json.dumps({"epoch": ep, "loss": tot / n, "agree": agree / n, "n": n}), flush=True)
    ck["model"] = model.state_dict()
    ck.pop("opt", None)
    torch.save(ck, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["gen", "distill"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--games", type=int, default=10000)
    ap.add_argument("--shard_games", type=int, default=64)
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--tau", type=float, default=0.25)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--target", default="action", choices=["action", "q"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--search_mode", default="rollout", choices=["rollout", "critic"])
    ap.add_argument("--follow", action="store_true", help="reload --ckpt before every shard")
    a = ap.parse_args()
    if a.mode == "gen":
        if a.follow:
            while not os.path.exists(a.ckpt):
                time.sleep(10)
        generate(load(a.ckpt, a.device), a.games, a.data, a.N, a.M, a.beta, a.device, a.shard_games, a.seed,
                 mode=a.search_mode, follow_ckpt=a.ckpt if a.follow else "")
    else:
        distill(a.ckpt, a.data, a.out, a.device, a.tau, a.epochs, target_kind=a.target)


if __name__ == "__main__":
    main()
