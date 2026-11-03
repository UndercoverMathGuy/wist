"""Evaluate checkpoints against baselines (incl. slow PIMC).

python -m ladis.eval --ckpt runs/v1/ckpt.pt --deals 400 --pimc_k 16
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

from .arena import duplicate
from .bots import rule_bot, random_bot
from .model import ActorCritic, load_state_compat
from .pimc import PIMC
from .train import nn_policy


def load(path, device="cpu"):
    ck = torch.load(path, map_location=device)
    a = ck.get("args", {})
    m = ActorCritic(a.get("width", 512), a.get("depth", 4), hist=a.get("hist", False),
                    arch=a.get("arch", "mlp"), d=a.get("d", 256), layers=a.get("layers", 6),
                    heads=a.get("heads", 8)).to(device)
    load_state_compat(m, ck["model"])
    return m.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--deals", type=int, default=400)
    ap.add_argument("--pimc_k", type=int, default=16)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", default="")
    ap.add_argument("--B", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--nn_B", type=int, default=4000, help="GPU batch for the cheap nn-vs-rule eval")
    ap.add_argument("--search", action="store_true", help="evaluate SearchPolicy instead of the raw net")
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--M", type=int, default=32)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--prior", default="uniform", choices=["uniform", "belief"])
    ap.add_argument("--mode", default="rollout", choices=["rollout", "critic"])
    ap.add_argument("--nn_deals", type=int, default=0, help="deals for search-vs-network (default: --deals)")
    args = ap.parse_args()
    torch.set_num_threads(2)
    m = load(args.ckpt, args.device)
    g = nn_policy(m)
    pm = PIMC(K=args.pimc_k, workers=args.workers)
    res = {"ckpt": args.ckpt, "deals": args.deals, "pimc_k": args.pimc_k}
    t = time.time()
    D = dict(B=args.B, seed=args.seed, device=args.device)
    if args.search:
        from .search import SearchPolicy
        sp = SearchPolicy(m, N=args.N, M=args.M, beta=args.beta, prior=args.prior, mode=args.mode)
        res.update(search=dict(N=args.N, beta=args.beta, prior=args.prior, mode=args.mode))

        def stage(key, fn):  # write each matchup as soon as it finishes (partial results survive)
            t0 = time.time()
            res[key] = fn()
            res[key + "_secs"] = round(time.time() - t0)
            print(json.dumps({key: res[key], "secs": res[key + "_secs"]}), flush=True)
            if args.out:
                with open(args.out + ".partial", "a") as f:
                    f.write(json.dumps({"stage": key, "result": res[key], "secs": res[key + "_secs"],
                                        "search": res["search"], "ckpt": args.ckpt}) + "\n")
        stage("search_vs_nn", lambda: duplicate(sp, g, args.nn_deals or args.deals, **D))
        stage("search_vs_pimc", lambda: duplicate(sp, pm, args.deals, **D))
        stage("adhoc_search+pimc_vs_pimc", lambda: duplicate((sp, pm), pm, args.deals, **D))
        stage("nn_vs_pimc_same_deals", lambda: duplicate(g, pm, args.deals, **D))
    else:
        res["nn_vs_rule"] = duplicate(g, rule_bot, 20000, B=args.nn_B, seed=args.seed, device=args.device)
        res["pimc_vs_rule"] = duplicate(pm, rule_bot, args.deals, **D)
        res["nn_vs_pimc"] = duplicate(g, pm, args.deals, **D)
        # ad-hoc: our bot with a PIMC partner vs PIMC pair (strong-human proxy)
        res["adhoc_nn+pimc_vs_pimc"] = duplicate((g, pm), pm, args.deals, **D)
    res["secs"] = time.time() - t
    print(json.dumps(res), flush=True)
    if args.out:
        with open(args.out, "a") as f:  # noqa
            f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
