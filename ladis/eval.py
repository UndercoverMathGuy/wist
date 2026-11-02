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
from .model import ActorCritic
from .pimc import PIMC
from .train import nn_policy


def load(path, device="cpu"):
    ck = torch.load(path, map_location=device)
    a = ck.get("args", {})
    m = ActorCritic(a.get("width", 512), a.get("depth", 4), hist=a.get("hist", False)).to(device)
    m.load_state_dict(ck["model"])
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
    args = ap.parse_args()
    torch.set_num_threads(2)
    m = load(args.ckpt)
    g = nn_policy(m)
    pm = PIMC(K=args.pimc_k, workers=args.workers)
    res = {"ckpt": args.ckpt, "deals": args.deals, "pimc_k": args.pimc_k}
    t = time.time()
    res["nn_vs_rule"] = duplicate(g, rule_bot, 20000, B=4000, seed=args.seed)
    res["pimc_vs_rule"] = duplicate(pm, rule_bot, args.deals, B=args.B, seed=args.seed)
    res["nn_vs_pimc"] = duplicate(g, pm, args.deals, B=args.B, seed=args.seed)
    # ad-hoc: our bot with a PIMC partner vs PIMC pair (strong-human proxy)
    res["adhoc_nn+pimc_vs_pimc"] = duplicate((g, pm), pm, args.deals, B=args.B, seed=args.seed)
    res["secs"] = time.time() - t
    print(json.dumps(res), flush=True)
    if args.out:
        with open(args.out, "a") as f:
            f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
