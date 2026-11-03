"""Fast, unsaturated progress benchmark against fixed reference bots.

For each reference checkpoint R:
  pair  : our pair vs R pair                        (tricks/game, duplicate)
  adhoc : our bot + R partner vs R pair            (one seat swapped in)
Same deals every call (fixed seed) so checkpoint-to-checkpoint differences are not luck.

python -m ladis.bench --ckpt runs/tf1/latest.pt --refs runs/v1/ckpt_paused.pt runs/tf1/ckpt_it150_rewind.pt
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time

import torch

from .arena import duplicate
from .bots import rule_bot
from .eval import load
from .train import nn_policy


def bench(ckpt, refs, deals, B, device, seed=2024):
    me = nn_policy(load(ckpt, device))
    it = torch.load(ckpt, map_location="cpu").get("iter")
    res = {"ckpt": ckpt, "iter": it, "deals": deals, "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    for r in refs:
        name = "rule" if r == "rule" else os.path.basename(os.path.dirname(r)) + "/" + os.path.basename(r)
        ref = rule_bot if r == "rule" else nn_policy(load(r, device))
        pair = duplicate(me, ref, deals, B=B, device=device, seed=seed)
        adhoc = duplicate((me, ref), ref, deals, B=B, device=device, seed=seed)
        res[name] = {"pair": [round(pair[0], 4), round(pair[1], 4)], "adhoc": [round(adhoc[0], 4), round(adhoc[1], 4)]}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--refs", nargs="+", required=True)
    ap.add_argument("--deals", type=int, default=16384)
    ap.add_argument("--B", type=int, default=2048,
                    help="keep fixed: deals are drawn per batch, so B defines the deal set")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="")
    ap.add_argument("--snapshot", action="store_true", help="copy ckpt first (it may be rewritten mid-bench)")
    a = ap.parse_args()
    ck = a.ckpt
    if a.snapshot:
        ck = a.ckpt + ".bench"
        shutil.copyfile(a.ckpt, ck)
    r = bench(ck, a.refs, a.deals, a.B, a.device)
    r["ckpt"] = a.ckpt
    print(json.dumps(r), flush=True)
    if a.out:
        with open(a.out, "a") as f:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
