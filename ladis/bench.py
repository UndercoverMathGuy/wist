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


def _tf1_hooks(a):
    """One-off jobs piggy-backing on the running bench loop (it re-imports this file every cycle,
    so these take effect without restarting training):
      * critic_test: does v1's critic predict tf1's outcomes better than tf1's own critic?
      * anchor: freeze the current tf1 checkpoint once and benchmark against it from now on
        (an independent primary benchmark next to v1)."""
    import glob
    run_dir = os.path.dirname(a.out)
    if not os.path.exists(os.path.join(run_dir, "critic_test.json")):
        try:
            from .critic_test import run as critic_run
            src = a.ckpt + ".critictest"
            shutil.copyfile(a.ckpt, src)
            res = critic_run(src, "runs/v1/ckpt_paused.pt", 2048, 4, torch.device(a.device))
            res.update(tf=a.ckpt, iter=torch.load(src, map_location="cpu").get("iter"))
            with open(os.path.join(run_dir, "critic_test.json"), "w") as f:
                json.dump(res, f, indent=1)
            print(json.dumps({"critic_test": res}), flush=True)
        except Exception as e:  # never break the bench loop
            print(json.dumps({"critic_test_error": repr(e)[:300]}), flush=True)
    anchors = sorted(glob.glob(os.path.join(run_dir, "ckpt_anchor_it*.pt")))
    if not anchors:
        it = torch.load(a.ckpt, map_location="cpu").get("iter")
        dst = os.path.join(run_dir, f"ckpt_anchor_it{it}.pt")
        shutil.copyfile(a.ckpt, dst)
        anchors = [dst]
    for anc in anchors:
        if anc not in a.refs:
            a.refs.append(anc)


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
    if a.out.endswith("tf1/bench.jsonl"):
        _tf1_hooks(a)
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
