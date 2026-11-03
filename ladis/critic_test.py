"""Which critic predicts tf1's actual outcomes better: tf1's own, or v1's?

Plays fresh league rollouts with the *current* tf1 policy, computes Monte-Carlo returns (final
trick margin from every learner decision onward, gamma = 1), and scores both critics on exactly
the same states. Both critics take identical inputs (flat features + other three hands).

python -m ladis.critic_test --tf runs/tf1/latest.pt --v1 runs/v1/ckpt_paused.pt --out runs/tf1/critic_test.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import time

import torch
import torch.nn.functional as F

from .eval import load
from .model import obs_index
from .train import League, rollout


@torch.no_grad()
def run(tf_path, v1_path, B, rollouts, device):
    tf = load(tf_path, device)
    v1 = load(v1_path, device)
    league = League()
    league.add(tf)
    preds_tf, preds_v1, rets = [], [], []
    for _ in range(rollouts):
        buf, _ = rollout(tf, league, B, device, [0.5, 0.2, 0.3], None)
        rew, m = buf["rew"], buf["mask"]
        ret = torch.flip(torch.cumsum(torch.flip(rew, [0]), 0), [0])  # MC return from trick k onward
        obs = {k: v[m] for k, v in buf["obs"].items()}
        n = len(ret[m])
        for i in range(0, n, 65536):
            o = obs_index(obs, slice(i, i + 65536))
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                preds_tf.append(tf.value(o).float())
                preds_v1.append(v1.value(o).float())
        rets.append(ret[m])
        del buf
    y = torch.cat(rets)
    out = {"samples": len(y), "return_var": round(y.var().item(), 4)}
    for name, p in (("tf1_critic", torch.cat(preds_tf)), ("v1_critic", torch.cat(preds_v1))):
        out[name] = {"mse": round(F.mse_loss(p, y).item(), 4),
                     "smooth_l1": round(F.smooth_l1_loss(p, y).item(), 4),
                     "explained_var": round(1 - (y - p).var().item() / y.var().item(), 4),
                     "bias": round((p - y).mean().item(), 4)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", required=True)
    ap.add_argument("--v1", required=True)
    ap.add_argument("--B", type=int, default=2048)
    ap.add_argument("--rollouts", type=int, default=4)
    ap.add_argument("--out", default="")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    src = a.tf
    if a.out:  # snapshot: the trainer rewrites latest.pt
        src = a.tf + ".critictest"
        shutil.copyfile(a.tf, src)
    t = time.time()
    res = run(src, a.v1, a.B, a.rollouts, torch.device(a.device))
    res.update(tf=a.tf, iter=torch.load(src, map_location="cpu").get("iter"), secs=round(time.time() - t, 1))
    print(json.dumps(res), flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
