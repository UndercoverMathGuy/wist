"""GPU speed test for the transformer actor: reference vs fast vs fast+len_sort vs +compile.

python -m ladis.speed --out runs/speed.json
"""
from __future__ import annotations

import argparse
import json
import time

import torch
import torch.nn.functional as F

from .bots import rule_bot
from .engine import WhistBatch
from .model import ActorCritic, obs_index
from .train import micro_chunks


def collect(model, B, device):
    """Observations of the player to act at every one of the 52 steps of B games."""
    env = WhistBatch(B, device).reset()
    per_step = []
    while not env.finished:
        per_step.append((model.observe(env), env.legal_mask()))
        env.step(rule_bot(env))
    return per_step


def sync():
    torch.cuda.synchronize()


def train_pass(model, opt, obs, legal, N, mb, micro, len_sort, ref):
    perm = torch.randperm(N, device=legal.device)
    for i in range(0, N, mb):
        jb = perm[i:i + mb]
        opt.zero_grad(set_to_none=True)
        chunks = micro_chunks(jb, obs, micro, len_sort)
        for j in chunks:
            o = obs_index(obs, j)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = (model.actor.forward_reference(o) if ref else model.actor(o)).float()
            lg = lg.masked_fill(~legal[j], -1e9)
            tgt = legal[j].float().argmax(1)  # any legal card: a real CE loss with real gradients
            loss = F.cross_entropy(lg, tgt) * (len(j) / len(jb))
            loss.backward()
        opt.step()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--N", type=int, default=65536)
    ap.add_argument("--mb", type=int, default=16384)
    ap.add_argument("--micro", type=int, default=4096)
    ap.add_argument("--B", type=int, default=8192)
    a = ap.parse_args()
    dev = torch.device("cuda")
    torch.manual_seed(0)
    model = ActorCritic(arch="tf", d=256, layers=6, heads=8).to(dev)
    per_step = collect(model, a.B, dev)
    obs = {k: torch.cat([o[k] for o, _ in per_step]) for k in per_step[0][0]}
    legal = torch.cat([l for _, l in per_step])
    sel = torch.randperm(len(legal), device=dev)[:a.N]
    obs, legal = obs_index(obs, sel), legal[sel]
    res = {"gpu": torch.cuda.get_device_name(), "N": a.N, "mb": a.mb, "micro": a.micro, "train": {}, "rollout": {}}

    configs = [("reference", dict(ref=True, len_sort=False)),
               ("fast", dict(ref=False, len_sort=False)),
               ("fast+len_sort", dict(ref=False, len_sort=True)),
               ("fast+len_sort+compile", dict(ref=False, len_sort=True, compile=True))]
    for name, cfg in configs:
        try:
            if cfg.get("compile"):
                t0 = time.time()
                model.actor.compile_core()
                train_pass(model, torch.optim.Adam(model.parameters(), lr=0, fused=True), obs, legal,
                           a.N, a.mb, a.micro, True, False)  # warm-up / compile all shapes
                sync()
                res["compile_secs"] = round(time.time() - t0, 1)
            opt = torch.optim.Adam(model.parameters(), lr=0, fused=True)
            train_pass(model, opt, obs, legal, min(a.N, 2 * a.mb), a.mb, a.micro, cfg["len_sort"], cfg["ref"])
            sync()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            train_pass(model, opt, obs, legal, a.N, a.mb, a.micro, cfg["len_sort"], cfg["ref"])
            sync()
            dt = time.time() - t0
            res["train"][name] = {"samples_per_s": round(a.N / dt), "secs": round(dt, 2),
                                  "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1)}
            # rollout-style inference: every step of a game at batch B
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for o, _ in per_step[::13]:
                    (model.actor.forward_reference(o) if cfg["ref"] else model.actor(o))
                sync()
                t0 = time.time()
                for o, _ in per_step:
                    (model.actor.forward_reference(o) if cfg["ref"] else model.actor(o))
                sync()
            res["rollout"][name] = {"secs_per_52_steps": round(time.time() - t0, 3)}
        except Exception as e:  # keep going; report what failed
            res["train"][name] = {"error": repr(e)[:300]}
        print(json.dumps({name: [res["train"].get(name), res["rollout"].get(name)]}), flush=True)
    model.actor._core_compiled = None
    base = res["train"].get("reference", {}).get("samples_per_s")
    for k, v in res["train"].items():
        if base and "samples_per_s" in v:
            v["speedup"] = round(v["samples_per_s"] / base, 2)
    print(json.dumps(res), flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
