# Checkpoints

Full PyTorch checkpoints (model weights + Adam optimizer state + training args), gzipped and split into
<50 MB parts because GitHub rejects files over 100 MB.

| File | What | Strength (tricks/game, duplicate) |
|---|---|---|
| `tf1_it578_eod.pt` | Transformer + belief head, iteration 578 — **end of day, newest** (7 more iterations after the eval) | vs v1 −0.029 pair / −0.013 one-seat (it 571); belief accuracy 46.3% |
| `tf1_it571_final.pt` | Transformer + **belief head**, iteration 571 — evaluated with search (see below) | vs v1 −0.022 pair / −0.017 one-seat (it 560); belief accuracy 46.2% (card counting 38.6%); vs PIMC −0.095 pair / +0.010 one-seat (it ~500) |
| `tf1_it481_main.pt` | Transformer (d256, 6 layers), main run, iteration 481 (before the belief head) | vs v1 −0.038 pair (it 481) |
| `tf1_it450_anchor.pt` | Transformer, iteration 450 — frozen "anchor" benchmark | |
| `tf1_it150_rewind.pt` | Transformer, iteration 150 — rewind point before the imitation experiment | vs v1 −0.216 pair |
| `v1_mlp_it1300.pt` | v1 residual-MLP policy, iteration 1300 — fixed reference (~PIMC level) | vs PIMC −0.04 ± 0.04 |

Reassemble + verify:

```bash
cd checkpoints
for n in tf1_it578_eod tf1_it571_final tf1_it481_main tf1_it450_anchor tf1_it150_rewind v1_mlp_it1300; do cat $n.pt.gz.part* | gunzip > $n.pt; done
sha256sum -c SHA256SUMS
```

Load: `from ladis.eval import load; model = load("checkpoints/tf1_it481_main.pt")` (CPU) or pass `"cuda"`.
Resume training: `python -m ladis.train --run runs/x --arch tf --resume checkpoints/tf1_it578_eod.pt --belief_coef 0.2 --belief_lr 1e-3 ...`

`logs/` has the training log (`log.jsonl`), the vs-v1 / anchor benchmark history (`bench.jsonl`),
PIMC evals (`pimc_eval.jsonl`), the critic comparison (`critic_test.json`) and the GPU speed tests.

## Final evaluation (tf1 it571 + search, 2026-11-03)

Duplicate deals, tricks/game for the first-named side (± 1 s.e.). Search = 32 sampled hidden-card layouts per
decision, each candidate card rolled out with the network in all four seats, gated (play the network's card unless
another is > 2 s.e. better). Logs: `logs/final_eval_belief.jsonl`, `logs/final_eval_uniform.jsonl`.

| Matchup | Deals | Result |
|---|---|---|
| belief-weighted search vs plain network | 300 | **+0.153 ± 0.031** |
| uniform-sampling search vs plain network (same deals) | 300 | +0.170 ± 0.032 |
| belief-weighted search vs PIMC (pair) | 150 | **+0.043 ± 0.050** |
| belief-weighted search + PIMC partner vs PIMC pair (one seat) | 150 | **+0.080 ± 0.050** |
| plain network vs PIMC (same deals) | 150 | −0.057 ± 0.048 |
