# Checkpoints

Full PyTorch checkpoints (model weights + Adam optimizer state + training args), gzipped and split into
<50 MB parts because GitHub rejects files over 100 MB.

| File | What | Strength (tricks/game, duplicate) |
|---|---|---|
| `tf1_it571_final.pt` | Transformer + **belief head**, iteration 571 — **final / best** | vs v1 −0.022 pair / −0.017 one-seat (it 560); belief accuracy 46.2% (card counting 38.6%); vs PIMC −0.095 pair / +0.010 one-seat (it ~500) |
| `tf1_it481_main.pt` | Transformer (d256, 6 layers), main run, iteration 481 (before the belief head) | vs v1 −0.038 pair (it 481) |
| `tf1_it450_anchor.pt` | Transformer, iteration 450 — frozen "anchor" benchmark | |
| `tf1_it150_rewind.pt` | Transformer, iteration 150 — rewind point before the imitation experiment | vs v1 −0.216 pair |
| `v1_mlp_it1300.pt` | v1 residual-MLP policy, iteration 1300 — fixed reference (~PIMC level) | vs PIMC −0.04 ± 0.04 |

Reassemble + verify:

```bash
cd checkpoints
for n in tf1_it571_final tf1_it481_main tf1_it450_anchor tf1_it150_rewind v1_mlp_it1300; do cat $n.pt.gz.part* | gunzip > $n.pt; done
sha256sum -c SHA256SUMS
```

Load: `from ladis.eval import load; model = load("checkpoints/tf1_it481_main.pt")` (CPU) or pass `"cuda"`.
Resume training: `python -m ladis.train --run runs/x --arch tf --resume checkpoints/tf1_it571_final.pt --belief_coef 0.2 --belief_lr 1e-3 ...`

`logs/` has the training log (`log.jsonl`), the vs-v1 / anchor benchmark history (`bench.jsonl`),
PIMC evals (`pimc_eval.jsonl`), the critic comparison (`critic_test.json`) and the GPU speed tests.
