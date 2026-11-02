# job: expert iteration round 1 on v1 (gen -> distill -> eval)
pgrep -f "tools/authserve.py" >/dev/null || { setsid nohup python3 tools/authserve.py runs > http.log 2>&1 < /dev/null & }
rm -f runs/IDLE
mkdir -p runs/exit1
CK=runs/v1/ckpt_paused.pt
pids=""
for s in 1 2 3 4; do
  python3 -m ladis.exit gen --ckpt $CK --data runs/exit1/data --games 2500 --shard_games 128 --N 128 --beta 0 --seed $s > runs/exit1/gen_$s.out 2>&1 &
  pids="$pids $!"
done
wait $pids
date -Is > runs/exit1/GEN_DONE
python3 -m ladis.exit distill --ckpt $CK --data runs/exit1/data --out runs/exit1/v1x.pt --epochs 2 > runs/exit1/distill.out 2>&1
python3 -m ladis.eval --ckpt runs/exit1/v1x.pt --device cuda --deals 300 --pimc_k 16 --workers 14 --out runs/exit1/eval.jsonl > runs/exit1/eval.out 2>&1
python3 - > runs/exit1/h2h.out 2>&1 <<'PY'
import json
from ladis.arena import duplicate
from ladis.eval import load
from ladis.train import nn_policy
a = nn_policy(load("runs/exit1/v1x.pt", "cuda")); b = nn_policy(load("runs/v1/ckpt_paused.pt", "cuda"))
print(json.dumps({"v1x_vs_v1": duplicate(a, b, 50000, B=8192, device="cuda", seed=11)}))
PY
date -Is > runs/exit1/DONE
