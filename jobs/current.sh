# job: tf1 pure PPO (continuing after the it-150 rewind) + light benchmark (4096 deals / 15 min); exact it-200 bench for the imitation-vs-PPO comparison
pkill -f "ladis\." ; for p in $(pgrep -f "bash jobs/current.sh"); do [ "$p" != "$$" ] && kill "$p" 2>/dev/null; done
sleep 5; pkill -9 -f "ladis\." ; sleep 2
pgrep -f "tools/authserve.py" >/dev/null || { setsid nohup python3 tools/authserve.py runs > http.log 2>&1 < /dev/null & }
rm -f runs/IDLE
R=runs/tf1
REFS="runs/v1/ckpt_paused.pt $R/ckpt_it150_rewind.pt"
# the existing iteration-200 snapshot belongs to the abandoned imitation branch: move it aside (once)
[ -f $R/snap_imitation_200.pt ] || mv $R/snaps/snap_0000200.pt $R/snap_imitation_200.pt
RES="--resume $R/ckpt.pt"; [ $R/latest.pt -nt $R/ckpt.pt ] && RES="--resume $R/latest.pt"
echo "=== restart $(date -Is): pure PPO continue ($RES), + light bench loop" >> $R/train.out
python3 -m ladis.train --run $R --arch tf --d 256 --layers 6 --heads 8 --B 8192 --mb 16384 --micro 4096 --lr 2e-4 \
  --eval_every 25 --snap_every 100 --latest_every 10 --target_kl 0.03 \
  --hours 3.85 $RES >> $R/train.out 2>&1 &
TP=$!
( while kill -0 $TP 2>/dev/null; do
    # light periodic bench of the newest checkpoint
    python3 -m ladis.bench --ckpt $R/latest.pt --snapshot --deals 4096 --refs $REFS --out $R/bench.jsonl >> $R/bench.out 2>&1
    sleep 900
  done ) &
# exact iteration-200 comparison (pure PPO vs the imitation branch's +0.024 over the rewind point):
# the trainer writes a permanent league snapshot at iteration 200; wrap it as a checkpoint and bench it once
( until [ -f $R/snaps/snap_0000200.pt ] || ! kill -0 $TP 2>/dev/null; do sleep 60; done
  sleep 20
  python3 - <<'PY' && python3 -m ladis.bench --ckpt runs/tf1/ckpt_it200_ppo.pt --deals 16384 --refs $REFS --out $R/bench.jsonl >> $R/bench.out 2>&1
import torch
args = torch.load("runs/tf1/ckpt_it150_rewind.pt", map_location="cpu")["args"]
sd = torch.load("runs/tf1/snaps/snap_0000200.pt", map_location="cpu")
torch.save({"model": sd, "iter": 200, "args": args}, "runs/tf1/ckpt_it200_ppo.pt")
PY
) &
( while kill -0 $TP 2>/dev/null; do
    sleep 2400
    [ -f $R/ckpt.pt ] || continue
    cp $R/ckpt.pt $R/ckpt_eval.pt
    python3 -m ladis.eval --ckpt $R/ckpt_eval.pt --device cuda --nn_B 500 --B 50 --deals 200 --pimc_k 16 --workers 12 --out $R/pimc_eval.jsonl >> $R/eval.out 2>&1
  done ) &
wait $TP
pkill -f "ladis\."
date -Is > runs/IDLE
