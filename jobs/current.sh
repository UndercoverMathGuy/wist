# job: tf1 pure PPO (continuing after the it-150 rewind) + fast benchmark vs fixed refs every ~10 min
pkill -f "ladis\." ; for p in $(pgrep -f "bash jobs/current.sh"); do [ "$p" != "$$" ] && kill "$p" 2>/dev/null; done
sleep 5; pkill -9 -f "ladis\." ; sleep 2
pgrep -f "tools/authserve.py" >/dev/null || { setsid nohup python3 tools/authserve.py runs > http.log 2>&1 < /dev/null & }
rm -f runs/IDLE
R=runs/tf1
REFS="runs/v1/ckpt_paused.pt $R/ckpt_it150_rewind.pt"
RES="--resume $R/ckpt.pt"; [ $R/latest.pt -nt $R/ckpt.pt ] && RES="--resume $R/latest.pt"
echo "=== restart $(date -Is): pure PPO continue ($RES), + bench loop" >> $R/train.out
python3 -m ladis.train --run $R --arch tf --d 256 --layers 6 --heads 8 --B 8192 --mb 16384 --micro 4096 --lr 2e-4 \
  --eval_every 25 --snap_every 100 --latest_every 10 --target_kl 0.03 \
  --hours 4.1 $RES >> $R/train.out 2>&1 &
TP=$!
( for c in $R/ckpt_it150_rewind.pt $R/ckpt_imitation_branch.pt; do
    python3 -m ladis.bench --ckpt $c --refs $REFS --out $R/bench.jsonl >> $R/bench.out 2>&1
  done
  while kill -0 $TP 2>/dev/null; do
    python3 -m ladis.bench --ckpt $R/latest.pt --snapshot --refs $REFS --out $R/bench.jsonl >> $R/bench.out 2>&1
    sleep 600
  done ) &
( while kill -0 $TP 2>/dev/null; do
    sleep 2400
    [ -f $R/ckpt.pt ] || continue
    cp $R/ckpt.pt $R/ckpt_eval.pt
    python3 -m ladis.eval --ckpt $R/ckpt_eval.pt --device cuda --nn_B 500 --B 50 --deals 200 --pimc_k 16 --workers 12 --out $R/pimc_eval.jsonl >> $R/eval.out 2>&1
  done ) &
wait $TP
pkill -f "ladis\."
date -Is > runs/IDLE
