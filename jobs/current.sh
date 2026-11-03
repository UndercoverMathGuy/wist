# job: continue MAIN training from the belief arm (B) of the A/B, belief loss on, until the RunPod balance runs out.
pkill -f "ladis\." ; for p in $(pgrep -f "bash jobs/current.sh"); do [ "$p" != "$$" ] && kill "$p" 2>/dev/null; done
for i in $(seq 1 30); do pgrep -f "ladis\.train" >/dev/null || break; sleep 5; done
pkill -9 -f "ladis\." ; sleep 2
pgrep -f "tools/authserve.py" >/dev/null || { setsid nohup python3 tools/authserve.py runs > http.log 2>&1 < /dev/null & }
rm -f runs/IDLE
R=runs/tf1
# Decision (user): stop the A/B now (cut early) and adopt the belief arm, since its loss is falling
# (accuracy 0.42 vs 0.36 baseline). Belief is a known blind spot + needed for belief-weighted search.
WIN=B; BCOEF=0.2
[ -f $R/ckpt_ab_winner.pt ] || cp runs/ab$WIN/latest.pt $R/ckpt_ab_winner.pt
RES="--resume $R/ckpt_ab_winner.pt"
[ -f $R/latest.pt ] && [ $R/latest.pt -nt $R/ckpt_ab_winner.pt ] && RES="--resume $R/latest.pt"
REFS="runs/v1/ckpt_paused.pt $(ls $R/ckpt_anchor_it*.pt | head -1)"
echo "=== restart $(date -Is): A/B cut early; continue from belief arm $WIN (belief_coef $BCOEF) $RES, until balance runs out" >> $R/train.out
python3 -m ladis.train --run $R --arch tf --d 256 --layers 6 --heads 8 --B 16384 --mb 16384 --micro 4096 \
  --lr 3e-5 --lr_end 1e-5 --hours 4 --eval_every 25 --snap_every 100 --latest_every 10 --target_kl 0.03 \
  --len_sort 1 --compile 1 --belief_coef $BCOEF --belief_lr 1e-3 $RES >> $R/train.out 2>&1 &
TP=$!
( while kill -0 $TP 2>/dev/null; do
    if python3 -m ladis.bench --ckpt $R/latest.pt --snapshot --deals 4096 --B 2048 --refs $REFS --out $R/bench.jsonl >> $R/bench.out 2>&1; then
      sleep 450
    else
      sleep 60
    fi
  done ) &
( sleep 600
  while kill -0 $TP 2>/dev/null; do
    [ -f $R/ckpt.pt ] && cp $R/ckpt.pt $R/ckpt_eval.pt && \
      python3 -m ladis.eval --ckpt $R/ckpt_eval.pt --device cuda --nn_B 500 --B 50 --deals 200 --pimc_k 16 --workers 12 --out $R/pimc_eval.jsonl >> $R/eval.out 2>&1
    sleep 2400
  done ) &
wait $TP
pkill -f "ladis\."
date -Is > runs/IDLE
