# job: after the final eval, resume belief training from the final checkpoint until the balance runs out.
pkill -f "ladis\." ; for p in $(pgrep -f "bash jobs/current.sh"); do [ "$p" != "$$" ] && kill "$p" 2>/dev/null; done
for i in $(seq 1 30); do pgrep -f "ladis\.train" >/dev/null || break; sleep 5; done
pkill -9 -f "ladis\." ; sleep 2
pgrep -f "tools/authserve.py" >/dev/null || { setsid nohup python3 tools/authserve.py runs > http.log 2>&1 < /dev/null & }
R=runs/tf1
REFS="runs/v1/ckpt_paused.pt $(ls $R/ckpt_anchor_it*.pt | head -1)"
RES="--resume $R/ckpt_final.pt"; [ $R/latest.pt -nt $R/ckpt_final.pt ] && RES="--resume $R/latest.pt"
echo "=== restart $(date -Is): post-eval, resume belief training ($RES) until balance runs out" >> $R/train.out
python3 -m ladis.train --run $R --arch tf --d 256 --layers 6 --heads 8 --B 16384 --mb 16384 --micro 4096 \
  --lr 2.2e-5 --lr_end 1e-5 --hours 3 --eval_every 25 --snap_every 100 --latest_every 10 --target_kl 0.03 \
  --len_sort 1 --compile 1 --belief_coef 0.2 --belief_lr 1e-3 $RES >> $R/train.out 2>&1 &
TP=$!
( while kill -0 $TP 2>/dev/null; do
    if python3 -m ladis.bench --ckpt $R/latest.pt --snapshot --deals 4096 --B 2048 --refs $REFS --out $R/bench.jsonl >> $R/bench.out 2>&1; then
      sleep 450
    else
      sleep 60
    fi
  done ) &
wait $TP
date -Is > runs/IDLE
