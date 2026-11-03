# job: tf1 pure PPO (after the it-150 rewind) + light benchmark (4096 deals / 15 min) + live dashboard at /dashboard
pkill -f "ladis\." ; for p in $(pgrep -f "bash jobs/current.sh"); do [ "$p" != "$$" ] && kill "$p" 2>/dev/null; done
sleep 5; pkill -9 -f "ladis\." ; sleep 2
# restart the file server so it picks up the /dashboard route (page itself is read live from tools/dashboard.html)
pkill -f "tools/authserve.py"; sleep 1
setsid nohup python3 tools/authserve.py runs > http.log 2>&1 < /dev/null &
rm -f runs/IDLE
R=runs/tf1
REFS="runs/v1/ckpt_paused.pt $R/ckpt_it150_rewind.pt"
# the existing iteration-200 snapshot belongs to the abandoned imitation branch: move it aside (once)
[ -f $R/snap_imitation_200.pt ] || mv $R/snaps/snap_0000200.pt $R/snap_imitation_200.pt
RES="--resume $R/ckpt.pt"; [ $R/latest.pt -nt $R/ckpt.pt ] && RES="--resume $R/latest.pt"
echo "=== restart $(date -Is): pure PPO continue ($RES), + light bench loop" >> $R/train.out
python3 -m ladis.train --run $R --arch tf --d 256 --layers 6 --heads 8 --B 8192 --mb 16384 --micro 4096 --lr 2e-4 \
  --eval_every 25 --snap_every 100 --latest_every 10 --target_kl 0.03 \
  --hours 3.3 $RES >> $R/train.out 2>&1 &
TP=$!
( while kill -0 $TP 2>/dev/null; do
    # light periodic bench of the newest checkpoint
    python3 -m ladis.bench --ckpt $R/latest.pt --snapshot --deals 4096 --refs $REFS --out $R/bench.jsonl >> $R/bench.out 2>&1
    sleep 900
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
