# job: tf1 retuned (from it-100 ckpt): 1 search worker, imitation weight 0.1, PIMC eval with small GPU batch
# kill strays from previous jobs (nested subshells survive the runner's kill of direct children)
pkill -f "ladis\." ; sleep 5; pkill -9 -f "ladis\." ; sleep 2
pgrep -f "tools/authserve.py" >/dev/null || { setsid nohup python3 tools/authserve.py runs > http.log 2>&1 < /dev/null & }
rm -f runs/IDLE
R=runs/tf1; mkdir -p $R/x
rm -f $R/x/shard_*.pt
RES=""; [ -f $R/ckpt.pt ] && RES="--resume $R/ckpt.pt"
echo "=== restart $(date -Is): retune: 1 search worker, exit_coef 0.1" >> $R/train.out
python3 -m ladis.train --run $R --arch tf --d 256 --layers 6 --heads 8 --B 8192 --mb 16384 --micro 4096 --lr 2e-4 \
  --eval_every 25 --snap_every 100 --latest_every 10 --exit_dir $R/x --exit_coef 0.1 --exit_start 85 \
  --hours 5.8 $RES >> $R/train.out 2>&1 &
TP=$!
sleep 60
python3 -m ladis.exit gen --ckpt $R/latest.pt --follow --search_mode critic --N 64 --beta 0 \
  --data $R/x --games 100000000 --shard_games 128 --seed 21 >> $R/gen_1.out 2>&1 &
( while kill -0 $TP 2>/dev/null; do
    sleep 2400
    [ -f $R/ckpt.pt ] || continue
    cp $R/ckpt.pt $R/ckpt_eval.pt
    python3 -m ladis.eval --ckpt $R/ckpt_eval.pt --device cuda --nn_B 500 --B 50 --deals 200 --pimc_k 16 --workers 10 --out $R/pimc_eval.jsonl >> $R/eval.out 2>&1
  done ) &
wait $TP
pkill -f "ladis.exit gen"; pkill -f "ladis.eval"
date -Is > runs/IDLE
