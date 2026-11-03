# job: tf1 resumed. Imitation (search) loss weight 0.25, off until iteration 85; search workers start near then.
pgrep -f "tools/authserve.py" >/dev/null || { setsid nohup python3 tools/authserve.py runs > http.log 2>&1 < /dev/null & }
rm -f runs/IDLE
R=runs/tf1; mkdir -p $R/x
rm -f $R/x/shard_*.pt
RES=""; [ -f $R/ckpt.pt ] && RES="--resume $R/ckpt.pt"
echo "=== restart $(date -Is): grad accumulation mb 16384 (micro 4096), exit_coef 0.25, exit_start 85" >> $R/train.out
python3 -m ladis.train --run $R --arch tf --d 256 --layers 6 --heads 8 --B 8192 --mb 16384 --micro 4096 --lr 2e-4 \
  --eval_every 25 --snap_every 100 --latest_every 10 --exit_dir $R/x --exit_coef 0.25 --exit_start 85 \
  --hours 8 $RES >> $R/train.out 2>&1 &
TP=$!
( until grep -q '"it": 7[0-9],' $R/train.out || ! kill -0 $TP 2>/dev/null; do sleep 30; done
  for s in 1 2 3; do
    python3 -m ladis.exit gen --ckpt $R/latest.pt --follow --search_mode critic --N 64 --beta 0 \
      --data $R/x --games 100000000 --shard_games 128 --seed $((s + 10)) >> $R/gen_$s.out 2>&1 &
  done; wait ) &
( while kill -0 $TP 2>/dev/null; do
    sleep 2700
    [ -f $R/ckpt.pt ] || continue
    cp $R/ckpt.pt $R/ckpt_eval.pt
    python3 -m ladis.eval --ckpt $R/ckpt_eval.pt --device cuda --deals 200 --pimc_k 16 --workers 6 --out $R/pimc_eval.jsonl >> $R/eval.out 2>&1
  done ) &
wait $TP
pkill -f "ladis.exit gen"; pkill -f "ladis.eval"
date -Is > runs/IDLE
