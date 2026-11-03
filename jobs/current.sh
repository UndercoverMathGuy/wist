# job: transformer from scratch, PPO self-play league + search in the loop (async expert iteration)
pgrep -f "tools/authserve.py" >/dev/null || { setsid nohup python3 tools/authserve.py runs > http.log 2>&1 < /dev/null & }
rm -f runs/IDLE
R=runs/tf1; mkdir -p $R/x
RES=""; [ -f $R/ckpt.pt ] && RES="--resume $R/ckpt.pt"
python3 -m ladis.train --run $R --arch tf --d 256 --layers 6 --heads 8 --B 8192 --mb 4096 --lr 2e-4 \
  --eval_every 25 --snap_every 100 --latest_every 10 --exit_dir $R/x --exit_coef 1.0 --hours 9 $RES >> $R/train.out 2>&1 &
TP=$!
sleep 90
for s in 1 2 3; do
  python3 -m ladis.exit gen --ckpt $R/latest.pt --follow --search_mode critic --N 64 --beta 0 \
    --data $R/x --games 100000000 --shard_games 128 --seed $s >> $R/gen_$s.out 2>&1 &
done
( while kill -0 $TP 2>/dev/null; do
    sleep 2700
    [ -f $R/ckpt.pt ] || continue
    cp $R/ckpt.pt $R/ckpt_eval.pt
    python3 -m ladis.eval --ckpt $R/ckpt_eval.pt --device cuda --deals 200 --pimc_k 16 --workers 6 --out $R/pimc_eval.jsonl >> $R/eval.out 2>&1
  done ) &
wait $TP
pkill -f "ladis.exit gen"; pkill -f "ladis.eval"
date -Is > runs/IDLE
