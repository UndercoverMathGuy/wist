# job: tf1 REWIND to iteration-150 checkpoint (end of the plain-PPO stretch), continue as PURE PPO (no search, no imitation)
# kill strays: python processes AND orphaned loop subshells from earlier jobs
pkill -f "ladis\." ; for p in $(pgrep -f "bash jobs/current.sh"); do [ "$p" != "$$" ] && kill "$p" 2>/dev/null; done
sleep 5; pkill -9 -f "ladis\." ; sleep 2
pgrep -f "tools/authserve.py" >/dev/null || { setsid nohup python3 tools/authserve.py runs > http.log 2>&1 < /dev/null & }
rm -f runs/IDLE
R=runs/tf1
# preserve the rewind point before the eval loop can overwrite ckpt_eval.pt
[ -f $R/ckpt_it150_rewind.pt ] || cp $R/ckpt_eval.pt $R/ckpt_it150_rewind.pt
# keep the abandoned imitation branch for reference
[ -f $R/ckpt_imitation_branch.pt ] || cp $R/latest.pt $R/ckpt_imitation_branch.pt
rm -f $R/x/shard_*.pt
echo "=== restart $(date -Is): REWIND to ckpt_it150_rewind.pt, pure PPO (no search/imitation), target_kl 0.03" >> $R/train.out
python3 -m ladis.train --run $R --arch tf --d 256 --layers 6 --heads 8 --B 8192 --mb 16384 --micro 4096 --lr 2e-4 \
  --eval_every 25 --snap_every 100 --latest_every 10 --target_kl 0.03 \
  --hours 4.3 --resume $R/ckpt_it150_rewind.pt >> $R/train.out 2>&1 &
TP=$!
( while kill -0 $TP 2>/dev/null; do
    sleep 2400
    [ -f $R/ckpt.pt ] || continue
    cp $R/ckpt.pt $R/ckpt_eval.pt
    python3 -m ladis.eval --ckpt $R/ckpt_eval.pt --device cuda --nn_B 500 --B 50 --deals 200 --pimc_k 16 --workers 12 --out $R/pimc_eval.jsonl >> $R/eval.out 2>&1
  done ) &
wait $TP
pkill -f "ladis.eval"
date -Is > runs/IDLE
