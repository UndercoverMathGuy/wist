# job: tf1 pure PPO, fast path (len_sort + compile), LR 5e-5 -> 2e-5 linear decay, rollout batch 16384;
#      bench vs v1 every ~7.5 min; PIMC eval 10 min after start then every ~40 min
pkill -f "ladis\." ; for p in $(pgrep -f "bash jobs/current.sh"); do [ "$p" != "$$" ] && kill "$p" 2>/dev/null; done
for i in $(seq 1 30); do pgrep -f "ladis\.train" >/dev/null || break; sleep 5; done   # let the trainer save & exit
pkill -9 -f "ladis\." ; sleep 2
pgrep -f "tools/authserve.py" >/dev/null || { setsid nohup python3 tools/authserve.py runs > http.log 2>&1 < /dev/null & }
rm -f runs/IDLE
R=runs/tf1
END=$(date -u -d "today 12:45" +%s); NOW=$(date -u +%s); HOURS=$(python3 -c "print(max(0.2, ($END-$NOW)/3600))")
# continue the LR schedule (5e-5 at 10:39 UTC -> 2e-5 at 12:45 UTC) from wherever it is now
LR0=$(python3 -c "
import time,calendar
t0=calendar.timegm(time.strptime(time.strftime('%Y-%m-%d',time.gmtime())+' 10:39','%Y-%m-%d %H:%M'))
t1=calendar.timegm(time.strptime(time.strftime('%Y-%m-%d',time.gmtime())+' 12:45','%Y-%m-%d %H:%M'))
f=min(1,max(0,(time.time()-t0)/(t1-t0))); print('%.3e'%(5e-5+(2e-5-5e-5)*f))")
REFS="runs/v1/ckpt_paused.pt"
RES="--resume $R/ckpt.pt"; [ $R/latest.pt -nt $R/ckpt.pt ] && RES="--resume $R/latest.pt"
echo "=== restart $(date -Is): bench retry fix; lr $LR0->2e-5 (schedule continued), B 16384 ($RES) hours $HOURS" >> $R/train.out
python3 -m ladis.train --run $R --arch tf --d 256 --layers 6 --heads 8 --B 16384 --mb 16384 --micro 4096 --lr $LR0 --lr_end 2e-5 \
  --eval_every 25 --snap_every 100 --latest_every 10 --target_kl 0.03 --len_sort 1 --compile 1 \
  --hours $HOURS $RES >> $R/train.out 2>&1 &
TP=$!
( while kill -0 $TP 2>/dev/null; do
    # smaller GPU batch so it fits next to training + PIMC eval; on failure retry after 60 s, not 7.5 min
    if python3 -m ladis.bench --ckpt $R/latest.pt --snapshot --deals 4096 --B 1024 --refs $REFS --out $R/bench.jsonl >> $R/bench.out 2>&1; then
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
