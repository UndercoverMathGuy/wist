# job: PAUSE training -> GPU speed test (reference / fast / +len_sort / +compile) -> resume tf1 with the fastest
#      configuration that ran cleanly (+ bench loop vs v1 every ~7.5 min, PIMC eval every ~40 min)
pkill -f "ladis\." ; for p in $(pgrep -f "bash jobs/current.sh"); do [ "$p" != "$$" ] && kill "$p" 2>/dev/null; done
for i in $(seq 1 30); do pgrep -f "ladis\.train" >/dev/null || break; sleep 5; done   # let a trainer save & exit
pkill -9 -f "ladis\." ; sleep 2
pgrep -f "tools/authserve.py" >/dev/null || { setsid nohup python3 tools/authserve.py runs > http.log 2>&1 < /dev/null & }
rm -f runs/IDLE
R=runs/tf1
echo "=== paused $(date -Is) for speed test" >> $R/train.out
date -Is > runs/PAUSED_FOR_SPEEDTEST
python3 -m ladis.speed --out runs/speed.json > runs/speed.out 2>&1
rm -f runs/PAUSED_FOR_SPEEDTEST
# choose flags from the speed test
FLAGS=$(python3 - <<'PY'
import json
try:
    t = json.load(open("runs/speed.json"))["train"]
except Exception:
    print("--len_sort 0 --compile 0"); raise SystemExit
sp = lambda k: t.get(k, {}).get("samples_per_s", 0)
best_ls = sp("fast+len_sort") >= sp("fast")
use_compile = sp("fast+len_sort+compile") > 1.05 * max(sp("fast+len_sort"), sp("fast"))
print(f"--len_sort {int(best_ls or use_compile)} --compile {int(use_compile)}")
PY
)
END=$(date -u -d "today 12:45" +%s); NOW=$(date -u +%s); HOURS=$(python3 -c "print(max(0.2, ($END-$NOW)/3600))")
REFS="runs/v1/ckpt_paused.pt"
RES="--resume $R/ckpt.pt"; [ $R/latest.pt -nt $R/ckpt.pt ] && RES="--resume $R/latest.pt"
echo "=== restart $(date -Is): resume after speed test ($RES) flags: $FLAGS hours $HOURS" >> $R/train.out
python3 -m ladis.train --run $R --arch tf --d 256 --layers 6 --heads 8 --B 8192 --mb 16384 --micro 4096 --lr 2e-4 \
  --eval_every 25 --snap_every 100 --latest_every 10 --target_kl 0.03 $FLAGS \
  --hours $HOURS $RES >> $R/train.out 2>&1 &
TP=$!
( while kill -0 $TP 2>/dev/null; do
    python3 -m ladis.bench --ckpt $R/latest.pt --snapshot --deals 4096 --refs $REFS --out $R/bench.jsonl >> $R/bench.out 2>&1
    sleep 450
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
