# job: BELIEF-HEAD A/B v2 (belief head on its own lr 1e-3; v1 of the A/B was cut: head learned too slowly at 3e-5)
#   stop main tf1 (graceful save) -> two arms from the same checkpoint, concurrently, 40 iterations each:
#   A = PPO (control), B = PPO + 0.2 * belief loss. Then 16k-deal benches: each vs anchor & v1, and A vs B.
pkill -f "ladis\." ; for p in $(pgrep -f "bash jobs/current.sh"); do [ "$p" != "$$" ] && kill "$p" 2>/dev/null; done
for i in $(seq 1 30); do pgrep -f "ladis\.train" >/dev/null || break; sleep 5; done
pkill -9 -f "ladis\." ; sleep 2
pgrep -f "tools/authserve.py" >/dev/null || { setsid nohup python3 tools/authserve.py runs > http.log 2>&1 < /dev/null & }
rm -f runs/IDLE runs/AB_DONE
# park the aborted first attempt
[ -d runs/ab1_aborted ] || { mkdir -p runs/ab1_aborted; mv runs/abA runs/abB runs/ab_bench.jsonl runs/ab_bench.out runs/ab1_aborted/ 2>/dev/null; }
R=runs/tf1
START=$R/ckpt_ab_start.pt
[ -f $START ] || cp $R/latest.pt $START
IT0=$(python3 -c "import torch;print(torch.load('$START',map_location='cpu')['iter'])")
ITN=$((IT0 + 40))
echo "=== belief A/B v2 (belief_lr 1e-3) $(date -Is); start iter $IT0" >> $R/train.out
COMMON="--arch tf --d 256 --layers 6 --heads 8 --B 8192 --mb 16384 --micro 2048 --lr 3e-5 --eval_every 1000 \
  --snap_every 1000 --latest_every 10 --target_kl 0.03 --len_sort 1 --compile 1 --seed 0 --iters $ITN --resume $START"
mkdir -p runs/abA runs/abB
python3 -m ladis.train --run runs/abA $COMMON > runs/abA/train.out 2>&1 &
PA=$!
python3 -m ladis.train --run runs/abB $COMMON --belief_coef 0.2 --belief_lr 1e-3 > runs/abB/train.out 2>&1 &
PB=$!
wait $PA $PB
ANCHOR=$(ls $R/ckpt_anchor_it*.pt | head -1)
for arm in A B; do
  other=$([ $arm = A ] && echo B || echo A)
  python3 -m ladis.bench --ckpt runs/ab$arm/latest.pt --deals 16384 --B 2048 \
    --refs $ANCHOR runs/v1/ckpt_paused.pt runs/ab$other/latest.pt --out runs/ab_bench.jsonl >> runs/ab_bench.out 2>&1
done
date -Is > runs/AB_DONE
date -Is > runs/IDLE
