# job: FINAL. Stop training (graceful save) -> belief-weighted search eval of the final tf1 checkpoint,
#      with the whole GPU, results written per matchup as they finish (money runs out ~16:50 UTC).
pkill -f "ladis\." ; for p in $(pgrep -f "bash jobs/current.sh"); do [ "$p" != "$$" ] && kill "$p" 2>/dev/null; done
for i in $(seq 1 30); do pgrep -f "ladis\.train" >/dev/null || break; sleep 5; done
pkill -9 -f "ladis\." ; sleep 2
pgrep -f "tools/authserve.py" >/dev/null || { setsid nohup python3 tools/authserve.py runs > http.log 2>&1 < /dev/null & }
R=runs/tf1
cp $R/latest.pt $R/ckpt_final.pt
echo "=== FINAL: training stopped $(date -Is) for the belief-search eval" >> $R/train.out
date -Is > runs/FINAL_EVAL_STARTED
# 1) belief-weighted search: vs the plain network (GPU only), then vs PIMC, one-seat vs PIMC, network vs PIMC
python3 -m ladis.eval --ckpt $R/ckpt_final.pt --device cuda --search --prior belief --beta 0 --N 32 \
  --nn_deals 300 --deals 150 --B 50 --pimc_k 16 --workers 14 --out runs/final_eval.jsonl > runs/final_eval.out 2>&1
# 2) if money remains: uniform-prior search vs the network on the same deals (does belief beat PIMC-style sampling?)
python3 -m ladis.eval --ckpt $R/ckpt_final.pt --device cuda --search --prior uniform --beta 0 --N 32 \
  --nn_deals 300 --deals 0 --B 50 --pimc_k 16 --workers 14 --out runs/final_eval_uniform.jsonl > runs/final_eval_uniform.out 2>&1
date -Is > runs/FINAL_EVAL_DONE
