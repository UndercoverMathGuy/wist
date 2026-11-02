# job: point 1, step A - evaluate gated search on paused v1 (no training running)
pkill -f "ladis\." ; sleep 2; rm -f runs/PAUSED
CK=runs/v1/ckpt_paused.pt
python3 -m ladis.eval --ckpt $CK --device cuda --search --N 128 --beta 0.5 --deals 200 --B 100 --pimc_k 16 --workers 7 --out runs/search2_eval.jsonl > runs/search2_b05.out 2>&1 &
python3 -m ladis.eval --ckpt $CK --device cuda --search --N 128 --beta 0.0 --deals 200 --B 100 --pimc_k 16 --workers 7 --out runs/search2_eval.jsonl > runs/search2_b0.out 2>&1 &
wait
date -Is > runs/STEP_A_DONE
