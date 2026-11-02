# job: SECURITY - replace the open file server with a token-authenticated one, then re-run step A
pkill -f "http.server 8000"; pkill -f "ladis\."; sleep 2
pgrep -f "tools/authserve.py" >/dev/null || setsid nohup python3 tools/authserve.py runs > http.log 2>&1 < /dev/null &
sleep 2
rm -f runs/STEP_A_DONE
CK=runs/v1/ckpt_paused.pt
python3 -m ladis.eval --ckpt $CK --device cuda --search --N 128 --beta 0.5 --deals 200 --B 100 --pimc_k 16 --workers 7 --out runs/search2_eval.jsonl > runs/search2_b05.out 2>&1 &
python3 -m ladis.eval --ckpt $CK --device cuda --search --N 128 --beta 0.0 --deals 200 --B 100 --pimc_k 16 --workers 7 --out runs/search2_eval.jsonl > runs/search2_b0.out 2>&1 &
wait
date -Is > runs/STEP_A_DONE
