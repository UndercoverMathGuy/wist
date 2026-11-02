# job: v1 (flat features) + v2 (play-history features) training; periodic PIMC eval; one-off search eval
nvidia-smi --query-gpu=name,memory.total --format=csv > runs/gpu.txt
R=""; [ -f runs/v1/ckpt.pt ] && R="--resume runs/v1/ckpt.pt"
python3 -m ladis.train --run runs/v1 --B 8192 --mb 32768 --eval_every 25 --snap_every 100 --hours 10 $R >> runs/v1.out 2>&1 &
R2=""; [ -f runs/v2/ckpt.pt ] && R2="--resume runs/v2/ckpt.pt"
python3 -m ladis.train --run runs/v2 --hist --lr 2e-4 --B 8192 --mb 32768 --eval_every 25 --snap_every 100 --hours 10 $R2 >> runs/v2.out 2>&1 &
sleep 60
cp runs/v1/ckpt.pt runs/v1/ckpt_search.pt
python3 -m ladis.eval --ckpt runs/v1/ckpt_search.pt --device cuda --search --N 128 --M 32 --deals 200 --B 100 --pimc_k 16 --workers 6 --out runs/search_eval.jsonl > runs/search_eval.out 2>&1 &
while true; do
  for v in v1 v2; do
    [ -f runs/$v/ckpt.pt ] || continue
    cp runs/$v/ckpt.pt runs/$v/ckpt_eval.pt
    python3 -m ladis.eval --ckpt runs/$v/ckpt_eval.pt --deals 300 --pimc_k 16 --workers 8 --out runs/pimc_eval.jsonl
  done
  sleep 300
done
