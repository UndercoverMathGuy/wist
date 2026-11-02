# job: PPO league v1 (resume) + periodic PIMC eval on spare CPU cores
nvidia-smi --query-gpu=name,memory.total --format=csv > runs/gpu.txt
R=""; [ -f runs/v1/ckpt.pt ] && R="--resume runs/v1/ckpt.pt"
python3 -m ladis.train --run runs/v1 --B 8192 --mb 32768 --eval_every 25 --snap_every 100 --hours 10 $R &
sleep 120
while true; do
  cp runs/v1/ckpt.pt runs/v1/ckpt_eval.pt
  python3 -m ladis.eval --ckpt runs/v1/ckpt_eval.pt --deals 400 --pimc_k 16 --workers 14 --out runs/v1/pimc_eval.jsonl
  sleep 600
done
