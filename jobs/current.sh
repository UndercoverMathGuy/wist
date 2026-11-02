# job: PPO league v1
nvidia-smi --query-gpu=name,memory.total --format=csv > runs/gpu.txt
python -m ladis.train --run runs/v1 --B 8192 --mb 32768 --eval_every 25 --snap_every 100 --hours 10
