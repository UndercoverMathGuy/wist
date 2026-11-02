# job: PAUSED by user request. Training stopped; kill leftover eval workers, stay idle.
pkill -f "ladis\." ; sleep 3
for v in v1 v2; do
  [ -f runs/$v/ckpt.pt ] && [ ! -f runs/$v/ckpt_paused.pt ] && cp runs/$v/ckpt.pt runs/$v/ckpt_paused.pt
done
pgrep -af "ladis\." > runs/paused_procs.txt || echo "no ladis processes running" > runs/paused_procs.txt
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv >> runs/paused_procs.txt
date -Is > runs/PAUSED
