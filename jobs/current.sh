# job: PAUSED by user request. Training stopped; snapshot latest checkpoints and stay idle.
for v in v1 v2; do
  [ -f runs/$v/ckpt.pt ] && cp runs/$v/ckpt.pt runs/$v/ckpt_paused.pt
done
pgrep -af "ladis\." > runs/paused_procs.txt || echo "no ladis processes running" > runs/paused_procs.txt
date -Is > runs/PAUSED
