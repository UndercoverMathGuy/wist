# job: END OF DAY. Gracefully stop all training (trainer saves latest.pt on SIGTERM), then stay idle.
pkill -f "ladis\." ; for p in $(pgrep -f "bash jobs/current.sh"); do [ "$p" != "$$" ] && kill "$p" 2>/dev/null; done
for i in $(seq 1 30); do pgrep -f "ladis\.train" >/dev/null || break; sleep 5; done
pkill -9 -f "ladis\." ; sleep 2
pgrep -f "tools/authserve.py" >/dev/null || { setsid nohup python3 tools/authserve.py runs > http.log 2>&1 < /dev/null & }
echo "=== END OF DAY: training stopped $(date -Is)" >> runs/tf1/train.out
pgrep -af "ladis\." > runs/end_status.txt || echo "no ladis processes running" > runs/end_status.txt
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv >> runs/end_status.txt
date -Is > runs/END_OF_DAY
