# job: STOP all runs (user request: switching to transformer). Keep auth server, stay idle.
pkill -f "ladis\." ; sleep 3
pgrep -f "tools/authserve.py" >/dev/null || { setsid nohup python3 tools/authserve.py runs > http.log 2>&1 < /dev/null & }
pgrep -af "ladis\." > runs/stop_status.txt || echo "no ladis processes running" > runs/stop_status.txt
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv >> runs/stop_status.txt
date -Is > runs/IDLE
