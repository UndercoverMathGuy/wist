# job: reload the auth server (adds browser cookie login at /login), then stay idle
pkill -f "tools/authserve.py"; sleep 1
setsid nohup python3 tools/authserve.py runs > http.log 2>&1 < /dev/null &
sleep 2
pgrep -af "tools/authserve.py" > runs/server_status.txt
date -Is > runs/IDLE
