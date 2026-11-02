#!/usr/bin/env bash
# Runs inside a RunPod pod. Pulls this branch every 60s; when jobs/current.sh changes,
# (re)starts it. Serves ./runs read-only on :8000 so logs/checkpoints can be fetched.
set -ux
echo "podloop start $(date -Is)"; which git python python3 curl; nvidia-smi -L || true
BR=${BRANCH:-claude/wonderful-gates-vi0uuy}
cd /workspace
[ -d wist ] || git clone -b "$BR" https://github.com/UndercoverMathGuy/wist
cd wist
mkdir -p runs
pip install -q endplay > pip.log 2>&1 &
setsid nohup python3 tools/authserve.py runs > http.log 2>&1 &
last=""; pid=""
while true; do
  git fetch -q origin "$BR" && git reset -q --hard "origin/$BR"
  cur=$(sha1sum jobs/current.sh 2>/dev/null | cut -c1-40)
  if [ "$cur" != "$last" ]; then
    if [ -n "$pid" ]; then pkill -TERM -P "$pid"; kill "$pid" 2>/dev/null; sleep 5; fi
    echo "$(date -Is) starting job $cur" >> runs/podloop.log
    nohup bash jobs/current.sh > "runs/job_$(date +%s).log" 2>&1 &
    pid=$!; last=$cur
  fi
  sleep 60
done
