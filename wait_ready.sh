#!/usr/bin/env bash
# Wait until the test server on the given port has its model loaded (or its process is gone). wait_ready.sh [PORT] [PROC] [SECONDS]
# Basalt answers /health with "status ok" while it is still loading ("loaded": false) - that is not ready.
PORT="${1:-8090}"; PROC="${2:-llama-server}"; MAX="${3:-240}"
t0=$(date +%s)
while :; do
  h=$(curl -s -m 2 "http://127.0.0.1:$PORT/health")
  if echo "$h" | grep -q -i 'ok' && ! echo "$h" | grep -q '"loaded": false'; then echo "ready after $(( $(date +%s) - t0 ))s"; exit 0; fi
  if ! tasklist 2>/dev/null | grep -qi "$PROC"; then echo "process $PROC is not running"; exit 1; fi
  if [ $(( $(date +%s) - t0 )) -ge "$MAX" ]; then echo "not ready after ${MAX}s"; exit 2; fi
  ping -n 3 127.0.0.1 >/dev/null
done
