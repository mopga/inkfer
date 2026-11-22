#!/usr/bin/env bash
# Decode rate of a build over a short context: quickdec.sh APPDIR LABEL [engine args...]
# Three greedy 300-token requests (the first warms the expert cache); prints the engine's "done" lines.
ROOT=/c/WORKDIR/inkling
APPV="$1"; LABEL="$2"; shift 2
cd "$ROOT" || exit 1
bash stop_engine.sh ninfer >/dev/null 2>&1
(NAPP="$APPV" bash run_ninfer393.sh "$LABEL" --max-context 32768 --expert-residency host --host-cache-mib 4096 --kv-dtype rk4v4 "$@" &)
sleep 4
bash wait_ready.sh 8082 ninfer-serve.exe 240 >/dev/null || { tail -5 "logs/$LABEL.log"; exit 1; }
for i in 1 2 3; do
  curl -s -m 120 http://127.0.0.1:8082/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"inkling","temperature":0,"max_tokens":300,"reasoning_effort":"none","messages":[{"role":"user","content":"Write a Python function that merges overlapping intervals, then explain its complexity in detail."}]}' >/dev/null
done
bash stop_engine.sh ninfer >/dev/null 2>&1
echo "== $LABEL: $*"
grep "req#.*done" "logs/$LABEL.log" | sed 's/.*INFO  //' | cut -c1-210
