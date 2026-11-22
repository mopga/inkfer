#!/usr/bin/env bash
# The deployed build (engines\ninfer-trail, a second instance on the test port 8082) on the request set of
# plans\p10-fork5-ladder.txt, 131k only: a like-for-like decode comparison with the fork build (same prompts,
# reasoning_effort low, 600-token answers). Label trail-same-131k.
ROOT=/c/WORKDIR/inkling
PY="/d/MODELS/Strata/control/python/python.exe"
LABEL="${1:-trail-same-131k}"; FILLS="${2:-131072}"; APPV="${3:-ninfer-trail}"
cd "$ROOT" || exit 1
bash stop_engine.sh ninfer >/dev/null 2>&1
(NAPP="$APPV" bash run_ninfer393.sh "$LABEL" --max-context 524288 --expert-residency host --host-cache-mib 20480 \
  --kv-dtype rk4v4 --vision --vision-residency overlay --vision-upscale 1.5 &)
sleep 5
bash wait_ready.sh 8082 ninfer-serve.exe 420 || { tail -5 "logs/$LABEL.log"; exit 1; }
"$PY" bench/ctxbench.py --base http://127.0.0.1:8082 --model inkling --label "$LABEL" --proc ninfer-serve.exe \
  --lean --gen 600 --turn-tokens 800 --turns 3 --extra '{"reasoning_effort":"low"}' --fills "$FILLS" > "logs/ctxbench-$LABEL.out" 2>&1
bash stop_engine.sh ninfer >/dev/null 2>&1
cut -c1-240 "logs/ctxbench-$LABEL.out"
grep "req#.*done" "logs/$LABEL.log" | sed 's/.*INFO  //' | cut -c1-230
