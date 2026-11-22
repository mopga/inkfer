#!/usr/bin/env bash
# Check of the fork's 16 GB launch files (src\ninfer-fork\configs\inkling2-mini-16gb*.ini) through its
# scripts\run-inkling.bat in the checkout layout, with the packaged server of engines\ninfer-inkfer:
#   small     start, short ladder to ~122k (MXFP8 KV, 131,072 window; the runs of November 22 carry
#             the label lf-speed, the profile's first name)
#   balanced  start, short ladder to 131k and ~250k (rk8v4, 262,144 window)
#   max       start, one chat request, tools\apicheck.py, tools\vischeck.py (rk4v4, 524,288 window;
#             its ladder is plans\p10-fork5-ladder.txt, the same options)
# then agent-tests\trail_check.sh on the same build. Port 8082. Output: logs\launchcheck.out
ROOT=/c/WORKDIR/inkling
PY="/d/MODELS/Strata/control/python/python.exe"
FORK=$ROOT/src/ninfer-fork
OUT=$ROOT/logs/launchcheck.out
cd "$ROOT" || exit 1
: > "$OUT"
export NINFER_MODEL='D:\MODELS\inkling2mini\ninfer\Inkling-2-Mini-UD-IQ4_XS-ninfer-v3.ninfer'
export NINFER_PORT=8082
export NINFER_SERVER='C:\WORKDIR\inkling\engines\ninfer-inkfer\app\ninfer-serve.exe'

start() {  # start PROFILE LABEL
  bash stop_engine.sh ninfer >/dev/null 2>&1
  echo "== $(date +%T) run-inkling.bat $1" >> "$OUT"
  (cd "$FORK/scripts" && cmd //c ".\run-inkling.bat $1" > "$ROOT/logs/$2.log" 2>&1 &)
  sleep 5
  if ! bash wait_ready.sh 8082 ninfer-serve.exe 420 >> "$OUT" 2>&1; then
    echo "$1: NOT STARTED" >> "$OUT"; tail -5 "logs/$2.log" >> "$OUT"; return 1
  fi
  head -3 "logs/$2.log" >> "$OUT"
  grep -i -E "kv|context|trail|vision" "logs/$2.log" | head -8 | cut -c1-220 >> "$OUT"
}
ladder() {  # ladder LABEL FILLS
  "$PY" bench/ctxbench.py --base http://127.0.0.1:8082 --model inkling-2-mini --label "$1" --proc ninfer-serve.exe \
    --lean --gen 600 --turn-tokens 800 --turns 3 --extra '{"reasoning_effort":"low"}' --fills "$2" > "logs/ctxbench-$1.out" 2>&1
  cut -c1-260 "logs/ctxbench-$1.out" >> "$OUT"
}

start small lf-small && ladder lf-small 122000
start balanced lf-balanced && ladder lf-balanced 131072,250000
if start max lf-max; then
  curl -s -m 300 http://127.0.0.1:8082/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"inkling-2-mini","messages":[{"role":"user","content":"Reply with the single word: ready"}],"max_tokens":200}' | cut -c1-600 >> "$OUT"
  echo >> "$OUT"
  echo "== apicheck" >> "$OUT"; PYTHONUTF8=1 "$PY" tools/apicheck.py 8082 >> "$OUT" 2>&1
  echo "== vischeck" >> "$OUT"; PYTHONUTF8=1 "$PY" tools/vischeck.py 8082 >> "$OUT" 2>&1
fi
bash stop_engine.sh ninfer >/dev/null 2>&1
unset NINFER_MODEL NINFER_PORT NINFER_SERVER
if [ -z "$SKIP_TRAIL" ]; then
  echo "== trail_check $(date +%T)" >> "$OUT"
  TAPP=ninfer-inkfer bash agent-tests/trail_check.sh >> "$OUT" 2>&1
fi
bash stop_engine.sh ninfer >/dev/null 2>&1
echo "launch file check finished $(date +%T)" >> "$OUT"
