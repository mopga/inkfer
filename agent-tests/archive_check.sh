#!/usr/bin/env bash
# Check of a release archive unpacked into an empty directory, as a user would run it:
#   archive_check.sh DIR      DIR is the unpacked ninfer-windows-x64-... folder (run-inkling.bat inside)
# For each profile of run-inkling.bat (small, balanced, max): start on port 8082 with NINFER_MODEL set,
# one chat request; on max also the ladder to 131k (label rel-max-131k). Output: logs\archivecheck.out
ROOT=/c/WORKDIR/inkling
PY="/d/MODELS/Strata/control/python/python.exe"
DIR="$1"
OUT=$ROOT/logs/archivecheck.out
cd "$ROOT" || exit 1
[ -f "$DIR/run-inkling.bat" ] || { echo "no run-inkling.bat in $DIR"; exit 1; }
: > "$OUT"
export NINFER_MODEL='D:\MODELS\inkling2mini\ninfer\Inkling-2-Mini-UD-IQ4_XS-ninfer-v3.ninfer'
export NINFER_PORT=8082
"$DIR/ninfer-serve.exe" --version >> "$OUT" 2>&1

for PROFILE in small balanced max; do
  bash stop_engine.sh ninfer >/dev/null 2>&1
  echo "== $(date +%T) run-inkling.bat $PROFILE" >> "$OUT"
  (cd "$DIR" && cmd //c ".\run-inkling.bat $PROFILE" > "$ROOT/logs/rel-$PROFILE.log" 2>&1 &)
  sleep 5
  if ! bash wait_ready.sh 8082 ninfer-serve.exe 420 >> "$OUT" 2>&1; then
    echo "$PROFILE: NOT STARTED" >> "$OUT"; tail -5 "logs/rel-$PROFILE.log" >> "$OUT"; continue
  fi
  head -2 "logs/rel-$PROFILE.log" >> "$OUT"
  grep -E "capacity \|" "logs/rel-$PROFILE.log" | head -1 | cut -c1-200 >> "$OUT"
  curl -s -m 300 http://127.0.0.1:8082/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"inkling-2-mini","messages":[{"role":"user","content":"Reply with the single word: ready"}],"max_tokens":200}' \
    | "$PY" -c "import json,sys; d=json.load(sys.stdin); print('chat:', repr(d['choices'][0]['message']['content']), d['usage'])" >> "$OUT" 2>&1
  if [ "$PROFILE" = max ]; then
    "$PY" bench/ctxbench.py --base http://127.0.0.1:8082 --model inkling-2-mini --label rel-max-131k --proc ninfer-serve.exe \
      --lean --gen 600 --turn-tokens 800 --turns 3 --extra '{"reasoning_effort":"low"}' --fills 131072 > logs/ctxbench-rel-max-131k.out 2>&1
    cut -c1-260 logs/ctxbench-rel-max-131k.out >> "$OUT"
  fi
done
bash stop_engine.sh ninfer >/dev/null 2>&1
unset NINFER_MODEL NINFER_PORT
echo "archive check finished $(date +%T)" >> "$OUT"
