#!/usr/bin/env bash
# Check of the trail-point patch (engines\ninfer-trail): tools\trailtest.py on the patched build with the trail on,
# then with the trail off (NINFER_TRAIL_GRID=0), then a comparison of the greedy answers.
#   trail_check.sh [TOKENS] [engine args...]      default 45000 tokens, 262144 window, rk8v4
ROOT=/c/distrib/strata-tuning/inkling
TOKENS="${1:-45000}"; shift
ARGS="${*:---max-context 262144 --expert-residency host --host-cache-mib 16384 --kv-dtype rk8v4}"
cd "$ROOT" || exit 1
for STAGE in trail ref; do
  bash stop_engine.sh basalt >/dev/null 2>&1; bash stop_engine.sh ninfer >/dev/null 2>&1
  if [ $STAGE = ref ]; then export NINFER_TRAIL_GRID=0; else unset NINFER_TRAIL_GRID; fi
  (NAPP=ninfer-trail bash run_ninfer393.sh "trailtest-$STAGE" $ARGS --log-stats-panel on &)
  sleep 5; bash wait_ready.sh 8082 ninfer-serve.exe 420 || { echo "engine did not start"; tail -5 "logs/trailtest-$STAGE.log"; exit 1; }
  echo "== $STAGE $(date +%T)"; grep -i "trail" "logs/trailtest-$STAGE.log" | head -2
  PYTHONUTF8=1 python tools/trailtest.py 8082 inkling $STAGE "$TOKENS" || exit 1
  grep "req#.*done" "logs/trailtest-$STAGE.log" | sed 's/.*INFO  //' | cut -c1-175
done
unset NINFER_TRAIL_GRID
bash stop_engine.sh ninfer >/dev/null 2>&1
echo "== compare"; PYTHONUTF8=1 python tools/trailtest.py - - compare
echo "trail check finished $(date +%T)"
