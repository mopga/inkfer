#!/usr/bin/env bash
# Two conversations taking turns with a deliberately small host store (tools\alttest.py): the first one must keep
# its cache while the second makes several requests in a row.
#   alt_check.sh [LABEL] [HOST_CACHE_MIB] [alttest args...]      defaults: alttest 3072; NAPP picks the build
ROOT=/c/WORKDIR/inkling
LABEL="${1:-alttest}"; MIB="${2:-3072}"; shift 2 2>/dev/null
cd "$ROOT" || exit 1
bash stop_engine.sh basalt >/dev/null 2>&1; bash stop_engine.sh ninfer >/dev/null 2>&1
(NAPP="${NAPP:-ninfer-trail}" bash run_ninfer393.sh "$LABEL" --max-context 262144 --expert-residency host --host-cache-mib "$MIB" --kv-dtype rk8v4 --log-stats-panel on &)
sleep 5; bash wait_ready.sh 8082 ninfer-serve.exe 420 || { echo "engine did not start"; tail -5 "logs/$LABEL.log"; exit 1; }
echo "== $LABEL, build ${NAPP:-ninfer-trail}, host store $MIB MiB, $(date +%T)"
PYTHONUTF8=1 python tools/alttest.py 8082 inkling "$@"; RC=$?
grep -i "host store\|long anchor" "logs/$LABEL.log" | cut -c1-200 | tail -8
bash stop_engine.sh ninfer >/dev/null 2>&1
echo "alt check finished $(date +%T), exit $RC"; exit $RC
