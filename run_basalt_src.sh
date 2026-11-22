#!/usr/bin/env bash
# Start the locally built Basalt (main 97e2e71 + BASALT_INK_* patches) with the Inkling 2 Mini pack.
#   run_basalt_src.sh app|baseline LABEL [server args...]     env: BENV="VAR=V VAR2=V" (BASALT_INK_* switches)
# app = patched build, baseline = unpatched main. Log: inkling/logs/LABEL.log. Port 8081, this computer only.
set -u
TREE="$1"; LABEL="$2"; shift 2
ROOT=/c/distrib/strata-tuning/inkling
APP=$ROOT/engines/basalt-src/$TREE
export PYTHONUTF8=1
for kv in ${BENV:-}; do export "$kv"; done
cd "$APP" || exit 1
exec ./bin/basalt.exe serve "D:/AI/models/inkling2mini/basalt/Inkling-2-Mini-NVFP4-q8.basalt" --port 8081 "$@" > "$ROOT/logs/$LABEL.log" 2>&1
