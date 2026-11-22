#!/usr/bin/env bash
# Start Basalt 0.5.5 with the Inkling 2 Mini pack for a test run.   run_basalt.sh LABEL [server args...]
# Log: inkling/logs/LABEL.log. Port 8081, this computer only. The CUDA 13 runtime is taken from Strata's engine folder.
set -u
LABEL="$1"; shift
ROOT=/c/distrib/strata-tuning/inkling
APP=$ROOT/engines/basalt/app/basalt-v0.5.5-windows-x64
export PATH="/c/Users/USER/AppData/Local/Strata/components/engine/0.1.59-windows-x86_64-cuda-cuda13:$PATH"
export PYTHONUTF8=1
cd "$APP" || exit 1
exec ./bin/basalt.exe serve "D:/AI/models/inkling2mini/basalt/Inkling-2-Mini-NVFP4-q8.basalt" --port 8081 "$@" > "$ROOT/logs/$LABEL.log" 2>&1
