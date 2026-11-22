#!/usr/bin/env bash
# Start NInfer-all (release 2026-11-21.4) with Inkling 2 Mini UD-IQ4_XS for a test run.
#   run_ninfer.sh LABEL [ninfer-serve args...]
# Log: inkling/logs/LABEL.log. Port 8082, this computer only. cuBLAS 13 ships inside the build.
set -u
LABEL="$1"; shift
ROOT=/c/distrib/strata-tuning/inkling
APP=$ROOT/engines/ninfer/app/ninfer-windows-x64-cuda13.4-sm75-80-86-89-120a-bd8195f42e87
cd "$APP" || exit 1
exec ./ninfer-serve.exe "D:/AI/models/inkling2mini/ninfer/Inkling-2-Mini-UD-IQ4_XS-ninfer-v3.ninfer" \
  --model-id "${NMODEL:-inkling}" --host 127.0.0.1 --port "${NPORT:-8082}" --cuda-backend 13 \
  --request-log-jsonl "$ROOT/logs/$LABEL.requests.jsonl" "$@" \
  > "$ROOT/logs/$LABEL.log" 2>&1
