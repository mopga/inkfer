#!/usr/bin/env bash
# Start the local experimental NInfer-all build (PR #393: quantized KV for Inkling, + 2 local patches) for a test run.
#   run_ninfer393.sh LABEL [ninfer-serve args...]     e.g. --kv-dtype rk4v4 --max-context 524288
# NPORT=8081 NMODEL=inkling-balanced run_ninfer393.sh ... : another port / model id (agent runs go to 8081)
# NAPP=ninfer-trail run_ninfer393.sh ... : the build with trail points (engines/ninfer-trail), default ninfer-pr393
# Log: inkling/logs/LABEL.log. Port 8082, this computer only.
set -u
LABEL="$1"; shift
ROOT=/c/WORKDIR/inkling
APP=$ROOT/engines/${NAPP:-ninfer-pr393}/app
cd "$APP" || exit 1
exec ./ninfer-serve.exe "D:/MODELS/inkling2mini/ninfer/Inkling-2-Mini-UD-IQ4_XS-ninfer-v3.ninfer" \
  --model-id "${NMODEL:-inkling}" --host 127.0.0.1 --port "${NPORT:-8082}" --cuda-backend 13 \
  --request-log-jsonl "$ROOT/logs/$LABEL.requests.jsonl" "$@" \
  > "$ROOT/logs/$LABEL.log" 2>&1
