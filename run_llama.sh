#!/usr/bin/env bash
# Start the Unsloth llama.cpp server with Inkling 2 Mini for a test run.
#   run_llama.sh LABEL QUANT [llama-server args...]      QUANT: IQ4_XS | Q4_K_XL
# Log: inkling/logs/LABEL.log. Port 8090, this computer only. cuBLAS 13 is taken from Strata's engine folder.
set -u
LABEL="$1"; QUANT="$2"; shift 2
ROOT=/c/distrib/strata-tuning/inkling
APP=$ROOT/engines/llamacpp-unsloth-b12588/app
export PATH="/c/Users/USER/AppData/Local/Strata/components/engine/0.1.59-windows-x86_64-cuda-cuda13:$PATH"
cd "$APP" || exit 1
exec ./llama-server.exe -m "D:/AI/models/inkling2mini/Inkling-2-Mini-UD-$QUANT.gguf" --jinja -np 1 \
  --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0.0 --host 127.0.0.1 --port 8090 --alias inkling --metrics "$@" \
  > "$ROOT/logs/$LABEL.log" 2>&1
