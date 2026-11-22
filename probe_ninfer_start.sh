#!/usr/bin/env bash
# Does NInfer start with these options? probe_ninfer_start.sh LABEL [args]  -> prints the capacity lines or the error. Stops the server afterwards.
ROOT=/c/distrib/strata-tuning/inkling; L="$1"; shift
cd "$ROOT"; bash stop_engine.sh ninfer >/dev/null 2>&1
(bash run_ninfer.sh "$L" "$@" &)
for i in $(seq 1 60); do
  sleep 2
  if curl -s -m 2 http://127.0.0.1:8082/health | grep -q -i ok; then break; fi
  if ! tasklist 2>/dev/null | grep -qi ninfer-serve.exe; then break; fi
done
echo "== $L: $*"
grep -v throughput "logs/$L.log" | grep -i -E "error|capacity|kv|expert|cache|pinned|host|chunk|unsupported|not supported|invalid" | cut -c1-330 | head -${PROBE_LINES:-14}
nvidia-smi --query-gpu=memory.used --format=csv,noheader
[ -n "${PROBE_KEEP:-}" ] || bash stop_engine.sh ninfer >/dev/null 2>&1
