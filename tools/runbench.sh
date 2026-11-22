#!/bin/sh
# runbench.sh LABEL PORT CTX KV HOSTCACHE [extra args...] : start the deployed ninfer-serve in the background, log to logs/LABEL.log
L=$1; P=$2; C=$3; KV=$4; HC=$5; shift 5
H=/c/WORKDIR/inkling
"$H/engines/ninfer-trail/app/ninfer-serve.exe" D:/MODELS/inkling2mini/ninfer/Inkling-2-Mini-UD-IQ4_XS-ninfer-v3.ninfer \
  --model-id inkling-max --host 127.0.0.1 --port $P --cuda-backend 13 --max-context $C --expert-residency host \
  --host-cache-mib $HC --kv-dtype $KV --log-stats-panel on "$@" > "$H/logs/$L.log" 2>&1 &
for i in $(seq 1 300); do curl -sf http://127.0.0.1:$P/health >/dev/null 2>&1 && grep -q listening "$H/logs/$L.log" && break; sleep 1; done
grep -iE "listening|error|vision|overlay|cache" "$H/logs/$L.log" | cut -c1-300 | head -20
nvidia-smi --query-gpu=memory.used --format=csv,noheader
