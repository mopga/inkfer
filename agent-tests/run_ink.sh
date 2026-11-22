#!/usr/bin/env bash
# Start one Inkling engine configuration on port 8081, run the three-step agent chain on it, stop the engine.
#   run_ink.sh CONFIG RUN [chain|task]
#     CONFIG  prof-balanced | prof-max | n393-rk8 | n393-rk4 | n393-rk4-512k | n393-bf16-128k | bpatched | bpatched-512k   (see the case below)
#     RUN     name of the run (results: agent-tests\runs\RUN, work copy: agent-tests\done\RUN)
#     chain   (default) run_chain.sh;  long: run_long.sh (300-400k-token reading task);  task: the big autonomous task (harness\task-prompt.md) in a fresh clone of _seed
# The engine answers to the model id of the profile (inkling-balanced / inkling-max / inkling-speed), so oh-my-pi uses
# that profile's context window and compaction threshold (harness\omp-home\models.yml, config.yml).
ROOT=/c/WORKDIR/inkling; A=$ROOT/agent-tests
CONFIG="$1"; RUN="$2"; WHAT="${3:-chain}"
BP="BASALT_INK_ADAPT=1 BASALT_INK_STAGING=1 BASALT_INK_ADAPT_LOG=1 BASALT_INK_DECODE_KEEP_MIB=256 BASALT_INK_ADAPT_INTERVAL=32 BASALT_INK_ADAPT_SWAPS=96"
cd "$ROOT" || exit 1
export NAPP="${NAPP:-ninfer-trail}"     # NInfer build: engines/ninfer-trail (PR393 + trail points); NAPP=ninfer-pr393 for the one without
bash stop_engine.sh basalt >/dev/null 2>&1; bash stop_engine.sh ninfer >/dev/null 2>&1
case "$CONFIG" in
  n393-rk8)       ID=inkling-balanced; PROC=ninfer-serve.exe
                  (NPORT=8081 NMODEL=$ID bash run_ninfer393.sh "agent-$RUN" --max-context 262144 --expert-residency host --host-cache-mib 16384 --kv-dtype rk8v4 &) ;;
  n393-rk4)       ID=inkling-balanced; PROC=ninfer-serve.exe
                  (NPORT=8081 NMODEL=$ID bash run_ninfer393.sh "agent-$RUN" --max-context 262144 --expert-residency host --host-cache-mib 16384 --kv-dtype rk4v4 &) ;;
  n393-rk4-512k)  ID=inkling-max; PROC=ninfer-serve.exe
                  (NPORT=8081 NMODEL=$ID bash run_ninfer393.sh "agent-$RUN" --max-context 524288 --expert-residency host --host-cache-mib 20480 --kv-dtype rk4v4 &) ;;
  n393-bf16-128k) ID=inkling-speed; PROC=ninfer-serve.exe
                  (NPORT=8081 NMODEL=$ID bash run_ninfer393.sh "agent-$RUN" --max-context 131072 --expert-residency host --host-cache-mib 12288 &) ;;
  prof-balanced)  ID=inkling-balanced; PROC=ninfer-serve.exe      # = profiles.json balanced (vision overlay, far-key build)
                  (NPORT=8081 NMODEL=$ID bash run_ninfer393.sh "agent-$RUN" --max-context 262144 --expert-residency host --host-cache-mib 16384 --kv-dtype rk8v4 --vision --vision-residency overlay --vision-upscale 1.5 &) ;;
  prof-max)       ID=inkling-max; PROC=ninfer-serve.exe           # = profiles.json max
                  (NPORT=8081 NMODEL=$ID bash run_ninfer393.sh "agent-$RUN" --max-context 524288 --expert-residency host --host-cache-mib 20480 --kv-dtype rk4v4 --vision --vision-residency overlay --vision-upscale 1.5 &) ;;
  bpatched)       ID=inkling-balanced; PROC=basalt.exe
                  (BENV="$BP" bash run_basalt_src.sh app "agent-$RUN" --alias $ID --max-context 262144 --kv fp8 &) ;;
  bpatched-512k)  ID=inkling-max; PROC=basalt.exe
                  (BENV="$BP" bash run_basalt_src.sh app "agent-$RUN" --alias $ID --max-context 524288 --kv fp8 --spec 1 &) ;;
  *) echo "unknown config $CONFIG"; exit 2 ;;
esac
sleep 5
bash wait_ready.sh 8081 $PROC 420 || { echo "engine did not start: $(tail -3 logs/agent-$RUN.log)"; exit 1; }
sleep 3
echo "$(date +%T) $CONFIG ready, model inkling/$ID, $WHAT $RUN"
if [ "$WHAT" = task ]; then
  W="C:/WORKDIR/agent-work"
  [ -e "$W/$RUN" ] && { echo "work copy $W/$RUN exists"; exit 1; }
  git clone -q "$A/repos/_seed" "$W/$RUN" || exit 1
  git -C "$W/$RUN" checkout -q -b work 1974882 || exit 1      # the state before Qwen's run1/run2
  OMP_MAXTIME=90m bash "$A/run_step.sh" "$RUN" task "inkling/$ID" "C:/WORKDIR/harness/task-prompt.md"
  mkdir -p "$A/done"
  for i in 1 2 3 4 5 6 7 8 9 10 11 12; do mv "$W/$RUN" "$A/done/$RUN" 2>/dev/null && break; ping -n 6 127.0.0.1 >/dev/null; done
  [ -d "$A/done/$RUN" ] || cp -a "$W/$RUN" "$A/done/$RUN"
elif [ "$WHAT" = long ]; then
  bash "$A/run_long.sh" "$RUN" "inkling/$ID"
else
  bash "$A/run_chain.sh" "$RUN" "inkling/$ID"
fi
nvidia-smi --query-gpu=memory.used --format=csv,noheader > "$A/runs/$RUN/vram-end.txt" 2>/dev/null
cd "$ROOT"; bash stop_engine.sh basalt >/dev/null 2>&1; bash stop_engine.sh ninfer >/dev/null 2>&1
echo "$(date +%T) $RUN finished"
