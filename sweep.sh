#!/usr/bin/env bash
# One engine start + one ctxbench run per line of a plan file, one after another (single GPU).
#   sweep.sh PLANFILE
# Plan line:  ENGINE|LABEL|ENGINE ARGS|CTXBENCH ARGS[|ENV]   (ENGINE: basalt | bpatched | bmain | ninfer | ninfer393; # = skipped)
#   ENV: space-separated VAR=VALUE for the engine (BASALT_INK_* switches). After ctxbench a greedy determinism set is saved
#   (bench/runs/det-LABEL.json) when the plan line's CTXBENCH ARGS contain --det (the flag itself is not passed on).
# Engine log: logs/LABEL.log, bench output: logs/ctxbench-LABEL.out, results: bench/runs/LABEL.json. Progress: logs/sweep.log
ROOT=/c/WORKDIR/inkling
PY="/d/MODELS/Strata/control/python/python.exe"
cd "$ROOT" || exit 1
while IFS='|' read -r ENGINE LABEL EARGS BARGS ENVS; do
  case "$ENGINE" in ''|\#*) continue;; esac
  case "$ENGINE" in
    basalt) PORT=8081; PROC=basalt.exe;  RUN="run_basalt.sh $LABEL --alias inkling $EARGS";;
    bpatched) PORT=8081; PROC=basalt.exe; RUN="run_basalt_src.sh app $LABEL --alias inkling $EARGS";;
    bmain) PORT=8081; PROC=basalt.exe; RUN="run_basalt_src.sh baseline $LABEL --alias inkling $EARGS";;
    ninfer) PORT=8082; PROC=ninfer-serve.exe; RUN="run_ninfer.sh $LABEL $EARGS";;
    ninfer393) PORT=8082; PROC=ninfer-serve.exe; RUN="run_ninfer393.sh $LABEL $EARGS";;
    *) echo "unknown engine $ENGINE" >> logs/sweep.log; continue;;
  esac
  bash stop_engine.sh basalt >/dev/null 2>&1; bash stop_engine.sh ninfer >/dev/null 2>&1
  echo "$(date +%T) start $LABEL: $EARGS ${ENVS:-}" >> logs/sweep.log
  (BENV="${ENVS:-}" bash $RUN &)
  sleep 5
  if ! bash wait_ready.sh $PORT $PROC 420 >> logs/sweep.log 2>&1; then
    echo "$(date +%T) $LABEL: NOT STARTED: $(grep -i -E 'error|refus|short|fatal' logs/$LABEL.log | tail -2 | cut -c1-300)" >> logs/sweep.log
    continue
  fi
  sleep 3
  if ! tasklist 2>/dev/null | grep -qi "$PROC"; then
    echo "$(date +%T) $LABEL: DIED AFTER START: $(tail -2 logs/$LABEL.log | cut -c1-300)" >> logs/sweep.log; continue
  fi
  DET=0; case "$BARGS" in *--det*) DET=1; BARGS="${BARGS//--det/}";; esac
  eval "\"$PY\" bench/ctxbench.py --base http://127.0.0.1:$PORT --model inkling --label $LABEL --proc $PROC $BARGS" > "logs/ctxbench-$LABEL.out" 2>&1
  [ "$DET" = 1 ] && "$PY" bench/detcheck.py --base http://127.0.0.1:$PORT --label $LABEL > "logs/det-$LABEL.out" 2>&1
  echo "$(date +%T) done $LABEL" >> logs/sweep.log
  grep -v warmup "logs/ctxbench-$LABEL.out" | cut -c1-260 >> logs/sweep.log
done < "$1"
bash stop_engine.sh basalt >/dev/null 2>&1; bash stop_engine.sh ninfer >/dev/null 2>&1
echo "$(date +%T) sweep finished: $1" >> logs/sweep.log
