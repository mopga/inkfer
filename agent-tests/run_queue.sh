#!/usr/bin/env bash
# The comparison queue after long-ink1: Inkling chain (max) and big task (balanced), then Qwen's long task and a second
# Qwen chain on the Strata engine, then the balanced Inkling profile back through the launcher (agents' default = Inkling).
#   run_queue.sh            (waits for runs/long-ink1.log to say "finished")
ROOT=/c/distrib/strata-tuning/inkling; A=$ROOT/agent-tests; PY=/d/AI/models/Strata/control/python/python.exe
PYW=/d/AI/models/Strata/control/python/pythonw.exe
cd "$ROOT" || exit 1
until grep -q "long-ink1 finished" "$A/runs/long-ink1.log" 2>/dev/null; do sleep 20; done
echo "$(date +%T) queue start"
bash "$A/run_ink.sh" prof-max ink-p1 chain > "$A/runs/ink-p1.log" 2>&1;           echo "$(date +%T) ink-p1 done"
bash "$A/run_ink.sh" prof-balanced ink-task3 task > "$A/runs/ink-task3.log" 2>&1;  echo "$(date +%T) ink-task3 done"
"$PYW" "$ROOT/start_qwen.pyw" --engine;                                            echo "$(date +%T) qwen engine: $?"
bash "$A/run_long.sh" long-qwen1 strata/iq3_s > "$A/runs/long-qwen1.log" 2>&1;     echo "$(date +%T) long-qwen1 done"
bash "$A/run_chain.sh" qwen-b strata/iq3_s > "$A/runs/qwen-b.chain.log" 2>&1;      echo "$(date +%T) qwen-b done"
"$PYW" "$ROOT/start_inkling.pyw" balanced;                                         echo "$(date +%T) inkling balanced back: $?"
echo "$(date +%T) queue finished"
