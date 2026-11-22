#!/usr/bin/env bash
# The long-context autonomous task: read ~14k lines of the Strata server in full (300-400k tokens), write ENV.md /
# ROUTES.md, a code change with a test, a subagent's TESTS-4XX.md, then recall questions in the same session without tools.
#   run_long.sh RUN MODEL        e.g. run_long.sh long-ink1 inkling/inkling-max ; run_long.sh long-qwen1 strata/iq3_s
# The engine must already be running (run_ink.sh CONFIG RUN long starts Inkling; for Qwen: start_qwen.pyw --engine).
# Results: agent-tests\runs\RUN\{L1,L2}; the working copy is moved to agent-tests\done\RUN.
A="C:/WORKDIR/inkling/agent-tests"; W="C:/WORKDIR/agent-work"
RUN="$1"; MODEL="$2"
[ -e "$W/$RUN" ] && { echo "work copy $W/$RUN exists"; exit 1; }
git clone -q "$A/repos/_seed" "$W/$RUN" || exit 1
git -C "$W/$RUN" checkout -q -b work 1974882 || exit 1       # the Strata sources as given, before any agent's run
OMP_MAXTIME="${OMP_MAXTIME:-150m}" bash "$A/run_step.sh" "$RUN" L1 "$MODEL" "$A/prompts/L1-deepread.md"
OMP_MAXTIME=20m bash "$A/run_step.sh" "$RUN" L2 "$MODEL" "$A/prompts/L2-recall.md" -c
mkdir -p "$A/done"
for i in 1 2 3 4 5 6 7 8 9 10 11 12; do mv "$W/$RUN" "$A/done/$RUN" 2>/dev/null && break; ping -n 6 127.0.0.1 >/dev/null; done
[ -d "$A/done/$RUN" ] || { cp -a "$W/$RUN" "$A/done/$RUN" && echo "work copy copied (the original stays in $W/$RUN)"; }
echo "long $RUN finished $(date +%T)"
