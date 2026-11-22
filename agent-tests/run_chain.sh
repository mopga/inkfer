#!/usr/bin/env bash
# The three-step session chain for one model on a fresh copy of the bug-seeded repo:
#   run_chain.sh RUN MODEL        e.g. run_chain.sh qwen-a strata/iq3_s ; run_chain.sh ink-a inkling/inkling
# step1 analysis (read-only) -> step2 bug fixing (same session, -c) -> step3 memory questions (same session, -c).
# The working copy is moved to done/RUN afterwards so the next model cannot see it.
A="C:/WORKDIR/inkling/agent-tests"; W="C:/WORKDIR/agent-work"
RUN="$1"; MODEL="$2"
[ -e "$W/$RUN" ] && { echo "work copy $W/$RUN exists"; exit 1; }
git clone -q "$A/repos/_bugseed" "$W/$RUN" || exit 1
bash "$A/run_step.sh" "$RUN" step1 "$MODEL" "$A/prompts/t1-analysis.md"
bash "$A/run_step.sh" "$RUN" step2 "$MODEL" "$A/prompts/t2-bugfix.md" -c
bash "$A/run_step.sh" "$RUN" step3 "$MODEL" "$A/prompts/t3-retention.md" -c
mkdir -p "$A/done"
# the agent's processes may hold the folder for a few seconds after they exit
for i in 1 2 3 4 5 6 7 8 9 10 11 12; do mv "$W/$RUN" "$A/done/$RUN" 2>/dev/null && break; ping -n 6 127.0.0.1 >/dev/null; done
[ -d "$A/done/$RUN" ] || { cp -a "$W/$RUN" "$A/done/$RUN" && echo "work copy copied (the original stays in $W/$RUN)"; }
echo "chain $RUN finished $(date +%T)"
