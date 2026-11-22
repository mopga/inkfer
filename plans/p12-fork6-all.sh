#!/usr/bin/env bash
# Everything that qualifies the fork build before a release, one after another on port 8082:
# the short ladder with the greedy determinism set (plans\p12-fork6-ladder.txt), then the launch files,
# the API and vision checks and the trail check (agent-tests\launchfiles_check.sh).
cd /c/WORKDIR/inkling || exit 1
bash sweep.sh plans/p12-fork6-ladder.txt
bash agent-tests/launchfiles_check.sh
bash stop_engine.sh ninfer >/dev/null 2>&1
echo "p12 finished $(date +%T)"
