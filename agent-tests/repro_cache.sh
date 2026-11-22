#!/usr/bin/env bash
# Diagnostics: why does the first request of a resumed session (omp -c) miss the engine's prompt cache?
#   repro_cache.sh NAME [engine args...]      NInfer PR393 on port 8082, logging proxy on 8081, three short omp steps
# Request bodies: logs\proxy-NAME\NNNN.json, one line per request: logs\proxy-NAME\index.txt
ROOT=/c/WORKDIR/inkling; A=$ROOT/agent-tests; W="C:/WORKDIR/agent-work"
NAME="$1"; shift
cd "$ROOT" || exit 1
bash stop_engine.sh basalt >/dev/null 2>&1; bash stop_engine.sh ninfer >/dev/null 2>&1
rm -rf "logs/proxy-$NAME"; mkdir -p "logs/proxy-$NAME"
(NPORT=8082 NMODEL=inkling-balanced bash run_ninfer393.sh "repro-$NAME" --max-context 262144 --expert-residency host --host-cache-mib 16384 --kv-dtype rk8v4 "$@" &)
sleep 5; bash wait_ready.sh 8082 ninfer-serve.exe 420 || { echo "engine did not start"; exit 1; }
PYTHONUTF8=1 python tools/logproxy.py 8081 8082 "logs/proxy-$NAME" &
PROXY=$!
sleep 2
rm -rf "$W/$NAME" "$A/runs/$NAME"; git clone -q "$A/repos/_bugseed" "$W/$NAME" || exit 1
P="$A/runs/$NAME"; mkdir -p "$P"
printf 'Прочитай файл README.md целиком и одним абзацем скажи, что это за проект. Больше ничего не делай.' > "$P/p1.md"
printf 'Создай файл NOTE.md с одной строкой "проверка" и ответь одним словом: готово.' > "$P/p2.md"
printf 'Не вызывая инструменты, ответь одним предложением: какой файл ты читал в самом начале?' > "$P/p3.md"
OMP_MAXTIME=10m bash "$A/run_step.sh" "$NAME" s1 inkling/inkling-balanced "$P/p1.md"
OMP_MAXTIME=10m bash "$A/run_step.sh" "$NAME" s2 inkling/inkling-balanced "$P/p2.md" -c
OMP_MAXTIME=10m bash "$A/run_step.sh" "$NAME" s3 inkling/inkling-balanced "$P/p3.md" -c
kill $PROXY 2>/dev/null
grep "req#.*done" "logs/repro-$NAME.log" | cut -c1-200
bash stop_engine.sh ninfer >/dev/null 2>&1
rm -rf "$W/$NAME"
echo "repro $NAME finished $(date +%T)"
