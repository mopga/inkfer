#!/usr/bin/env bash
# Один шаг агентного прогона через oh-my-pi (без окна), в изолированной копии проекта.
#   run_step.sh RUN STEP MODEL PROMPTFILE [доп. аргументы omp]
#   RUN   — имя прогона; рабочая копия проекта: C:\distrib\strata-tuning\agent-work\RUN (создаётся заранее)
#   STEP  — имя шага (папка вывода agent-tests\runs\RUN\STEP); шаги одного RUN с ключом -c продолжают одну сессию
#   MODEL — провайдер/модель из harness\omp-home\models.yml: strata/iq3_s или inkling/inkling
# Переменные: OMP_THINKING (high), OMP_MAXTIME (60m), OMP_EXTRA_OVERLAY (доп. надстройка, напр. overlay-ctx128k.yml).
H="C:/distrib/strata-tuning/harness"; A="C:/distrib/strata-tuning/inkling/agent-tests"
RUN="$1"; STEP="$2"; MODEL="$3"; PROMPT="$4"; shift 4
OUT="$A/runs/$RUN/$STEP"; mkdir -p "$OUT" "$A/runs/$RUN/sessions"
EXTRA=(); [ -n "${OMP_EXTRA_OVERLAY:-}" ] && EXTRA=(--config "$A/$OMP_EXTRA_OVERLAY")
cd "C:/distrib/strata-tuning/agent-work/$RUN" || exit 1
date +%FT%T > "$OUT/started.txt"; date +%s > "$OUT/started.epoch"
echo "model=$MODEL prompt=$PROMPT thinking=${OMP_THINKING:-high} maxtime=${OMP_MAXTIME:-60m} extra=${OMP_EXTRA_OVERLAY:-} args=$*" > "$OUT/params.txt"
bash "$H/omp-run.sh" -p --mode json --approval-mode write --config "$A/overlay-perm.yml" "${EXTRA[@]}" --no-title \
  --thinking "${OMP_THINKING:-high}" --max-time "${OMP_MAXTIME:-60m}" --model "$MODEL" \
  --session-dir "$A/runs/$RUN/sessions" "$@" "$(cat "$PROMPT")" < /dev/null > "$OUT/events.jsonl" 2> "$OUT/stderr.txt"
echo "exit $?" > "$OUT/exit.txt"
date +%FT%T > "$OUT/finished.txt"; date +%s > "$OUT/finished.epoch"
