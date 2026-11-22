#!/usr/bin/env bash
# Запуск oh-my-pi в изоляции: всё, что программа пишет «в профиль пользователя», попадает в harness\omp-user,
# конфиг агента — в harness\omp-home. Настоящий профиль пользователя не трогается.
# Использование: bash omp-run.sh [аргументы omp...]   (рабочая папка — текущая)
H="C:/WORKDIR/harness"
mkdir -p "$H/omp-user" "$H/omp-user/natives"
export USERPROFILE="C:\\WORKDIR\\harness\\omp-user"
export HOME="$H/omp-user"
export PI_CODING_AGENT_DIR="C:\\WORKDIR\\harness\\omp-home"
export PI_NATIVES_DIR="C:\\WORKDIR\\harness\\omp-user\\natives"
export PI_NO_TITLE=1                       # без отдельного запроса заголовка сессии
export PI_AUTO_QA=0                        # без автоматических отчётов о проблемах инструментов
export OTEL_SDK_DISABLED=true              # без экспорта телеметрии
export PI_BUILD_URL="http://127.0.0.1:9"   # проверка обновлений уходит в никуда
exec "$H/omp/omp.exe" "$@"
