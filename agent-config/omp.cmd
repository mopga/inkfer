@echo off
rem oh-my-pi 18.18.8 -> Strata (шлюз приложения http://127.0.0.1:47831/v1, модель iq3_s).
rem Запуск из папки проекта:  C:\WORKDIR\harness\omp.cmd
rem Конфиг агента лежит в harness\omp-home (models.yml, config.yml), а не в профиле пользователя.
rem В профиле программа создаёт только %USERPROFILE%\.omp\install-id (случайный идентификатор установки).
setlocal
set "H=C:\WORKDIR\harness"
set "PI_CODING_AGENT_DIR=%H%\omp-home"
set "PI_NATIVES_DIR=%H%\omp-user\natives"
set "PI_NO_TITLE=1"
set "PI_AUTO_QA=0"
set "OTEL_SDK_DISABLED=true"
set "PI_BUILD_URL=http://127.0.0.1:9"
"%H%\omp\omp.exe" %*
endlocal
