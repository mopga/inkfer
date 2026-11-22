@echo off
rem omp-web 0.5.1 - окно в браузере для oh-my-pi -> Strata (http://127.0.0.1:47831/v1, модель iq3_s).
rem Открывает http://127.0.0.1:30141 (только этот компьютер). Остановить: закрыть это окно или Ctrl+C.
rem Свои настройки и сессии: harness\omp-web\home (отдельно от терминального oh-my-pi в harness\omp-home).
setlocal
set "H=C:\WORKDIR\harness"
set "PI_CODING_AGENT_DIR=%H%\omp-web\home"
set "PI_NATIVES_DIR=%H%\omp-web\natives"
set "PI_NO_TITLE=1"
set "PI_AUTO_QA=0"
set "OTEL_SDK_DISABLED=true"
set "PI_BUILD_URL=http://127.0.0.1:9"
set "NEXT_TELEMETRY_DISABLED=1"
set "BUN_INSTALL_CACHE_DIR=%H%\omp-web\bun-cache"
set "OMP_WEB_BUN=%H%\omp-web\bun\bun-windows-x64\bun.exe"
"%OMP_WEB_BUN%" "%H%\omp-web\app\node_modules\omp-web\bin\omp-web.js" %*
endlocal
