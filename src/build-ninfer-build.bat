@echo off
rem Build the launchers and the runtime DLLs. Usage: build-ninfer-build.bat [jobs]   (default 6)
rem Runs at low priority: the machine may be benchmarking a model at the same time.
call "%~dp0build-ninfer-env.bat"
rem MSVC prints /showIncludes in the console code page; rules.ninja carries the prefix in CP866 (see BUILD-NOTES).
rem A UTF-8 console (65001) makes ninja record no header deps, so pin 866 for the build.
chcp 866 >nul
set "JOBS=%~1"
if "%JOBS%"=="" set "JOBS=6"
start "" /low /b /wait cmake --build "%NINFER_BUILD%" --parallel %JOBS% --target ninfer-serve ninfer ninfer-perplexity ninfer_bench ninfer-calibrate ninfer-ngram-profile
echo build_exit=%ERRORLEVEL%
