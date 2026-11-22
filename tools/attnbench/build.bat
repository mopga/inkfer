@echo off
rem Builds attn_bench.exe (Inkling 2 attention timing) from the engine sources: call from cmd.
call C:\distrib\strata-tuning\inkling\src\build-ninfer-env.bat
set "S=%NINFER_SRC%"
cd /d %~dp0
nvcc -O3 -std=c++20 -DNOMINMAX -DWIN32_LEAN_AND_MEAN -DNINFER_SM120_FP8=1 -DNINFER_TIER_BLACKWELL=1 ^
  "--generate-code=arch=compute_120a,code=[sm_120a]" -Xcompiler="/EHsc /utf-8 /Zc:preprocessor /Zc:__cplusplus -MD" ^
  -I"%S%\include" -I"%S%\src" -isystem "%S%\third_party" %* attn_bench.cu -o attn_bench.exe -lcublas
