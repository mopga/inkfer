@echo off
rem Builds Basalt (inkling2 only) from C:\distrib\strata-tuning\inkling\src\basalt with MSVC 14.44 + the unpacked CUDA 13.4 toolkit.
rem   build-basalt.cmd            configure (first time) and build the engine programs
rem   build-basalt.cmd tests      the same with the CPU tests of the Inkling model (spill, ink, expert map, walk, pretok)
rem The executables land in src\basalt\bin (basalt.exe, basalt-bench.exe, basalt-probe.exe). install-basalt.cmd copies
rem basalt.exe into the runnable tree (engines\basalt-src\app\bin).
setlocal
set ROOT=C:\distrib\strata-tuning\inkling
set SRC=%ROOT%\src\basalt
set CUDA_PATH=%ROOT%\toolchain\cuda-13.4
set VS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools
call "%VS%\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.44 >nul || exit /b 1
set PATH=%CUDA_PATH%\bin;%CUDA_PATH%\bin\x64;%VS%\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;%VS%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;%PATH%
set TESTING=OFF
if /i "%1"=="tests" set TESTING=ON
set BUILD=%SRC%\build-ink
if not exist "%BUILD%\build.ninja" (
  cmake -S "%SRC%" -B "%BUILD%" -G Ninja -DCMAKE_BUILD_TYPE=Release -DBASALT_MODELS=inkling2 -DBUILD_TESTING=%TESTING% ^
    -DCUDAToolkit_ROOT="%CUDA_PATH%" -DCMAKE_CUDA_COMPILER="%CUDA_PATH%\bin\nvcc.exe" -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl || exit /b 1
) else (
  cmake -S "%SRC%" -B "%BUILD%" -DBUILD_TESTING=%TESTING% || exit /b 1
)
if /i "%1"=="tests" (
  cmake --build "%BUILD%" --target basalt inkling2_spill_test inkling2_ink_test inkling2_expert_map_test inkling2_walk_test inkling2_pretok_test || exit /b 1
) else (
  cmake --build "%BUILD%" --target basalt basalt-bench basalt-probe || exit /b 1
)
echo BUILD OK
endlocal
