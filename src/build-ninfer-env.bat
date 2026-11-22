@echo off
rem Environment for building NInfer (PR #393 branch) for sm_120a with MSVC 14.44 + CUDA 13.4.3 redist + vcpkg.
rem Usage (cmd):  call build-ninfer-env.bat   then run cmake / ninja.
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.44 >nul 2>&1
set "TOOLCHAIN=C:\WORKDIR\inkling\toolchain"
set "CUDA_PATH=%TOOLCHAIN%\cuda-13.4"
set "CUDAToolkit_ROOT=%CUDA_PATH%"
set "VCPKG_ROOT=%TOOLCHAIN%\vcpkg"
set "VCPKG_DISABLE_METRICS=1"
set "VCPKG_VISUAL_STUDIO_PATH=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
set "PATH=%CUDA_PATH%\bin;%CUDA_PATH%\bin\x64;%PATH%"
set "NINFER_SRC=C:\WORKDIR\inkling\src\ninfer-all"
set "NINFER_BUILD=C:\WORKDIR\inkling\src\build-ninfer-pr393"
