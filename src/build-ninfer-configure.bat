@echo off
rem Configure NInfer: Release, Ninja, sm_120a only, the release's shared-runtime layout (launchers + backends\cuda13).
call "%~dp0build-ninfer-env.bat"
cmake -S "%NINFER_SRC%" -B "%NINFER_BUILD%" -G Ninja -DCMAKE_BUILD_TYPE=Release ^
  "-DCMAKE_TOOLCHAIN_FILE=%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake" ^
  -DVCPKG_TARGET_TRIPLET=x64-windows -DVCPKG_HOST_TRIPLET=x64-windows ^
  "-DVCPKG_INSTALLED_DIR=%TOOLCHAIN%\vcpkg_installed" -DVCPKG_MANIFEST_INSTALL=OFF ^
  "-DCMAKE_CUDA_COMPILER=%CUDA_PATH%\bin\nvcc.exe" "-DCUDAToolkit_ROOT=%CUDA_PATH%" ^
  -DCMAKE_CUDA_ARCHITECTURES=120a -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl ^
  -DNINFER_WINDOWS_KERNEL_DLL=ON -DNINFER_SHARED_RUNTIME=ON ^
  -DBUILD_TESTING=OFF -DNINFER_BUILD_BENCHMARKS=ON
echo configure_exit=%ERRORLEVEL%
