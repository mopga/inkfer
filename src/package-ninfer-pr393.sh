#!/usr/bin/env bash
# Package the local NInfer build (PR #393 + local patches, sm_120a only) in the release's shared-runtime layout.
#   bash package-ninfer-pr393.sh
# Source: src/build-ninfer-pr393/portable.  Target: engines/ninfer-pr393/app.  The release folder is only read.
set -eu
ROOT=/c/WORKDIR/inkling
BUILD=$ROOT/src/build-ninfer-pr393
SRC=$ROOT/src/ninfer-all
REL="$ROOT/engines/ninfer/app/ninfer-windows-x64-cuda13.4-sm75-80-86-89-120a-bd8195f42e87"
VCPKG_BIN=$ROOT/toolchain/vcpkg_installed/x64-windows/bin
OUT=$ROOT/engines/ninfer-pr393/app
BACK=$OUT/backends/cuda13

mkdir -p "$BACK" "$OUT/help"
for exe in ninfer-serve ninfer ninfer-perplexity ninfer_bench ninfer-calibrate ninfer-ngram-profile; do
  cp -f "$BUILD/portable/$exe.exe" "$OUT/"
done
# The backend built here: the runtime, the kernel DLL and whatever the build staged beside them.
cp -f "$BUILD"/portable/backends/cuda13/*.dll "$BACK/"
cp -f "$BUILD/portable/backends/cuda13/architectures.txt" "$BACK/"
# FFmpeg, curl, zlib, pkgconf: the DLLs this build was linked against (same vcpkg baseline as the release).
for dll in "$VCPKG_BIN"/*.dll; do
  [ -f "$BACK/$(basename "$dll")" ] || cp -f "$dll" "$BACK/"
done
# cuBLAS runtime and the CUDA EULA: taken from the release as it ships them (CUDA 13.4.3, same files as the toolkit's).
cp -f "$REL/backends/cuda13/cublas64_13.dll" "$REL/backends/cuda13/cublasLt64_13.dll" \
      "$REL/backends/cuda13/NVIDIA-CUDA-EULA.txt" "$BACK/"
cp -f "$REL"/LICENSE* "$OUT/"
rm -rf "$OUT/configs"; cp -r "$SRC/configs" "$OUT/configs"
REV=$(git -C "$SRC" rev-parse HEAD)
cat > "$OUT/BUILD-INFO.json" <<EOF
{
  "local_experimental_build": true,
  "source": "iamwavecut/ninfer-all PR #393 (head f66e789) on release 2026-11-21.4 (bd8195f42e87) + 2 local patches",
  "source_revision": "$REV",
  "branch": "local/pr393-inkling",
  "cuda": "13.4",
  "cuda_compiler": "nvcc V13.4.92 (CUDA 13.4.3 redistributables), host MSVC 19.44 (toolset 14.44)",
  "platform": "windows-x64",
  "layout": "shared-runtime-v1",
  "kernel_dll": true,
  "cuda_archive_ptx_pruned": false,
  "architectures": ["120a"],
  "validation": "Launchers pass --help. GPU inference was not tested by the build step."
}
EOF
ls -la "$OUT" "$BACK" | awk '{print $5, $9}'
