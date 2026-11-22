# Third-party code and licences

This repository contains no third-party source trees and no binaries. It contains patches against two engines;
a patch is a derivative work of the code it changes and is distributed under that code's licence.

## NInfer-all

- Upstream: https://github.com/iamwavecut/ninfer-all — Apache License 2.0.
- Files here: `src/patches/ninfer/*.patch`, `src/ninfer-local-patches.diff`,
  `src/ninfer-local-patches-far-vision.diff`, and `tools/attnbench/`: `attn_bench.cu` (a copy of the file patch 5
  adds as `tools/attn_bench.cu`) and `far_attention.cuh` (a standalone prototype of the same kernel, derived from
  NInfer's Inkling 2 kernels).
- Base: pull request #393 head `f66e789eda919a15191e34828a9b73c41fc231a2` (2026-11-21), itself based on release
  `2026-11-21.4`.
- These files are licensed under the Apache License 2.0: https://www.apache.org/licenses/LICENSE-2.0
- Changes: six commits, described in `src/BUILD-NOTES-ninfer.md` (host-store eviction fixes, trail points,
  far-key attention on the tensor cores for quantized KV, `--vision-upscale`).

## Basalt

- Upstream: https://github.com/jesdga95/basalt — MIT License, Copyright (c) 2026 Jesús Gulfo.
- Files here: `src/patches/basalt/*.patch`, `src/basalt-ink-patches.diff`.
- Base: main `97e2e7110578aeb7f8708dab176b910589b35047`.
- These files are licensed under the MIT License, the same terms as upstream.
- Changes: four commits, described in `src/BUILD-NOTES-basalt.md` (card reserve and elastic keep, routing profile
  output, adaptive on-card expert set, prompt staging lent to decode).

## Not included

Model weights, the chat template shipped with the model, engine binaries, NVIDIA CUDA redistributables, vcpkg
packages, and the source code of the application used as the agents' test project are not part of this repository.
