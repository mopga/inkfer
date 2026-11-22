# NInfer-all: local experimental build with quantized KV for Inkling 2 (PR #393)

Built 2026-11-22 on this machine (Windows 11, Ryzen 7 7800X3D, RTX 5080). The build step did not run the GPU:
only `--help` was executed. Everything below is reproducible without system installers.

## What was built

| | |
|---|---|
| Source | `C:\WORKDIR\inkling\src\ninfer-all` (github.com/iamwavecut/ninfer-all) |
| Branch | `local/pr393-inkling` = PR #393 head `f66e789` + 6 local commits (`8a37fdd`, `88caff1`, `50d8a1e` trail points, `a9df5bf` host store eviction, `5b8595a` far keys on the tensor cores, `d0bd340` `--vision-upscale`) |
| Base of the PR | release `2026-11-21.4` (`bd8195f42e87`), the build already used in `engines\ninfer` |
| Build tree | `C:\WORKDIR\inkling\src\build-ninfer-pr393` |
| Package | `C:\WORKDIR\inkling\engines\ninfer-pr393\app` |
| GPU architectures | `120a` only (RTX 50 series). The release carries `75 80 86 89 120a`. |
| Layout | the release's `shared-runtime-v1`: small launchers + `backends\cuda13\ninfer-runtime.dll`, `ninfer_kernels.dll` |

PR #393 is open and unmerged upstream (state checked 2026-11-22). It differs from the release only in
`src/models/inkling2/*`, `src/runtime/engine/inkling2_core.cpp`, docs and tests; the artifact reader and the
converter are untouched, so the existing pack
`D:\MODELS\inkling2mini\ninfer\Inkling-2-Mini-UD-IQ4_XS-ninfer-v3.ninfer` (magic `NINFER`, version 3) loads as is.

## Result and checks

The build finished without errors: 1,258 ninja steps in about 50 minutes at 6 jobs, low priority, on a machine
that was busy with benchmarks. Package size 1.0 GiB (`ninfer-runtime.dll` 141 MB, `ninfer_kernels.dll` 346 MB,
cuBLAS 548 MB).

| Check | Result |
|---|---|
| `--help` of all six launchers with only `System32` on PATH (no toolkit, no vcpkg) | exit 0; ninfer-serve prints 602 lines, identical to the release's help |
| `ninfer-serve --version` | `ninfer-serve 2026-11-21.4+2026-11-21.4-12-g88caff1` |
| `--kv-dtype` line in the help | lists `rk8v4`, `rk4v4` (the release's help lists them too; it refuses them for Inkling at model load) |
| PR code present | the new gate message "quantizes the eight full-attention visits" is in this `ninfer-runtime.dll` and not in the release's |
| Device code | `cuobjdump -lelf`: 414 + 360 `sm_120` images in the two DLLs, no other architecture |
| GPU run | not done by the build step |

Note on `--help`: the launcher prints `help\<command>.txt` when that file exists and asks the backend only when it
does not. The files in `help\` were generated from this build with the folder temporarily empty. Do not redirect
`--help` into that folder: an empty file there makes `--help` print nothing.

## Toolchain

| Component | Version | Where |
|---|---|---|
| MSVC | 19.44.35229 (toolset 14.44.35207), VS 2022 BuildTools | `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools` (already installed) |
| CMake, Ninja | the ones bundled with BuildTools | on PATH after `vcvars64.bat` |
| CUDA toolkit | 13.4.3 redistributables, nvcc V13.4.92 | `C:\WORKDIR\inkling\toolchain\cuda-13.4` (see `READY.txt` there) |
| vcpkg | master `93ad765c7b`, manifest baseline `4bca8fd8` from the repo's `vcpkg.json` | `C:\WORKDIR\inkling\toolchain\vcpkg` |
| vcpkg packages | curl, ffmpeg 8.1.2 [avcodec, avdevice, avfilter, avformat, swresample, swscale, zlib], pkgconf, zlib (x64-windows) | `C:\WORKDIR\inkling\toolchain\vcpkg_installed\x64-windows` |

The CUDA toolkit is a merge of NVIDIA's official per-component archives
(`https://developer.download.nvidia.com/compute/cuda/redist/`, manifest `redistrib_13.4.3.json`, every archive
checked against the manifest's SHA-256): cuda_nvcc, cuda_cudart, cuda_crt, cccl, libnvvm, libnvptxcompiler,
libnvfatbin, cuda_nvml_dev, cuda_nvtx, cuda_profiler_api, cuda_cuobjdump, cuda_nvprune, libcublas. Downloads and
unpacked trees are in `toolchain\dl`. No Visual Studio CUDA integration is installed, so the generator is Ninja.

FFmpeg and curl are hard requirements of the Windows configure step (`cmake/Dependencies.cmake`), so they were
built with vcpkg (14 minutes) instead of being patched out.

## Build commands

Three small scripts in `C:\WORKDIR\inkling\src` (run from cmd or PowerShell, not Git Bash):

```
build-ninfer-env.bat         sets MSVC 14.44, CUDA_PATH, VCPKG_ROOT, source and build paths
build-ninfer-configure.bat   cmake configure
build-ninfer-build.bat [N]   ninja build with N jobs (default 6), low process priority
```

Dependencies (once):

```
C:\WORKDIR\inkling\toolchain\vcpkg-install.bat
```

Configure (what `build-ninfer-configure.bat` runs after `vcvars64.bat -vcvars_ver=14.44`):

```
cmake -S C:\WORKDIR\inkling\src\ninfer-all -B C:\WORKDIR\inkling\src\build-ninfer-pr393 -G Ninja -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_TOOLCHAIN_FILE=C:\WORKDIR\inkling\toolchain\vcpkg\scripts\buildsystems\vcpkg.cmake ^
  -DVCPKG_TARGET_TRIPLET=x64-windows -DVCPKG_HOST_TRIPLET=x64-windows ^
  -DVCPKG_INSTALLED_DIR=C:\WORKDIR\inkling\toolchain\vcpkg_installed -DVCPKG_MANIFEST_INSTALL=OFF ^
  -DCMAKE_CUDA_COMPILER=C:\WORKDIR\inkling\toolchain\cuda-13.4\bin\nvcc.exe ^
  -DCUDAToolkit_ROOT=C:\WORKDIR\inkling\toolchain\cuda-13.4 ^
  -DCMAKE_CUDA_ARCHITECTURES=120a -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl ^
  -DNINFER_WINDOWS_KERNEL_DLL=ON -DNINFER_SHARED_RUNTIME=ON -DBUILD_TESTING=OFF -DNINFER_BUILD_BENCHMARKS=ON
```

These are the flags of the upstream "Native Windows build" workflow, minus ccache, `--split-compile` and PTX
pruning, with one architecture instead of five.

Build:

```
cmake --build C:\WORKDIR\inkling\src\build-ninfer-pr393 --parallel 6 --target ninfer-serve ninfer ninfer-perplexity ninfer_bench ninfer-calibrate ninfer-ngram-profile
```

1,250 build steps (778 CUDA, 439 C++). The job count was kept at 6 and the priority low because the machine was
running model benchmarks at the same time and had about 10 GiB of free RAM.

Package:

```
bash C:\WORKDIR\inkling\src\package-ninfer-pr393.sh
```

It copies the launchers and `portable\backends\cuda13\*.dll` from the build tree, the FFmpeg/curl/zlib/pkgconf DLLs
from `vcpkg_installed`, `cublas64_13.dll`, `cublasLt64_13.dll` and the CUDA EULA from the release folder (which is
only read), and `configs\` from the source tree.

## How to rebuild after a source change

```
cd /d C:\WORKDIR\inkling\src
build-ninfer-build.bat 12
bash package-ninfer-pr393.sh
```

Stop any server started from `engines\ninfer-pr393\app` first: Windows will not overwrite a loaded DLL.
A change in `src/runtime/engine/inkling2_core.cpp` or `src/models/inkling2/executor.cpp` recompiles a few files
and relinks `ninfer-runtime.dll`.

## Local patches

All six are commits on `local/pr393-inkling`. Patches 1–4: `C:\WORKDIR\inkling\src\ninfer-local-patches.diff`
(`git diff f66e789..a9df5bf`); patches 5–6: `ninfer-local-patches-far-vision.diff` beside it (`git diff a9df5bf..d0bd340`). Patches 1 and 2 were then checked on the GPU by the context ladders (`inkling\NOTES.md`, section 6: the follow-up
turn takes 100 % from the cache at 66K to 496K tokens); patch 3 has its own check, described with it.

### 1. `8a37fdd` — a prompt resuming from a stored host image keeps that image while the sequence retires

`Inkling2Core::begin()` stores what the sequence held (a turn-closure image and an endpoint image) before it loads
the stored image the new prompt resumes from. `make_host_room()` freed pages by dropping the least recently used
image, which at long contexts — a host budget of two images — is the image about to be loaded. The prefix hit was
lost and the prompt prefilled from its start. The image is now marked `held` for the duration of `retire()` and
`make_host_room()` skips held images; if nothing else can be dropped, the sequence's own image is not stored.
A hit therefore needs two images in the budget instead of three.

```diff
@@ struct Stored
         std::uint64_t bytes     = 0;
         std::uint64_t last_used = 0;
+        // A prompt is about to resume from it: make_host_room leaves it, whatever its age.
+        bool held = false;
     };
@@ bool make_host_room(std::uint64_t bytes)
         while (free_pages.size() < pages) {
-            const auto oldest = std::min_element(stored.begin(), stored.end(), [](const Stored& a, const Stored& b) {
-                return a.last_used < b.last_used;
-            });
+            auto oldest = stored.end();
+            for (auto image = stored.begin(); image != stored.end(); ++image) {
+                if (image->held) { continue; }
+                if (oldest == stored.end() || image->last_used < oldest->last_used) { oldest = image; }
+            }
             if (oldest == stored.end()) { return false; }
@@ void begin(...)
                 const PrefixReusePath path   = stored[image].path;
-                retire(s);
-                // Retiring may have dropped or moved the image: find it again.
+                stored[image].held = true;
+                const auto release = [this] {
+                    for (Stored& kept : stored) { kept.held = false; }
+                };
+                try {
+                    retire(s);
+                } catch (...) {
+                    release();
+                    throw;
+                }
+                release();
+                // Retiring may have moved the image: find it again.
                 if (const auto index = stored_index(tokens, media)) {
```

Known cost: when the budget holds exactly two images, `retire()` may copy the turn-closure image to the host and
then drop it again for the endpoint image (one wasted device-to-host copy). The flag workaround without the patch
is a larger `--host-cache-mib`.

### 2. `88caff1` — a host image of quantized KV planes is sized by the plane bytes

Found while reading the PR: `Executor::image_bytes()` still sized the eight full-attention visits at F16
(32 KiB a token) for every format, while `copy_image()` copies only the codes and scales of `rk8v4` / `rk4v4`.
The store reserved F16-sized pages for every image and charged them to the host budget: about 3.7 GiB for a
120K-token image whose data is about 1.0 GiB under `rk4v4`. `image_bytes()` now counts exactly what
`copy_image()` copies, so the same host budget holds 3.6 times (rk4v4) or 2.5 times (rk8v4) more prefix images.
`bf16` / `fp16` are unchanged.

```diff
@@ std::uint64_t image_bytes(std::uint32_t position) const
         for (const VisitPlan* p : cache_plans()) {
-            if (p->full) {
+            if (p->full && p->k_codes.p != nullptr) {
+                // Quantized planes: what copy_image copies, a position's codes and scales a head.
+                for (const KvArea* area : {&p->k_codes, &p->k_scales, &p->v_codes, &p->v_scales}) {
+                    bytes += std::uint64_t(kv_heads) * position * area->position_bytes;
+                }
+            } else if (p->full) {
                 bytes += 2 * std::uint64_t(kv_heads) * position * kD * sizeof(__half);
             } else {
```

### 3. `50d8a1e` — trail points: a prompt that parts from the cached tokens mid-way resumes from a saved ring state

Packaged separately: `C:\WORKDIR\inkling\engines\ninfer-trail\app` (a copy of `ninfer-pr393\app` with the
rebuilt launchers, `ninfer-runtime.dll` and `ninfer_kernels.dll`; `--version` prints `...-g88caff1-dirty` because the
build ran before the commit). `engines\ninfer-pr393\app` stays as the build without this patch.

The problem: the Inkling core resumes a sequence only from a state that is a *whole prefix* of the new prompt (the
live end, the anchor at the end of the last prompt, a stored host image). When the client renders an earlier message
differently (oh-my-pi after `omp -c` session resume: 192K tokens computed again, 4 min 9 s), edits a turn, or a side
request changes what the sequence holds, nothing matches and the whole prompt is prefilled from token 0.

The fix: the full-attention rows on the device are only ever appended to, so going back to an earlier position of
tokens the sequence still holds needs only the *rings* at that position (sliding-window caches, the DSpark drafter's
contexts, the convolution histories: 213 MiB for Inkling 2 Mini with the drafter). Every `NINFER_TRAIL_GRID` prompt
tokens (default `max(8192, max_context/32)`) the rings are copied to pageable host memory, up to `NINFER_TRAIL_MIB`
(default 4096 = 19 points). At admission the core looks for the furthest point whose tokens the prompt starts with
(128-bit hash of the token prefix) below a position up to which the live sequence, or a stored image, holds exactly the
prompt's tokens; if that saves more than 1,024 tokens over the whole-prefix options it loads the rings there
(`Executor::load_rings`) and prefills the rest. The request log shows such a hit as `long anchor`.
With the trail full, the oldest point of another conversation goes first, then the point of this conversation whose
neighbours lie closest. `NINFER_TRAIL_GRID=0` turns the trail off (stock behaviour). Not used with pictures in the
prompt, with a DSpine drafter, or when the engine runs without prefix reuse.

Code: `src/models/inkling2/executor.{h,cpp}` (`trail_supported`, `rings_bytes`, `save_rings`, `load_rings`),
`src/runtime/engine/inkling2_core.cpp` (`TrailPoint`, `take_trail_point` after a prefill step, `pick_trail` /
`resume_from_trail` in `begin()`).

Check on the GPU (`agent-tests\trail_check.sh`, `tools\trailtest.py`, 44K-token conversation, 262,144 window, rk8v4,
temperature 0, reasoning and answer compared byte for byte with a run of the same build with `NINFER_TRAIL_GRID=0`):

| Request after the cold one | Cached with the trail | Time | Without the trail | Same output |
|---|---:|---:|---:|---|
| last user message changed | 40,960 of 44,262 | 3.9 s | 0 cached, 22.5 s | yes |
| the same after a 2K-token side request (state taken from the host store) | 40,960 of 44,251 | 3.6 s | 0 cached, 21.9 s | yes |
| an assistant message in the middle changed | 16,384 of 44,248 | 17.4 s | 0 cached, 22.6 s | yes |

Cost: one device-to-host copy of 213 MiB per 8,192 prompt tokens (about 0.1 s) and up to 4 GiB of RAM.

### 4. `a9df5bf` — the host store drops superseded images of a conversation before the last image of another

Found in the big agent task (`inkling\NOTES.md`, section 9): the main conversation (179K tokens) and a subagent
(125K) took turns with 99 % from the cache for a hundred requests, then the subagent made five requests in a row
while the main one waited for a 60-second tool, and the main conversation's next prompt (185,156 tokens) was read
again from the start: 4 min 5 s. The engine's request log (`--request-log-jsonl`, `throughput` events:
`context_cache.state_transfers`, `occupancy.host_kv_bytes`) shows why. A conversation leaves an image in the host
store at every request (3.06 GiB for the main one, 2.2 GiB for the subagent with rk8v4), the store was full
(15.5 of 16 GiB), and `make_host_room` dropped the least recently used images — which, after a few requests of
one conversation in a row, are all the images of the other, while the active conversation keeps several
generations of its own. Neither `find_stored` nor the trail (which needs a stored or live state of the tokens)
could help after that.

`make_host_room` now drops first the images that are *superseded*: a newer image, or the one coming in, shares
at least 7/8 of the image's tokens and is at most twice as long (the same conversation further on; a short
shared system prompt is not superseded by the conversations that begin with it). Only when there is none does
the least recently used image go, and the engine then prints
`Inkling 2: host store full, dropped the last state of a conversation (N tokens, M MiB); a larger --host-cache-mib keeps more of them`.
A prompt that needs a dropped older image of a conversation resumes from the newer image and a trail point
(patch 3), so the two patches belong together. Numerics are untouched: the patch only changes which stored
state survives.

Check on the GPU (`agent-tests\alt_check.sh`, `tools\alttest.py`): two conversations of 40K and 30K tokens,
the store cut to 3,072 MiB so that it fills as the 16 GiB one does at 180K + 125K; order A, B, A, B, A, then
seven requests of B in a row, then A.

| Build | A after one request of B | A after seven requests of B in a row |
|---|---:|---:|
| before the patch (`50d8a1e`) | 32,768 of 40,203 cached, 5.2 s | **0 of 40,690 cached, 18.7 s** |
| with the patch | 32,768 of 40,203 cached, 5.2 s | 32,768 of 40,690 cached, 5.6 s |

(In this test the answers are cut at 60 tokens and not sent back, so A resumes from a trail point 8K tokens
back rather than from its end; in agent sessions the same request takes 99 % from the cache.) The trail check
of patch 3 was repeated on this build with the same result (identical outputs).

What the patch does not cover: three or more large conversations whose *last* images together exceed
`--host-cache-mib` (the message above appears), and the pool of trail points, which is still shared by all
conversations (19 points by default, `NINFER_TRAIL_MIB`).

To build without a patch: `git -C ninfer-all revert <hash>` (or check out `pr393`) and rebuild.

Patches 5 and 6 were committed after the measurements, from the same working tree the deployed engine was built
from; only documentation and the qualification program's place changed at commit time (it now lives in the
repository as `tools/attn_bench.cu`; rebuilt from there, `16384 --check 1 --kv rk4v4` gives the same error against
the FP64 oracle for the far and the scalar route, max abs 1.2e-4). Neither commit ran the repository's test suite
or `compute-sanitizer`, and neither adds a test under `tests/` — the commit messages say so.

Rebuilt from the clean tree on 2026-11-22 20:00 (`build-ninfer-build.bat 12`, no reconfigure, log `build-clean.log`):
`--version` and `/health` now print `2026-11-21.4+2026-11-21.4-16-gd0bd340`, without `-dirty`. Only the objects that
carry the build id were recompiled (`device_profile_build_id.cpp`, `runtime_dispatch.cpp`, the launchers' and the
apps' `main.cpp`), plus `qwen3_5/model.cpp` and `load.cpp` from unchanged sources (see the build trap below), so
the six launchers and `ninfer-runtime.dll` were relinked and copied into `engines\ninfer-trail\app`;
`ninfer_kernels.dll` had nothing to rebuild and is the 15:46 file, byte-identical to the deployed one.
`BUILD-INFO.json` there names `d0bd340`. Checked after the restart on `balanced`: `tools\apicheck.py` 13 of 13,
`tools\vischeck.py` 11 of 11 in four runs out of five (one run 9 of 11: a correct answer the grader's pattern
missed — pattern widened since — and a repeated tool call instead of the answer; sampling is not pinned), one
turn through omp-web in `scratch-smoke` (`read` + `bash`, correct). The context ladders and the agent runs were
not repeated: they were measured on the previous binaries, built from the same sources.

### 5. `5b8595a` — far keys of a quantized full visit on the tensor cores (`kernels.cu`, `kernels.h`, `executor.cpp`)

Why: with rk4v4/rk8v4 the scalar attention over the full visits' long KV was the main cost at depth — the prompt
read fell from 1,684 tok/s at 66k to 332 at 496k and decode to 21–31 tok/s. Keys beyond every row's
relative-bias extent (the far keys) are seen by all rows of a group of eight without bias, so they form an ordinary
dense QK^T / PV product that fits MMA tiles: the new route dequantizes the far planes per tile into shared memory and
multiplies on the tensor cores, beside the existing scalar near-key splits, merging the partial softmaxes as the
split-K merge already does.

- Prompt chunk: 1 far slot per row group. Decode / verify: 21 near splits + 107 far slots, only once the deepest
  row has >= 6,144 keys (`kDecodeFarMinKeys`); the graph key carries a bit for it so both graph shapes are cached.
- sm_80+ and quantized planes only; F16 planes and cards below sm_80 keep the scalar route.
  `NINFER_INKLING2_FAR_SPLITS=near,far[,chunk]` overrides the slot counts for A/B runs (`N,0` = scalar everywhere).
- Qualification: `tools/attn_bench.cu` (built as `inkling\tools\attnbench\attn_bench.exe`) `--check 1` against the FP64 oracle of the whole attention, rk4v4 and rk8v4,
  depths 1k–508k, chunk and decode shapes; a context within the extent computes exactly as the scalar route.
  Greedy determinism set (`bench/detcheck.py`, 7 prompts below the threshold) identical to the build without the patch.
- Op results (RTX 5080, rk4v4, per visit): 2,048-row chunk 1.21x after 1k keys, 2.18x after 4k, 4.5x after 32k,
  5.6x after 260k (744 -> 134 ms) and 506k (1,445 -> 256 ms). Decode row: 0.31–0.42x at 1k keys (slower — hence the
  threshold), 0.95–1.10x at 6k, 1.19–1.39x at 8k, 1.9–2.3x at 16k, 5.8x at 508k (4.10 -> 0.71 ms). A second chunk far
  slot was 1–2 % faster at depth and 5 % slower at 3k, so one is kept. Logs: `inkling\logs\attnbench-*.txt`.
- End to end: `inkling\NOTES.md` (ladders `far2-512k-rk4v4`, `chunk-512k-vis`) and `README.md` section 3.

### 6. `d0bd340` — `--vision-upscale F` (frontend `processor.cpp`, options plumbing, `docs/cli.md`, `docs/serving.md`)

Inkling's tower takes an image at its own resolution in 40-pixel cells; screen text whose glyphs are smaller than a
20-pixel patch was misread. `--vision-upscale F` (1–4, default 1, Inkling 2 only — other families refuse a value
other than 1) enlarges each side by F before the existing budget bound. At 1.5 every line of the dense UI/code
screenshots was read where 1.0 missed some; 1920x1080 -> 3,013 positions instead of 1,357. Help text is static in
`app\help\ninfer-serve.txt` and `app\help\ninfer.txt` and was updated by hand.

### Build trap: header dependencies on a Russian MSVC

`cl /showIncludes` prints its prefix in Russian (CP866) on this machine (no English language pack; `VSLANG=1033`
and `chcp` do not help), while CMake wrote `msvc_deps_prefix` into `build-ninfer-pr393\CMakeFiles\rules.ninja` in
UTF-8. Ninja then recorded no header dependencies (`ninja -t deps <obj>` showed `#deps 0`), and a header edit
(a new options field) left stale objects: a struct-layout mismatch made the engine read garbage options
("Inkling 2 takes no rotation scaling"). Fixed by writing the prefix in `rules.ninja` as CP866 bytes (backup
`rules.ninja.bak-ru`) and a clean rebuild. A CMake reconfigure regenerates `rules.ninja` and brings the trap back:
check `ninja -t deps` after one.

The same trap has a second door (seen 2026-11-22): `cl` writes the prefix in the *console* code page. A build
started from a UTF-8 console (65001 — the agent's PowerShell tool is one) printed it in UTF-8, the CP866 prefix did
not match, and the 11 objects compiled in that run were recorded with `#deps 0` (two more, `qwen3_5/model.cpp` and
`load.cpp`, already were). `build-ninfer-build.bat` now runs `chcp 866` before the build; the 13 objects were
deleted and recompiled, and `ninja -t deps | findstr /c:"#deps 0"` is empty again. Run ninja directly only after
`chcp 866`.

## New flags and what the PR documents about them

`--kv-dtype rk8v4` and `--kv-dtype rk4v4` are accepted for Inkling 2 (the release refuses everything except
`bf16` / `fp16`). They quantize the keys and values of the eight full-attention visits only; the 40 windowed
visits' 1,024-token rings, the convolution histories and the DSpark drafter's caches stay F16. The other
quantized formats (`int8`, `fp8`, `nvfp4`, `k8v4`, `rk4v4-e8`, `rk2v4-e8`, `bf16p`) are still refused.

| `--kv-dtype` | Keys | Values | KV per token (full visits) | 131,072 tokens | 262,144 | 524,288 |
|---|---|---|---:|---:|---:|---:|
| `bf16` / `fp16` (default) | F16 | F16 | 32 KiB | 4.0 GiB | 8.0 GiB | 16.0 GiB |
| `rk8v4` | int8, groups of 64 | int4, groups of 32 | 12.75 KiB | 1.6 GiB | 3.2 GiB | 6.4 GiB |
| `rk4v4` | Lloyd-Max 4-bit | int4, groups of 32 | 8.75 KiB | 1.1 GiB | 2.2 GiB | 4.4 GiB |

The per-token figures are from `docs/cli.md` on the branch and match the code (`kv_format.h`: 8 visits x 8 KV
heads x (132 + 72) or (68 + 72) bytes); the context columns are that figure times the context, one sequence,
without the rings and histories (a constant of a few hundred MiB).

Quality, from the PR description (author's measurements on an RTX 5090, drafting off, not reproduced here):

| Metric | rk8v4 | rk4v4 |
|---|---|---|
| Mean KL of next-token distributions vs bf16 | 0.017 nats | 0.015 nats |
| Median KL | 0 | 0 |
| Corpus NLL over 13,370 tokens (bf16: 1.61399) | 1.61046 (-0.22 %) | 1.61532 (+0.08 %) |
| Throughput vs bf16 | within +-2.4 % | within +-2.4 % |

`docs/inkling2.md` itself gives no perplexity table for these formats. Restrictions stated in the PR and docs:
no F16 mirror is kept (bf16 stays the exactness reference); keys are not rotated before coding; the drafter is
untouched; nothing is said about limits on prefill chunk or prefix cache, and the code keeps both paths
(`image_bytes` / `copy_image` handle the planes, see patch 2). The PR author tested on Linux with CUDA 13.1;
this Windows build is the first of this branch as far as the repository shows.

## Launching

Same command line as the release, from the new folder, plus the KV format:

```
cd /d C:\WORKDIR\inkling\engines\ninfer-pr393\app
ninfer-serve.exe D:\MODELS\inkling2mini\ninfer\Inkling-2-Mini-UD-IQ4_XS-ninfer-v3.ninfer --model-id inkling --host 127.0.0.1 --port 8082 --cuda-backend 13 --kv-dtype rk4v4 <the other flags as before>
```

Differences from the release: the folder; `--kv-dtype rk4v4` or `rk8v4` is accepted; the binary contains only
the `120a` kernels, so it runs on the RTX 5080 and not on older cards.
