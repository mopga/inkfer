# Basalt: experimental Windows build with an adaptive on-card expert set (Inkling 2 Mini)

Source: `C:\distrib\strata-tuning\inkling\src\basalt`, branch `ink-adaptive-experts` on top of main `97e2e71` (0.5.6).
Nothing was pushed. The whole change as one diff: `src\basalt-ink-patches.diff`.

| Commit | What | Switch |
|---|---|---|
| `907603a` | plan/elastic reserves from the environment | `BASALT_INK_CARD_SPARE_MIB`, `_ELASTIC_KEEP_MIB`, `_SESSION_FIXED_MIB`, `_SPILL_MARGIN_MIB` |
| `4758f70` | decode's routing tally written as an expert profile | `BASALT_INK_PROFILE_OUT`, `_PROFILE_EVERY` |
| `6a0d172` | adaptive on-card set (re-rank, swaps, top-up) | `BASALT_INK_ADAPT` and friends |
| `97dce5e` | decode uses the idle prompt staging for RAM experts | `BASALT_INK_STAGING` |

With no `BASALT_INK_*` variable set, the patched engine takes the same code paths as main.

**Status: built and CPU-tested only. No GPU run was made by the build fork** (no server start, no benchmark). Everything
under "What to check on the GPU" below is unverified.

## Trees

| Path | Content |
|---|---|
| `engines\basalt-src\app\` | runnable tree: copy of the 0.5.5 release layout (`.venv`, `serve`, `tools`, `data`, `docs`), main's `serve\service.py`, **patched** `bin\basalt.exe`, `basalt-bench.exe`, `basalt-probe.exe`, cuBLAS 13.8 DLLs beside them |
| `engines\basalt-src\baseline\` | **unpatched** main `97e2e71` `bin\basalt.exe` (+ cuBLAS DLLs); `serve`, `tools`, `data`, `docs`, `.venv` are junctions to `app\` |
| `engines\basalt\app\basalt-v0.5.5-windows-x64\` | the release, untouched |

Launch is the same as the release, only the path differs:

```
C:\distrib\strata-tuning\inkling\engines\basalt-src\app\bin\basalt.exe serve PACK.basalt --port 8081 ...
```

`basalt.exe serve` starts `<tree>\.venv\Scripts\python.exe -m serve.server --engine-bin <itself>`; the environment is
inherited, so set the `BASALT_INK_*` variables in the shell (or the launching script) before the start. For an A/B run
against unpatched main use `engines\basalt-src\baseline\bin\basalt.exe` with the same arguments.

## Toolchain

- MSVC 14.44 (VS 2022 BuildTools, `vcvars64 -vcvars_ver=14.44`), the CMake and Ninja that ship with BuildTools.
- CUDA 13.4 toolkit unpacked in `toolchain\cuda-13.4` (no installer). The unpacked toolkit had no cuBLAS; libcublas
  13.8.0.4 (the redist archive of the same 13.4.3 manifest, SHA-256 verified) was merged in, see
  `toolchain\cuda-13.4\ADDED-libcublas.txt`. `cublas64_13.dll` and `cublasLt64_13.dll` are copied beside `basalt.exe`.
- Configure flags: `-G Ninja -DCMAKE_BUILD_TYPE=Release -DBASALT_MODELS=inkling2 -DCUDAToolkit_ROOT=<toolkit>
  -DCMAKE_CUDA_COMPILER=<toolkit>\bin\nvcc.exe -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl`, build dir
  `src\basalt\build-ink`.

## Build, test, install

```
C:\distrib\strata-tuning\inkling\src\build-basalt.cmd            rem engine programs -> src\basalt\bin (2-3 min incremental)
C:\distrib\strata-tuning\inkling\src\build-basalt.cmd tests      rem + the CPU tests of the Inkling model
C:\distrib\strata-tuning\inkling\src\install-basalt.cmd          rem copy basalt*.exe -> engines\basalt-src\app\bin (stop the engine first)
```

From Git Bash: `cmd.exe //c "C:\distrib\strata-tuning\inkling\src\build-basalt.cmd tests" > build.log 2>&1 < /dev/null`.

CPU tests (in `src\basalt\build-ink`, all pass): `inkling2_ink_test` (new: re-ranking, swap planning, hysteresis, sizes,
bounds), `inkling2_spill_test`, `inkling2_expert_map_test`, `inkling2_walk_test`, `inkling2_pretok_test`.

To rebuild the unpatched baseline: `git -C src\basalt checkout 97e2e71`, `build-basalt.cmd`, copy `src\basalt\bin\basalt.exe`
to `engines\basalt-src\baseline\bin\`, then `git checkout ink-adaptive-experts`.

## The switches

All are read once at first use (`src/models/inkling2/ink.hpp` documents them too). MIB values are integers.

### Reserves (commit 907603a)

| Variable | Default | Meaning |
|---|---|---|
| `BASALT_INK_CARD_SPARE_MIB` | 1024 | card memory the context plan keeps free; also what the elastic spill's first copy-back leaves |
| `BASALT_INK_ELASTIC_KEEP_MIB` | 512 | free bytes an elastic KV grow leaves / a restart's copy-back leaves |
| `BASALT_INK_SESSION_FIXED_MIB` | 768 | what the plan counts for a session beside its caches |
| `BASALT_INK_SPILL_MARGIN_MIB` | 128 | plan margin when experts leave the card |

Log (only when one differs): `ink: reserves from the environment: card spare N MiB (1024), elastic keep ..., session fixed ..., spill margin ...`.
These change the *plan*: a smaller spare lets a larger context load or keeps more experts on the card, at the price of
less headroom. First value to try: `CARD_SPARE_MIB=512`. For 524288 + fp8 + drafter (refused by main, "1252 MiB short"):
`CARD_SPARE_MIB=256 SPILL_MARGIN_MIB=0 SESSION_FIXED_MIB=400` - risky, watch for WDDM paging (shared GPU memory rising,
speed collapsing) rather than a clean error.

### Routing profile (commit 4758f70)

| Variable | Default | Meaning |
|---|---|---|
| `BASALT_INK_PROFILE_OUT` | unset | file the session writes decode's routing counts to, as an expert profile (most routed first) |
| `BASALT_INK_PROFILE_EVERY` | 4096 | also rewrite it every N decoded tokens (0: only when the session ends) |

Written atomically (`.tmp` + rename); ties keep the order of the profile the run started from. Load it with
`--expert-profile FILE` in the next run. Log: `ink: routing profile written to FILE (N routed lookups...)` (first write
and at the end); `ink: the routing profile FILE was not written: ...` once on failure. One session only. The tally
counts decode lookups (and prompts of 16 tokens or fewer), not the prompt path.

### Adaptive on-card set (commit 6a0d172)

Needs a spill (experts in RAM at the asked context) and one session.

| Variable | Default | Meaning |
|---|---|---|
| `BASALT_INK_ADAPT` | 0 | 1 switches the rounds on |
| `BASALT_INK_ADAPT_INTERVAL` | 64 | decoded tokens between rounds (a round also runs at the first decode window after every prompt) |
| `BASALT_INK_ADAPT_SWAPS` | 48 | max card buffers that change hands per round (48 x 4.8 MiB = 230 MiB of upload at most) |
| `BASALT_INK_ADAPT_DECAY` | 0.95 | every score is multiplied by this each round |
| `BASALT_INK_ADAPT_RATIO` / `_FLOOR` | 1.25 / 2 | a swap needs `hot > cold * ratio + floor` (hysteresis against churn) |
| `BASALT_INK_ADAPT_RERANK` | 1 | 0: keep the profile's order, no swaps (top-up/staging still run) |
| `BASALT_INK_DECODE_KEEP_MIB` | 0 | top-up: a round also makes new card copies of the hottest RAM experts while this much card memory stays free (0: off) |
| `BASALT_INK_ADAPT_FILLS` | 256 | max new copies (top-up + staging) per round |
| `BASALT_INK_PROMPT_KEEP_MIB` | 0 | before a prompt, top-up copies go back only until this much is free (0: all top-up copies go) |
| `BASALT_INK_ADAPT_LOG` | 0 | 1: a log line for every round (default: only rounds that moved something) |

A round (session thread, start of a decode window): read the tally, add the lookups since the last round to each RAM
expert's score, sort the RAM experts by score (stable; this order is then also what the engine's own elastic
copy-back/demotion follows), top up, plan swaps (`ink::plan_swaps`: hottest expert without a buffer takes the buffer of
the coldest holder that fits), execute, decay.

Without `DECODE_KEEP_MIB` the adaptation only re-distributes the copies the elastic spill still has. At deep fill
(e.g. 262144 fp8 at ~250k) there are almost none left while ~2.3 GiB sit free during decode, so **use the top-up
and/or the staging there** - re-ranking alone will show `0 copies`.

Log lines:

- at load: `ink: adaptive on-card set: a round every 64 decoded tokens, the order follows the scores, at most 48 swaps a round (hot > cold x 1.25 + 2.0), scores x 0.950 a round; top-up: on (keeps 1024 MiB free, ...); the prompt staging during decode: used`
  (or `ink: BASALT_INK_ADAPT is set, but every expert is on the card ...` / `... works with one session; off with N`)
- per round: `ink: round N: promoted P (S swaps, F new copies, T into the staging), demoted S; on the card X of 3072 experts (K kept, C copies, T in the staging); L lookups since the last round, H% on the card; M MiB free`
- before a prompt-path read: `ink: before a prompt: demoted D (T out of the staging, C top-up copies, M MiB freed); on the card X of 3072 experts; M MiB free`
- on a failed CUDA call in a round: `ink: a round's copy failed; the set stays as it is from here on` (rounds stop, the engine goes on)
- at session end: `ink: R rounds, S swaps, F new copies in the session; ...` then the engine's own
  `experts: N decode lookups, H% on the card, ...`. With ADAPT on, this figure (and the server's expert counters) is
  exact: each lookup is counted against the placement that served it. Without ADAPT the engine applies the *final*
  placement to the whole tally, as main does - compare like with like.

### Prompt staging for decode (commit 97dce5e)

`BASALT_INK_STAGING=1` (needs `BASALT_INK_ADAPT=1`). The spilled layers share one staging allocation (0.6-1.0 GiB at
deep contexts) that only the prompt path uses. With the switch, a round copies the hottest RAM experts without a card
buffer into it (flat slots, no new card memory); before the prompt path reads, their table entries go back to RAM (8
bytes each, no data moves), and the first decode window afterwards fills the slots again (about the staging's size of
upload once per request). Log at first use: `ink: the prompt staging holds RAM experts during decode: N slots of 4.80 MiB (G GiB)`.

### Suggested first runs

1. Baseline numbers with `baseline\bin\basalt.exe`, then the patched exe with no variables (must match).
2. `BASALT_INK_ADAPT=1 BASALT_INK_STAGING=1 BASALT_INK_ADAPT_LOG=1` - no extra card memory at all.
3. Add `BASALT_INK_DECODE_KEEP_MIB=1024`, then 768, then 512; stop lowering when speed drops or shared GPU memory grows.
4. `BASALT_INK_PROFILE_OUT=C:\...\conv.profile` in a typical session, then `--expert-profile` it for the next start.
5. Separately: `BASALT_INK_CARD_SPARE_MIB=512`.

## Why the output cannot change, and why nothing races

- Decode reads every expert through the per-layer device blob table (`blob_table[e]` = own slot, mapped RAM address, or
  a card buffer). The products depend on the blob's bytes, not on where they are read. A card buffer or staging slot is
  filled with exactly the `expert_stride` bytes of the pinned RAM blob before its table entry is pointed at it.
- Table entries and buffers change only inside `InklingModel::ink_round`, `ink_release_staging`, `promote`, `demote`.
  All are called from the session thread at the entry of `decode_window` / `prefill_chunk` / `prefill_embeddings`,
  i.e. between two windows: the previous window ended with a wait on the session stream, the two side streams
  (read-ahead, prefetch) fork from and join that stream inside each window, and only one session exists. No kernel is
  in flight.
- The round issues every write with `cudaMemcpyAsync` on the session stream from stable host storage, in the order
  "old entry -> RAM address, blob -> buffer, new entry -> buffer", and waits for the stream before returning; the
  window's kernels are launched afterwards on the same stream. No reliance on default-stream semantics.
- CUDA graphs capture the tables' addresses, not their entries; kernels read the entries when they execute.
- The read-ahead keeps blob *addresses* from the previous window (prefetch only, the data is discarded). Swaps reuse
  buffers in place and staging addresses stay mapped; whenever a buffer is freed (demotion, a failed round) the
  recorded counts are zeroed so no freed address is touched.
- The prompt path never reads the blob table: per chunk it copies every RAM expert of a layer into that layer's
  staging and zeroes the guard, so it overwrites every staging byte it reads. Staged experts are released (and top-up
  copies demoted) before any prompt-path read; `stage_from_card` counts real copies only, as before. Decode reads no
  byte of the staging except staged slots (kept slots lie in a range's own backing).
- Prompt staging and elastic KV growth: top-up copies are tracked in bytes and returned before a prompt, so the
  prompt sees the free memory it would have without the top-up; KV growth demotes the coldest copies exactly as main
  does (now in score order), and the next round tops up again only above `DECODE_KEEP_MIB`.

## What to check on the GPU (unverified)

1. Bit-identity: same prompt, greedy/fixed seed, baseline exe vs patched exe without variables vs patched with
   ADAPT (+STAGING, +DECODE_KEEP): token streams must be identical. Include a multi-turn chat (prompt -> decode ->
   prompt), a prompt longer than one chunk, a prompt of 16 tokens or fewer, and an image request if vision is used.
2. Hit rate: `H% on the card` per round and the final `experts:` line, against the 56-61 % seen at 250k fill.
3. Speed: decode tok/s and TTFT of the first decode window after each prompt (the refill: estimated 0.1-0.25 s).
4. Memory: free VRAM in the round lines stays near `DECODE_KEEP_MIB`; no growth of shared GPU memory (WDDM pages
   silently instead of failing `cudaMalloc`); the prompt after a topped-up decode is not slower than baseline.
5. Stability over a long session at real fill (KV growth while topped up; the `before a prompt` lines).

## Risks

- WDDM overcommit: a too small `DECODE_KEEP_MIB` or reserve does not fail, it pages - speed collapses. Decode's own
  transient needs are not modelled; start at 1024.
- Swap churn if routing is flat: raise `ADAPT_RATIO`/`ADAPT_INTERVAL` or lower `ADAPT_SWAPS` if rounds keep swapping
  the full bound.
- The staging path relies on the mapping layout of main 97e2e71 (range tail mapped over the shared staging from its
  first byte); re-check `load_experts` when rebasing.
- Multi-session servers: ADAPT and the profile writer switch themselves off.
- The reserves override changes the load plan itself; values far below the defaults can make loads fail late or page.
