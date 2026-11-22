# inkfer — Inkling 2 Mini on one RTX 5080 (16 GB): a 524k context that is actually usable

Notes, patches, launch profiles, benchmarks and agent tests from moving a local coding-agent stack to
**Inkling 2 Mini** on a single consumer card.

- Machine: RTX 5080 16 GB, Ryzen 7 7800X3D, 64 GB DDR5, Windows 11.
- Engine: [NInfer-all](https://github.com/iamwavecut/ninfer-all) PR #393 (quantized KV for Inkling 2) plus six local
  patches published here; [Basalt](https://github.com/jesdga95/basalt) with four local patches as a second engine;
  llama.cpp as a reference.
- Agent: oh-my-pi 18.18.8 (terminal) and omp-web 0.5.1 (browser), OpenAI-compatible API on `127.0.0.1:8081`.
- Quant: Unsloth UD-IQ4_XS in NInfer's v3 artifact format (15.4 GiB).

Every number below was measured on this one machine with the context **really filled**, not just allocated.
Vendor and engine-author claims were not copied into the tables. Dates: 21–22 November 2026.

The full write-up is in Russian: **[README.ru.md](README.ru.md)** (results, profiles, limits) and
**[NOTES.md](NOTES.md)** (the measurement journal, 16 sections, including the dead ends). The engine build notes are
in English: [src/BUILD-NOTES-ninfer.md](src/BUILD-NOTES-ninfer.md), [src/BUILD-NOTES-basalt.md](src/BUILD-NOTES-basalt.md).

## Results

### Speed against real context fill

Final build (NInfer PR #393 + patches 1–6), profile `max`: window 524,288, KV `rk4v4`, vision tower loaded.
In brackets: the same ladder before patch 5.

| Context filled | Prompt read, tok/s | Decode, tok/s | Cold read of the whole context | Next turn (cached), s | Planted notes found |
|---:|---:|---:|---:|---:|---|
| 8k (`speed`, bf16 KV) | 3,492 | 118–138 | 2.4 s | 0.25 | — |
| 32k (`speed`, bf16 KV) | 2,401 | 100–118 | 13.5 s | 0.42 | — |
| 131k | **2,621** (1,047) | **90–104** (57–77) | 50.6 s (2 min 05 s) | 0.47–0.52 | 3/3 |
| 260k | **1,850** (613) | **69–85** (40–51) | 2 min 03 s (6 min 50 s) | 0.65–0.87 | 3/3 |
| 391k | **1,490** (413) | **56–76** (30–37) | 4 min 01 s (15 min 48 s) | 0.88–0.93 | 3/3 |
| 504k | **1,278** (332) | **44–58** (21–31) | 6 min 09 s (24 min 54 s) | 1.0–1.1 | 3/3 |

"Planted notes" are three short notes at 10 / 50 / 90 % of the document that the model has to find and quote.
In all runs the page file did not grow, the disk was idle (experts are never read from the SSD), VRAM was 15.25 GiB
and the engine used up to 38.8 GiB of RAM. Speed depends on how much context is occupied, not on the window size.

### Agent work against Qwen3.8 Flash Next (the model this stack ran before)

Same prompts, same agent, isolated copies of the test project, graded by scripts that re-run the test suites.

| Task | Inkling 2 Mini | Qwen3.8 Flash Next (IQ3_S, 262k window) |
|---|---|---|
| Three-step chain: analyse a code base → find and fix three planted bugs → 8 memory questions | 15/15 facts, 3/3 bugs fixed, 8/8; 17.8 min on `max` (three runs: 17.8–19.2), 14.3 min on `balanced` | 15/15 and 14/15 facts, 3/3, 8/8; 16.3 and 16.9 min |
| Large unattended task with a sub-agent (10.5k-line server file, docs with line references, code change, tests) | 19 min 28 s, 205 requests, 97.4 % of prompt tokens from cache | 21 min 26 s (sub-agent did not start in that run) |
| Long task that does not fit in 262k: read 14,682 lines in full (≈ 370k tokens), write two documents, change code with a test, delegate to a sub-agent, then 10 questions from memory | **20.5 min, prompt up to 411k, 0 history compactions, 10/10**; all 14,682 lines really in context | 39.6 min, 4 compactions, 24 % of the lines in context, 5–6 of 10 |

How to read this: on tasks that fit in 200k the two models are **comparable** — run-to-run spread of one model is
minutes, and there are one to three runs per configuration. The long task was chosen to exceed Qwen's window; it
shows what a 524k window buys, not a general superiority. One run per model.

One unfavourable result is kept in the tables: in one of four NInfer chain runs Inkling sent the full test run to
a background task, ended its turn without waiting and never wrote the final report (the fixes and tests were
correct).

### Engines on this card

| Engine, quant | Window | Prompt read, tok/s | Decode, tok/s | What limits it |
|---|---:|---|---|---|
| llama.cpp (Unsloth build b12588), UD-IQ4_XS | 131k | 840 → 414 (8k → 120k) | 44 → 38 | experts run on the CPU |
| Basalt 0.5.5, NVFP4-q8 | 262k | 3,970 → 2,546 (66k → 251k) | 38–55 → 28–39 | every expert miss is read from RAM over PCIe |
| Basalt 0.5.5, NVFP4-q8 | 524k | 3,250 → 1,684 (132k → 496k) | 28–31 → 16 | same, plus 8 GiB of KV; drafts do not fit |
| Basalt main + local patches, NVFP4-q8 | 262k | 2,534 at 249k | 32–46 at 249k | same, eased by an adaptive on-card expert set |
| NInfer-all release 21.4, UD-IQ4_XS, bf16 KV | 131k | 3,492 → 1,104 (8k → 120k) | 118–138 → 63–75 | 32 KiB of KV per token pushes experts off the card |
| NInfer-all PR #393 + patches 1–4, rk8v4 / rk4v4 KV | 262k / 524k | 1,684 → 613 → 332 (66k → 251k → 496k) | 83–101 → 40–51 → 21–31 | scalar attention over a long KV |
| **NInfer-all PR #393 + patches 1–6, rk4v4 KV** | 524k | 2,621 → 1,850 → 1,278 (131k → 260k → 504k) | 90–104 → 69–85 → 44–58 | card memory (experts at depth) |

### What is not achieved

- 100+ tok/s decode holds only up to about 130k occupied tokens.
- Basalt still reads a prompt 1.35× faster at 250k (about 2,500 against 1,850 tok/s). It is not the main engine
  because it keeps a single conversation state: every side request of the agent costs a full re-read.
- Patch 5 does not speed up the exact-KV profile (`speed`, bf16): the tensor-core path exists only for quantized KV.
- A cold read of 500k tokens still takes six minutes. It happens once per conversation; after that 97–100 % of each
  prompt comes from cache.
- History compaction by the agent always re-reads from the start (the beginning of the prompt changes): 2–5 minutes
  at 200k. The compaction thresholds are therefore raised to 200k / 430k.

## Where the limits come from

- **Hardware.** The 4-bit weights are 14.7 GiB. Dense layers and about 1,870 of 3,072 experts fit on the card; the
  KV cache takes room from experts (3.3 GiB at 262k with rk8v4, 4.4 GiB at 524k with rk4v4). Deeper context means
  fewer experts on the card and more misses — this is the drop from ~120 to 69–85 tok/s at 260k and 44–58 at 504k.
  No setting fixes that on 16 GB.
- **Implementation (fixed here).** Attention over a long quantized KV was scalar (patch 5); a prompt could resume
  only from a whole cached prefix (patch 3); the host state store evicted the state it was about to need
  (patches 1, 4).
- **Defaults (fixed in the profiles).** `--host-cache-mib` at its default 8 GiB loses the prefix cache from 120k
  (109 s per turn instead of 0.9 s). The profiles use 12–20 GiB. More RAM than that gave no measurable gain.

## Engine patches

NInfer, on top of PR #393 head `f66e789eda919a15191e34828a9b73c41fc231a2` (the head as of 2026-11-21; the PR has
moved since). Series: [src/patches/ninfer](src/patches/ninfer); the same changes as two cumulative diffs in `src/`.

| # | Commit | What | Measured effect |
|---|---|---|---|
| 1 | `8a37fdd` | the state a prompt resumes from is not evicted from the host store | next turn at 120k: 0.9 s instead of 109 s |
| 2 | `88caff1` | a host image of quantized KV is sized by its real bytes | the store holds more states |
| 3 | `50d8a1e` | **trail points**: ring state (213 MiB) is copied to RAM every 8–16k prompt tokens | a prompt that diverges from the cache mid-way resumes from the nearest point: 3.6–3.9 s instead of 22 s at 44k; worst case at 190k 10–25 s instead of 4 min |
| 4 | `a9df5bf` | the store drops superseded images of the same conversation before the last image of another | the main chat keeps its cache while a sub-agent makes several requests in a row |
| 5 | `5b8595a` | **far keys of a quantized KV on the tensor cores** (sm_80+, rk KV only) | prompt read at 504k: 1,278 tok/s instead of 332; decode 44–58 instead of 21–31 |
| 6 | `d0bd340` | `--vision-upscale F`: enlarge an image before the vision tower | at 1.5 every line of dense UI/code screenshots is read; default behaviour unchanged |

Basalt, on top of main `97e2e7110578aeb7f8708dab176b910589b35047`: [src/patches/basalt](src/patches/basalt) — the
on-card expert set follows the conversation, prompt staging buffers are lent to experts during decode. Decode at
249k: 32–46 tok/s instead of 22–33. All switches are environment variables; without them the build behaves as
upstream.

How correctness was checked:

- Patch 5 against an FP64 oracle of the whole attention (`tools/attnbench/attn_bench.cu`, rk4v4 and rk8v4, depths
  1k–508k); seven greedy tasks are bit-identical with and without the patch.
- Trail points: reasoning and answer byte-identical to a recomputation from scratch (`agent-tests/trail_check.sh`).
- Basalt patches: 7/7 greedy tasks bit-identical.
- Quantized KV changes the output like any precision change, so it was checked by the planted notes (3/3 at every
  depth up to 504k) and by the agent runs above, not byte by byte.

Nothing here trades answer quality for speed beyond the KV quantization that PR #393 itself introduces; the `speed`
profile keeps the exact bf16 KV.

## Launch profiles

[profiles.json](profiles.json), started by `start_inkling.pyw PROFILE [--web | --tui]`. All NInfer profiles keep
experts in pinned RAM (`--expert-residency host`) and load the vision tower
(`--vision --vision-residency overlay --vision-upscale 1.5`).

| Profile | Window | KV | Engine RAM | Use |
|---|---:|---|---:|---|
| `speed` | 131,072 | bf16, 32 KiB/token | up to ~31 GiB | short and medium sessions, exact KV |
| `balanced` (default) | 262,144 | rk8v4, 12.75 KiB/token | up to ~35 GiB | everyday agent work |
| `max` | 524,288 | rk4v4, 8.75 KiB/token | up to ~39 GiB | very large code bases, long sessions |
| `basalt` (spare) | 262,144 | fp8, 16 KiB/token | ~15 GiB | fast prompt read, single state |

Vision in `overlay` mode costs 180 MiB of the on-card expert cache and ±2.5 % of speed (noise). The image checks
(`tools/vischeck.py`: photo, chart, two-image comparison, terminal screenshot, table, code, a dense 120-line
screenshot, tool call from image data, image at the start and at the end of a ~330k context) passed 13 of 13 on
the build they were written for. On the final binaries the 11 short checks passed in four runs of five; the fifth
had two misses — a grader pattern that was too narrow (since widened) and a tool called a second time instead of
an answer. Sampling is not pinned in these checks.

## Repository map

| Path | What |
|---|---|
| `README.ru.md`, `NOTES.md` | the full report and the measurement journal (Russian) |
| `profiles.json`, `start_inkling.pyw`, `start_qwen.pyw` | launch profiles; launchers that swap the loaded model and the agents' default model |
| `src/patches/ninfer`, `src/patches/basalt` | `git am` series for the two engines |
| `src/*.diff` | the same patches as cumulative diffs |
| `src/BUILD-NOTES-*.md`, `src/build-*.bat`, `src/*.cmd`, `src/*.sh` | how the engines were built on Windows (MSVC 14.44, CUDA 13.4.3 redistributables, vcpkg), including a header-dependency trap on a non-English MSVC |
| `bench/` | `ctxbench.py` (context-fill ladder with planted notes), `detcheck.py` (bit-exact output comparison), `summarize.py`, raw data in `bench/runs`, vision test images |
| `tools/` | `apicheck.py` (agent-facing API: streaming, reasoning, tool calls, stop/continue), `vischeck.py`, `visscale.py`, `mkvisimages.py`, `trailtest.py`, `alttest.py`, `logproxy.py`, `promptdiff.py`, `ninferlog.py`, `refcheck.py`, `make_shortcuts.ps1`, `attnbench/` |
| `agent-tests/` | task prompts, run scripts, graders (`grade.py`, `grade_long.py`), per-run grades and the comparison tables in `runs/` |
| `plans/`, `sweep.sh`, `run_*.sh` | benchmark series and engine start scripts for them |
| `logs/` | engine and benchmark logs per run |
| `agent-config/` | the oh-my-pi / omp-web `config.yml` and `models.yml` used (provider, context windows, compaction thresholds) |

## Reproducing

This is a working directory published as is, not a packaged product. Paths are hard-coded for the original machine
(`C:\WORKDIR\inkling` for this folder, `D:\MODELS\inkling2mini` for the model files, a Python 3.12
at `D:\MODELS\Strata\control\python`); expect to edit them. The scripts need only the standard library, plus
`psutil` (`bench/ctxbench.py`) and `Pillow` (`tools/mkvisimages.py`, `tools/visscale.py`).

1. Get the model: Inkling 2 Mini, Unsloth UD-IQ4_XS, as a NInfer v3 artifact (`…UD-IQ4_XS-ninfer-v3.ninfer`).
2. Build the engine (details in [src/BUILD-NOTES-ninfer.md](src/BUILD-NOTES-ninfer.md)):

   ```
   git clone https://github.com/iamwavecut/ninfer-all
   cd ninfer-all
   git fetch origin f66e789eda919a15191e34828a9b73c41fc231a2
   git checkout -b local/pr393-inkling f66e789eda919a15191e34828a9b73c41fc231a2
   git am /path/to/inkfer/src/patches/ninfer/*.patch
   ```

   If that commit is no longer served by GitHub, take the current PR #393 head and rebase the series.
3. Point `exe` and the model path in `profiles.json` at your build and file, then start a profile:

   ```
   pythonw start_inkling.pyw balanced
   ```

   or run the command line from the profile directly; the API is `http://127.0.0.1:8081/v1`.
4. Check and measure:

   ```
   python tools/apicheck.py 8081
   python tools/vischeck.py 8081
   python bench/ctxbench.py --base http://127.0.0.1:8081 --model inkling-max --label LABEL --lean --fills 131072,262144,393216,507904 --proc ninfer-serve.exe --extra "{\"reasoning_effort\":\"low\"}"
   bash agent-tests/trail_check.sh
   bash agent-tests/alt_check.sh
   ```

   The ladder to 504k takes about 25 minutes.

## What is not in this repository

- The model files and the engine binaries (they include NVIDIA runtime libraries), the build toolchain and the
  upstream source trees — only the patches are here.
- The test project the agent tasks ran on. It is a copy of third-party source code without a licence that allows
  redistribution, so the working copies and the agents' session transcripts are left out. The prompts, the graders
  and the grades are here; to re-run the tasks you need a project of your own and adapted graders.
- Raw request dumps (`*.requests.jsonl`) and a few build logs over 300 KiB.
- Windows shortcuts (`tools/make_shortcuts.ps1` creates them).

## Method caveats

One machine, one card, one quant per engine. Most agent comparisons are one to three runs per configuration, and the
spread between runs of the same model is minutes. Sampling was not pinned in the agent runs. Quality claims from
quant authors (KLD figures) were not re-measured. Treat the numbers as measurements of this setup, not as a
benchmark of the model.

## Licence and credits

The contents of this repository are under the [MIT licence](LICENSE), except the engine patches, which are
derivative works and carry the licence of their upstream: Apache-2.0 for NInfer-all, MIT for Basalt — see
[THIRD-PARTY.md](THIRD-PARTY.md).

Thanks to the authors of NInfer-all, of its PR #393 (quantized KV for Inkling 2), of Basalt, of oh-my-pi and
omp-web, and to Unsloth for the quants. The measurements, patches and write-ups were produced with Claude
(Anthropic) working as a coding agent on the owner's machine.
