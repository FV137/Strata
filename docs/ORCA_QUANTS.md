# OrcaRouter quantization choices

Setup offers all 15 quantizations and the separate Q8_0-MTP variant published in
[the OrcaRouter repository](https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF/tree/e43d00f4e2b8b40b89f75e9adeb1045ac34c8acc).
The pinned revision is `e43d00f4e2b8b40b89f75e9adeb1045ac34c8acc`.
These are experimental installer options. Adding an option does not establish its generation quality, speed, or GPU compatibility on your machine.

## Available files

Sizes are decimal GB for the model shards alone. Allow additional space for the pack, draft runtime, engine, and optional 0.91 GB vision projector.
The standalone MTP draft and F16 vision projector are auxiliary files, not additional target quantizations.

| `--model` | Shards | Download GB | Engine selection |
| --- | ---: | ---: | --- |
| `IQ2_M` | 2 | 80.09 | Pinned prebuilt or local build |
| `IQ2_XXS` | 2 | 74.85 | Pinned prebuilt or local build |
| `IQ3_M` | 2 | 89.54 | Pinned prebuilt or local build |
| `IQ3_XXS` | 2 | 85.20 | Pinned prebuilt or local build |
| `IQ4_XS` | 3 | 97.47 | Pinned prebuilt or local build |
| `Q2_K` | 2 | 80.45 | Build this checkout |
| `Q3_K_L` | 3 | 96.66 | Build this checkout |
| `Q3_K_M` | 3 | 94.14 | Build this checkout |
| `Q3_K_S` | 2 | 88.76 | Build this checkout |
| `Q4_K_M` | 3 | 119.15 | Pinned prebuilt or local build |
| `Q4_K_S` | 3 | 111.77 | Pinned prebuilt or local build |
| `Q5_K_M` | 3 | 134.11 | Pinned prebuilt or local build |
| `Q5_K_S` | 3 | 127.73 | Pinned prebuilt or local build |
| `Q6_K` | 5 | 167.64 | Build this checkout |
| `Q8_0` | 5 | 188.21 | Pinned prebuilt or local build |
| `Q8_0-MTP` | 5 | 190.98 | Pinned prebuilt or local build |

Q4_K_M remains the Orca menu default. Q8_0-MTP is a separately named artifact, not a ninth-bit precision level. Selecting it does **not** switch Strata to the repository’s standalone MTP draft; setup retains its existing original-Qwen draft runtime. The target verifies draft proposals. The gated Q8_0-MTP tensor layout has not been inspected here; its extra draft tensors may need additional compatibility work even though Q8_0 arithmetic is supported.

Q2_K, Q3_K_L/M/S, and Q6_K require this checkout’s new native kernels. Setup builds from local source and records that requirement so a later start or hardware switch cannot silently substitute the older pinned prebuilt. These five options are currently unavailable through setup on Windows AMD; use an NVIDIA source build or Linux HIP. Setup stops before dependency/model downloads on that unsupported path. The new CUDA/HIP paths still need validation on hardware.

## Install and switch

Use the full checkout, including `orca-quants.json`; copying `setup.py` alone is insufficient. Accept the model repository’s access terms and set `HF_TOKEN` locally. The token is used only for this repository over HTTPS on huggingface.co; it is not forwarded to redirects or written into a run config.

```sh
./setup.sh --setup --family orca --model IQ3_XXS --context 32768 --kv int8 --vision no --no-start
./setup.sh --setup --family orca --model Q4_K_M --context 32768 --kv int8 --vision no --no-start
./setup.sh --setup --family orca --model Q6_K --context 32768 --kv int8 --vision no --no-start
```

On Windows, replace `./setup.sh` with `START-HERE.bat`. Add `--yes` for the explicit choices and default answers when preparing a pod non-interactively. Compiler installation may be needed for source builds. No pod is provisioned by these commands.

Each choice gets a separate pack/tokenizer directory and config, such as `packs/orca-iq3_xxs`, `strata-orca-iq3_xxs.json`, and `strata-orca-q8_0-mtp.json`. Existing Qwen/Swift IQ3_XXS profiles retain their own files and memory estimates. Run `./setup.sh` and select a saved config to start it. Unset `HF_TOKEN` in your shell before starting inference if it is no longer needed.

Already-downloaded shards must retain every original filename:

```sh
./setup.sh --setup --family orca --model Q8_0 --gguf-dir /path/to/shards --context 32768 --vision no --no-start
```

Models use the resident RAM-budget path on one GPU. Setup uses the **full shard total as a conservative expert-arena upper bound**, not a measured RAM requirement. Its inherited 48 GB planning threshold is not a measured minimum for these models. Multiple GPUs with a layer split require enough host RAM for the full file set plus runtime headroom; a large GPU alone does not eliminate that requirement. Performance must be measured for each placement.

For an explicit multi-GPU split, add `--gpus all` (or selected IDs such as `--gpus 0,1`):

```sh
./setup.sh --setup --family orca --model Q4_K_M --gpus all --context 32768 --vision no --no-start --yes
```

On an existing install, `./setup.sh --gpus all --yes` updates the saved selection. An explicit split stops if
host RAM is insufficient or a resident budget was explicitly requested; setup no longer silently saves one GPU.
In Docker, expose GPUs with Docker’s `--gpus all` and select them in Strata with `-e GPUS=all`. Device IDs come
from `nvidia-smi` inside that environment; CUDA children use UUIDs and honor inherited visibility masks. See
[the multi-GPU guide](MULTI_GPU.md) for masked devices, numeric CUDA ordering, and diagnostic steps.

For a pod, keep the server on localhost and reach it through an SSH tunnel. If deliberately binding beyond localhost, configure an API key. Keep quant files on persistent storage and select one at a time; these options do not download all variants together.

## Pinned verification and offline reuse

`orca-quants.json` commits the exact filenames, byte counts, and Git LFS pointer blob IDs for all variants. Hugging Face exposes those blob IDs publicly but redacts gated weight SHA-256 values without access. For a newly selected quant, setup:

1. Reads the small raw LFS pointer at the pinned revision, using local Hugging Face access.
2. Checks its exact byte count and Git blob SHA-1 against the committed catalog, and strictly parses its declared weight size and SHA-256.
3. Checks downloaded or supplied local shards with those SHA-256 values before packing them.

This pins weight hashes indirectly through Git metadata; it is not a publisher signature. Q4_K_M and the vision projector retain their already-committed direct SHA-256 checks. Setup never falls back to `main` for Orca. Successful weight checks use local receipts tied to file size, timestamps and identity; changed files and old hash-only receipts are rehashed. The local filesystem is a trusted boundary.

Verified pointers are cached under the selected data folder’s `metadata/orca-lfs/<revision>/` and rechecked on every reuse. After that first authenticated lookup, local shards plus the metadata cache can be used offline for the model-verification step. Engine/source and Python dependencies also need to be available for a completely offline installation. A new quant supplied with `--gguf-dir` still requires its verified pointer metadata; filenames and sizes alone do not suffice.

## Compare output quality

Use the same source commit and engine build for every quant in a comparison. Building all candidates with `--build` avoids mixing the older prebuilt and new local kernels. Keep the prompt set, system prompt, chat template, thinking mode, context, KV type, output-token limit, sampling settings, seed, draft settings, and concurrency fixed. Use greedy sampling for an initial comparison, then repeat a fixed set of seeds for sampled tasks. Identical seeds do not guarantee identical answers across quantizations or hardware.

Start with your own frozen evaluation set and score it without knowing the quant label: coding tasks by tests passed, extraction by exact match or a fixed rubric, and review tasks by confirmed findings and false positives. Do not let the evaluated model grade its own answers. Save raw responses, truncation/errors, and prompt hashes; an empty or truncated answer is not a passing test.

An existing repeatable recall check can provide one narrow quality measure:

```sh
mkdir -p results
python tools/needle_bench.py --lengths 8k,32k --depths 10,50,90 --url http://127.0.0.1:8080 --out results/orca-q4_k_m-needles.json
```

Repeat against each selected quant with a different result filename. Keep the checkout and third-party docs unchanged: this tool builds its prompts from those files. It uses fixed code words, greedy sampling and thinking off. Record successful recalls out of completed cases **and** failures/skips separately. Needle recall does not measure coding or security-review quality.

Keep a run ledger:

| Quant | Prompt-set hash | Engine/source commit | GPU / RAM | Settings | Pass / total | False positives | Prompt / output tokens | End-to-end time | Peak VRAM / RAM |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Fill after running | | | | | | | | | |

Separate output quality from performance. Local-versus-pod runs change hardware, storage, offload, and possibly arithmetic; compare quants on one placement first, then repeat the matrix on the other. Record warm-up and cache state and repeat timing runs. Do not treat a faster download or higher tokens/s as evidence of better answers.

## Validation scope

Installer tests use mocked downloads and hardware. Small synthetic GGUF tests exercise packing, control-projection conversion, and preservation of native expert weights. Native parity tests are provided for the added formats. Full gated weights have not been downloaded or generated with during this change, and new GPU kernels have not been run here. No quality rankings or new inference benchmarks are claimed.

The older [IQ3_XXS compatibility report](ORCA.md) describes one historical measured configuration, not validation of every choice above.
