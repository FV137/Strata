# OrcaRouter Q4_K_M installer addition

Base: Strata v0.1.39, commit 6f32ec070f23ced9f50e704d854d775da52591ab.

Select `--family orca --model Q4_K_M` or choose OrcaRouter in the interactive menu.
Model repository: https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF
Pinned model revision: e43d00f4e2b8b40b89f75e9adeb1045ac34c8acc.
Three model shards total 119,150,722,944 bytes (~119.15 GB), plus packing/runtime files and the original Qwen MTP draft head. The optional Orca F16 projector is another 907,543,296 bytes.

## Integration

- Preserves the existing families and default selection; adds an explicit Orca family.
- Downloads this family's three original filenames at its pinned revision, verifies their published size and SHA-256, and refuses a fallback to main.
- Uses `--compat-bf16`, an independent Orca pack/tokenizer/config, and the existing resident RAM-budget path.
- Requires engine 0.1.38 or newer (Q5_0 expert support, upstream PR #473). The current 0.1.39 script still applies its ordinary engine-version checks.
- Uses the model's own optional F16 vision projector and checks its published size and SHA-256.
- Supports the gated download with an environment-provided HF_TOKEN. The bearer header is attached only to this repository on https://huggingface.co and is not forwarded by urllib redirects. It is not embedded in generated config files.
- Existing local shards can be supplied through `--gguf-dir`; retain their original filenames.

## Local use

Replace setup.py in a Strata v0.1.39 checkout with the supplied file; this is not a standalone program independent of Strata's other files.
Accept the model's Hugging Face access terms if needed, and supply HF_TOKEN locally. Do not send the token in chat.

```bash
./setup.sh --family orca --model Q4_K_M --context 32768 --vision no --no-start
```

This prepares the installation without launching inference. After a token-authenticated setup, unset HF_TOKEN before starting the server so subprocesses do not inherit that token. The installer does not remove environment variables for the parent shell.

For already-downloaded shards:

```bash
./setup.sh --family orca --model Q4_K_M --gguf-dir /path/to/orca-shards --context 32768 --vision no --no-start
```

Run `./setup.sh` again and select the prepared Orca configuration to start it. Keep the normal localhost binding; set an API key if exposing it elsewhere.

## Review and verification

- 58 focused tests pass: `python -m unittest tools.test_setup_orca tools.test_setup_pins tools.test_setup_choices -q`.
- Seven new tests initially failed in six cases against the unmodified installer, then all seven passed after implementation.
- Syntax compilation and `git diff --check` pass.
- Full installer test discovery: modified 265 tests, 47 failures, one error, one skipped. Untouched upstream: 258 tests, the identical 47 failing cases and one error, one skipped. No additional failing cases.
- Shared failure groups: golden setup/config expectations; WindowsDetection.test_prebuilt_hip_zip; IQ4XS.test_amd_is_not_asked (attempts to write ROCm test metadata under the runtime prefix).
- An existing ResourceWarning for an unclosed /proc/cpuinfo handle also appears in focused tests.

This is installer-level validation, not an inference benchmark. The three model files were not downloaded, packed, or run here: the model download requires authorized Hugging Face access, and no GPU inference environment was used. Upstream PR #473 reports an end-to-end run of Orca Q4_K_M and introduced the Q5_0 support used by this path.

RAM sizing deliberately uses the full shard total as a conservative upper bound for the expert arena because the gated tensor directories were not available for direct inspection. The 48 GB budget-mode planning threshold is inherited from the existing budget models and is not a measured minimum for Orca. The engine sizes actual expert tensors when it loads them. AMD and multi-GPU performance for this integration have not been measured.

## Existing upstream failing cases

- test_amd_is_not_asked (test_setup_unsloth.IQ4XS.test_amd_is_not_asked)
- test_enter_for_every_question (test_setup_golden.Golden.test_enter_for_every_question) [128GB-1x24GB qwen IQ3_S]
- test_enter_for_every_question (test_setup_golden.Golden.test_enter_for_every_question) [128GB-1x24GB qwen IQ3_XXS]
- test_enter_for_every_question (test_setup_golden.Golden.test_enter_for_every_question) [128GB-1x24GB qwen Q2_0]
- test_enter_for_every_question (test_setup_golden.Golden.test_enter_for_every_question) [128GB-1x24GB qwen recommended]
- test_enter_for_every_question (test_setup_golden.Golden.test_enter_for_every_question) [128GB-1x24GB unsloth UD-Q4_K_XL]
- test_enter_for_every_question (test_setup_golden.Golden.test_enter_for_every_question) [32GB-2x24GB qwen IQ3_S]
- test_enter_for_every_question (test_setup_golden.Golden.test_enter_for_every_question) [32GB-2x24GB qwen IQ3_XXS]
- test_enter_for_every_question (test_setup_golden.Golden.test_enter_for_every_question) [32GB-2x24GB qwen Q2_0]
- test_enter_for_every_question (test_setup_golden.Golden.test_enter_for_every_question) [32GB-2x24GB qwen recommended]
- test_enter_for_every_question (test_setup_golden.Golden.test_enter_for_every_question) [47GB-2x16GB qwen IQ3_S]
- test_enter_for_every_question (test_setup_golden.Golden.test_enter_for_every_question) [47GB-2x16GB qwen IQ3_XXS]
- test_enter_for_every_question (test_setup_golden.Golden.test_enter_for_every_question) [47GB-2x16GB qwen Q2_0]
- test_enter_for_every_question (test_setup_golden.Golden.test_enter_for_every_question) [47GB-2x16GB qwen recommended]
- test_enter_for_every_question (test_setup_golden.Golden.test_enter_for_every_question) [64GB-1x32GB qwen IQ3_S]
- test_enter_for_every_question (test_setup_golden.Golden.test_enter_for_every_question) [64GB-1x32GB qwen IQ3_XXS]
- test_enter_for_every_question (test_setup_golden.Golden.test_enter_for_every_question) [64GB-1x32GB qwen Q2_0]
- test_enter_for_every_question (test_setup_golden.Golden.test_enter_for_every_question) [64GB-1x32GB qwen recommended]
- test_enter_for_every_question (test_setup_golden.Golden.test_enter_for_every_question) [64GB-1x32GB unsloth UD-Q4_K_XL]
- test_enter_for_every_question (test_setup_golden.Golden.test_enter_for_every_question) [96GB-1x16GB qwen IQ3_S]
- test_enter_for_every_question (test_setup_golden.Golden.test_enter_for_every_question) [96GB-1x16GB qwen IQ3_XXS]
- test_enter_for_every_question (test_setup_golden.Golden.test_enter_for_every_question) [96GB-1x16GB qwen Q2_0]
- test_enter_for_every_question (test_setup_golden.Golden.test_enter_for_every_question) [96GB-1x16GB qwen recommended]
- test_enter_for_every_question (test_setup_golden.Golden.test_enter_for_every_question) [96GB-1x16GB unsloth UD-Q4_K_XL]
- test_prebuilt_hip_zip (test_setup_amd.WindowsDetection.test_prebuilt_hip_zip)
- test_yes (test_setup_golden.Golden.test_yes) [128GB-1x24GB qwen IQ3_S]
- test_yes (test_setup_golden.Golden.test_yes) [128GB-1x24GB qwen IQ3_XXS]
- test_yes (test_setup_golden.Golden.test_yes) [128GB-1x24GB qwen Q2_0]
- test_yes (test_setup_golden.Golden.test_yes) [128GB-1x24GB qwen recommended]
- test_yes (test_setup_golden.Golden.test_yes) [128GB-1x24GB unsloth UD-Q4_K_XL]
- test_yes (test_setup_golden.Golden.test_yes) [32GB-2x24GB qwen IQ3_S]
- test_yes (test_setup_golden.Golden.test_yes) [32GB-2x24GB qwen IQ3_XXS]
- test_yes (test_setup_golden.Golden.test_yes) [32GB-2x24GB qwen Q2_0]
- test_yes (test_setup_golden.Golden.test_yes) [32GB-2x24GB qwen recommended]
- test_yes (test_setup_golden.Golden.test_yes) [47GB-2x16GB qwen IQ3_S]
- test_yes (test_setup_golden.Golden.test_yes) [47GB-2x16GB qwen IQ3_XXS]
- test_yes (test_setup_golden.Golden.test_yes) [47GB-2x16GB qwen Q2_0]
- test_yes (test_setup_golden.Golden.test_yes) [47GB-2x16GB qwen recommended]
- test_yes (test_setup_golden.Golden.test_yes) [64GB-1x32GB qwen IQ3_S]
- test_yes (test_setup_golden.Golden.test_yes) [64GB-1x32GB qwen IQ3_XXS]
- test_yes (test_setup_golden.Golden.test_yes) [64GB-1x32GB qwen Q2_0]
- test_yes (test_setup_golden.Golden.test_yes) [64GB-1x32GB qwen recommended]
- test_yes (test_setup_golden.Golden.test_yes) [64GB-1x32GB unsloth UD-Q4_K_XL]
- test_yes (test_setup_golden.Golden.test_yes) [96GB-1x16GB qwen IQ3_S]
- test_yes (test_setup_golden.Golden.test_yes) [96GB-1x16GB qwen IQ3_XXS]
- test_yes (test_setup_golden.Golden.test_yes) [96GB-1x16GB qwen Q2_0]
- test_yes (test_setup_golden.Golden.test_yes) [96GB-1x16GB qwen recommended]
- test_yes (test_setup_golden.Golden.test_yes) [96GB-1x16GB unsloth UD-Q4_K_XL]
