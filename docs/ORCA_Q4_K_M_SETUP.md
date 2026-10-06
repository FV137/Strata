# OrcaRouter Q4_K_M setup

Q4_K_M remains available and is now part of the complete [OrcaRouter quantization setup](ORCA_QUANTS.md).
That guide covers all published choices, pinned verification, separate saved configs, and output comparisons.

```sh
./setup.sh --setup --family orca --model Q4_K_M --context 32768 --vision no --no-start
```

Use the full checkout, including `orca-quants.json`. The original three shard names, pinned revision, and
direct SHA-256 checks are retained. The model shards total 119,150,722,944 bytes.
