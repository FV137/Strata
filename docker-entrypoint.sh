#!/bin/sh
# Entry point for the Strata container. The engine is compiled during docker
# build and lives in the image, so the first start only downloads the model.
# The install config is kept on the /data volume so a recreated container skips
# the setup pass and goes straight to serving.
set -e
cd /opt/strata || exit 1

STRATA_DATA="${STRATA_DATA:-/data}"
FAMILY="${FAMILY:-qwen}"
MODEL="${MODEL:-IQ2_XS}"
CONTEXT="${CONTEXT:-32768}"
VISION="${VISION:-no}"          # no | yes | cpu (the image encoder on the CPU)
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
# Keep API_KEY as an alias; use the server's environment variable on every start,
# including an existing volume. Never put the secret in setup's command line.
if [ "${API_KEY+x}" = x ]; then
  if [ "${STRATA_API_KEY+x}" = x ] && [ "$API_KEY" != "$STRATA_API_KEY" ]; then
    echo "API key settings disagree: use only STRATA_API_KEY or API_KEY." >&2
    exit 1
  fi
  export STRATA_API_KEY="$API_KEY"
fi
KV="${KV:-}"                    # int8 | q4_0 | k8v4; empty: setup.py's own default (int8)
GPUS="${GPUS:-}"                # "0,2" or "all": one model across several cards (docs/MULTI_GPU.md)
GPU="${GPU:-}"                  # one card, numbered as nvidia-smi numbers them
LAYER_SPLIT="${LAYER_SPLIT:-}"  # with GPUS: where each later card's layers start (default: auto)
LOW_RAM="${LOW_RAM:-auto}"      # on: the experts come from the pack's experts.bin, not from RAM

# setup.py starts the newest strata-*.json it finds, so link in exactly the one
# this family and model were set up with. The config is the recorded output of
# that setup (the pack, the profile, the quant, the KV decision), not settings
# the entry point could rebuild from env vars. qwen has an empty family tag.
case "$FAMILY" in qwen) prefix="" ;; *) prefix="${FAMILY}-" ;; esac
tag="${prefix}$(printf '%s' "$MODEL" | tr 'A-Z' 'a-z')"
cfg="$STRATA_DATA/config/strata-$tag.json"
# Validate before setup downloads anything or starts an engine. A saved key still
# works, but an explicitly empty environment key must not silently fall back to it.
.venv/bin/python - "$HOST" "$cfg" <<'PYEOF'
import json, pathlib, sys
from serve.access import api_key_of, require_key_for_bind

try:
    path = pathlib.Path(sys.argv[2])
    cfg = json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else {}
    require_key_for_bind(sys.argv[1], api_key_of(None, cfg))
except (OSError, ValueError) as e:
    sys.exit(f"[strata] {e}")
PYEOF
mkdir -p "$STRATA_DATA/config"

# REINSTALL is only needed to change settings for a model that is already set up
# (context, vision, KV). HOST and the environment key apply on every start.
# Switching between models already on the
# volume needs no setup pass: their config is already there.
#
# KV / GPUS / GPU / LAYER_SPLIT are passed only when set, so an unset one keeps
# setup.py's own default. LOW_RAM is always passed: setup.py measures the PC's RAM
# from /proc/meminfo, which in a container is the host's total, not the container's
# limit, so a memory-capped container has to ask for the low-RAM mode itself.
if [ "${REINSTALL:-0}" = "1" ] || [ ! -f "$cfg" ]; then
  echo "Setting up $tag: downloading the model (~70 GB; the engine is already in the image)."
  set -- --family "$FAMILY" --model "$MODEL" --context "$CONTEXT" --vision "$VISION" \
    --data-dir "$STRATA_DATA" --host "$HOST" \
    --port "$PORT" --no-start --low-ram "$LOW_RAM"
  if [ -n "$KV" ]; then set -- "$@" --kv "$KV"; fi
  if [ -n "$GPUS" ]; then set -- "$@" --gpus "$GPUS"; fi
  if [ -n "$GPU" ]; then set -- "$@" --gpu "$GPU"; fi
  if [ -n "$LAYER_SPLIT" ]; then set -- "$@" --layer-split "$LAYER_SPLIT"; fi
  .venv/bin/python setup.py --setup --yes "$@"
  # Setup atomically replaces the working config, leaving the volume's saved copy
  # intact. Keep its existing key before copying the new model settings back.
  # Environment-only keys stay in the environment, including during key rotation.
  .venv/bin/python - "$cfg" "/opt/strata/strata-$tag.json" <<'PYEOF'
import json, pathlib, sys

saved, current = map(pathlib.Path, sys.argv[1:])
if saved.is_file() and current.is_file():
    old = json.loads(saved.read_text(encoding="utf-8-sig"))
    key = old.get("api_key")
    if isinstance(key, str) and key and key == key.strip():
        new = json.loads(current.read_text(encoding="utf-8-sig"))
        new["api_key"] = key
        tmp = current.with_name(current.name + ".tmp")
        tmp.write_text(json.dumps(new, indent=1), encoding="utf-8")
        tmp.replace(current)
PYEOF
  [ -e "/opt/strata/strata-$tag.json" ] && { cmp -s "/opt/strata/strata-$tag.json" "$cfg" || cp -f "/opt/strata/strata-$tag.json" "$cfg"; }
else
  [ -e "/opt/strata/strata-$tag.json" ] || ln -s "$cfg" "/opt/strata/strata-$tag.json"
fi

# Later starts skip straight here: setup.py finds the installed config and
# launches serve/server.py (OpenAI- and Anthropic-compatible API on :8080).
# GPUS / GPU / LAYER_SPLIT are repeated on purpose. Given at the start they pin the
# cards for this model, and setup.py saves them in its config; without them a config
# that names one card is offered once to a pair, on its own, when the host has two
# cards that can share the model (setup.py's offer_together, docs/MULTI_GPU.md).
set -- --host "$HOST" --port "$PORT"
if [ -n "$GPUS" ]; then set -- "$@" --gpus "$GPUS"; fi
if [ -n "$GPU" ]; then set -- "$@" --gpu "$GPU"; fi
if [ -n "$LAYER_SPLIT" ]; then set -- "$@" --layer-split "$LAYER_SPLIT"; fi
exec .venv/bin/python setup.py "$@"
