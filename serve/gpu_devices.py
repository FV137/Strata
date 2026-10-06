"""NVIDIA management IDs for configuration, stable UUIDs for CUDA child processes."""
from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path


def nvidia_gpus() -> list[dict]:
    """Query the devices exposed to nvidia-smi, including sparse container indices."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,compute_cap,driver_version",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return []
    if result.returncode:
        return []
    found = []
    for line in result.stdout.strip().splitlines():
        try:
            index, uuid, name, memory, capability, driver = [part.strip() for part in line.split(",")]
            found.append({"index": int(index), "uuid": uuid, "name": name, "vram_gb": float(memory) / 1024,
                          "arch": capability.replace(".", ""), "driver": driver})
        except ValueError:
            continue
    return found


def cuda_driver_uuids() -> list[str]:
    """Ask the driver which UUIDs this process sees, without creating a context.

    Called only in an isolated subprocess: cuInit reads CUDA's mask and device
    order once, so changing them in the setup/server process would be unreliable.
    """
    cuda = ctypes.WinDLL("nvcuda.dll") if os.name == "nt" else ctypes.CDLL("libcuda.so.1")
    status = cuda.cuInit(0)
    if status == 100:                                # CUDA_ERROR_NO_DEVICE
        return []
    if status:
        raise OSError(f"cuInit failed ({status})")
    count = ctypes.c_int()
    if cuda.cuDeviceGetCount(ctypes.byref(count)):
        raise OSError("cuDeviceGetCount failed")
    result = []
    for ordinal in range(count.value):
        device, raw = ctypes.c_int(), (ctypes.c_ubyte * 16)()
        if cuda.cuDeviceGet(ctypes.byref(device), ordinal) or cuda.cuDeviceGetUuid_v2(ctypes.byref(raw), device):
            raise OSError("CUDA device UUID query failed")
        result.append("GPU-" + str(uuid.UUID(bytes=bytes(raw))))
    return result


def cuda_mask_uuids(env: dict) -> list[str]:
    """Resolve numeric masks using CUDA's actual ordering, including containers."""
    try:
        result = subprocess.run([sys.executable, str(Path(__file__).resolve())], env=dict(env),
                                capture_output=True, text=True, timeout=15)
        values = json.loads(result.stdout) if result.returncode == 0 else None
        if isinstance(values, list) and all(isinstance(v, str) and v.startswith("GPU-") for v in values):
            return values
    except (OSError, subprocess.TimeoutExpired, ValueError):
        pass
    raise ValueError("Cannot resolve CUDA_VISIBLE_DEVICES using the CUDA driver. Check the NVIDIA driver in this "
                     "environment, or use GPU UUIDs from nvidia-smi for CUDA_VISIBLE_DEVICES; numeric CUDA "
                     "ordinals can differ from nvidia-smi GPU numbers.")


def cuda_visible_gpus(found: list[dict], env=None) -> list[dict]:
    """Honor an inherited CUDA mask; indices stay those shown by nvidia-smi.

    CUDA also accepts unique GPU UUID prefixes. An empty mask hides every GPU,
    and an invalid identifier ends the visible sequence (CUDA's documented rule).
    UUIDs are preferable when a launcher or container changes device numbering.
    """
    env = os.environ if env is None else env
    if "CUDA_VISIBLE_DEVICES" not in env:
        return found
    tokens = [token.strip() for token in env["CUDA_VISIBLE_DEVICES"].split(",")]
    if any(token.startswith("MIG-") for token in tokens):
        raise ValueError("MIG CUDA_VISIBLE_DEVICES cannot be mapped to Strata's whole-GPU selections. "
                         "MIG selection is not supported here; expose full GPUs and use their GPU UUIDs.")
    if any(token.isdigit() for token in tokens):
        ids = cuda_mask_uuids(env)
        by_uuid = {g["uuid"]: g for g in found}
        if any(value not in by_uuid for value in ids):
            raise ValueError("CUDA_VISIBLE_DEVICES resolves to a device absent from nvidia-smi's whole-GPU list. "
                             "Check the container's GPU exposure; MIG selections are not supported here.")
        return [by_uuid[value] for value in ids]
    visible = []
    for token in tokens:
        matches = [g for g in found if token.startswith("GPU-") and g.get("uuid", "").startswith(token)]
        if len(matches) != 1 or matches[0] in visible:
            break
        visible.append(matches[0])
    return visible


def cuda_device_value(indices: list[int], env: dict, inherited=None) -> str:
    """Map configuration IDs to UUIDs, without widening the launcher's CUDA mask."""
    found = nvidia_gpus()
    masks = [source for source in (inherited or {}, env) if "CUDA_VISIBLE_DEVICES" in source]
    if not found and not masks:
        # Direct server starts historically need no nvidia-smi (for example on
        # minimal installations). Keep that path when there is no mask to resolve.
        return ",".join(str(i) for i in indices)
    visible = found
    seen = set()
    for source in masks:
        mask = (source["CUDA_VISIBLE_DEVICES"], source.get("CUDA_DEVICE_ORDER"))
        if mask in seen:
            continue
        seen.add(mask)
        # Resolve each mask against the complete inventory, then intersect them.
        # Resolving a UUID prefix against a filtered set could hide an ambiguity.
        permitted = cuda_visible_gpus(found, source)
        visible = [g for g in visible if g in permitted]
    allowed = {g["index"]: g for g in visible}
    for index in indices:
        if index not in allowed:
            restrictions = list(dict.fromkeys(source["CUDA_VISIBLE_DEVICES"] for source in masks))
            restriction = (" with CUDA_VISIBLE_DEVICES=" + " and ".join(repr(v) for v in restrictions)
                           if restrictions else "")
            raise ValueError(f"GPU {index} is not available{restriction}. Check nvidia-smi in this environment "
                             "and select its visible GPU numbers; in Docker expose every selected card with "
                             "docker run --gpus. Use GPU UUIDs for CUDA_VISIBLE_DEVICES when numbering changes.")
    return ",".join(allowed[i]["uuid"] for i in indices)


if __name__ == "__main__":
    try:
        print(json.dumps(cuda_driver_uuids()))
    except (OSError, AttributeError) as error:
        sys.exit(str(error))
