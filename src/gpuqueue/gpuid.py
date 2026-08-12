"""GPU identity.

Keying the lock on the UUID rather than the index is load-bearing: two
processes with different CUDA_VISIBLE_DEVICES mappings both see their card
as index 0, so an index-keyed lock hands them different locks for the same
physical GPU.

Normalization is load-bearing for the same reason one level down. We read
nvidia-smi, which reports "GPU-<hex>"; a torch-based implementation of the
same protocol reports the bare hex. Unnormalized, those are two lock files
for one card.

torch is deliberately never imported here. A JAX project must be able to
use this lock, and importing torch in the runner daemon costs seconds and
initializes a CUDA context in the one process that should never touch the
card.
"""
from __future__ import annotations

import re
import subprocess

_PREFIX = re.compile(r"^(GPU|MIG)-", re.IGNORECASE)
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


class GpuIdError(RuntimeError):
    """No usable GPU identity could be derived."""


def normalize_gpu_uuid(raw: str) -> str:
    s = (raw or "").strip()
    s = _PREFIX.sub("", s)
    s = s.strip().lower()
    if not s:
        raise GpuIdError(f"empty GPU uuid: {raw!r}")
    return s


def _run(argv: list[str]) -> str:
    return subprocess.run(argv, check=True, capture_output=True,
                          text=True, timeout=15).stdout


def gpu_uuid_from_nvidia_smi(index: int = 0) -> str | None:
    try:
        out = _run(["nvidia-smi", "--query-gpu=uuid",
                    "--format=csv,noheader"])
    except Exception:
        return None
    lines = [l.strip() for l in out.splitlines() if l.strip()]
    return lines[index] if index < len(lines) else None


def gpu_name_index_fallback(index: int = 0) -> str | None:
    """Design-specified fallback for drivers that report no UUID.

    Weaker than a UUID — it cannot survive a CUDA_VISIBLE_DEVICES remap —
    but a degraded key shared by everyone on the box still serializes the
    card, and refusing to run is worse.
    """
    try:
        out = _run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    except Exception:
        return None
    lines = [l.strip() for l in out.splitlines() if l.strip()]
    return f"{lines[index]}-{index}" if index < len(lines) else None


def gpu_key(index: int = 0) -> str:
    for source in (gpu_uuid_from_nvidia_smi, gpu_name_index_fallback):
        raw = source(index)
        if raw:
            return normalize_gpu_uuid(raw)
    raise GpuIdError(
        "cannot derive a GPU key: nvidia-smi reports neither a uuid nor a "
        "device name. Is a GPU visible in this container?"
    )


def cuda_visible_value(index: int = 0) -> str | None:
    """What to put in CUDA_VISIBLE_DEVICES to pin a job to this card.

    Deliberately *not* `gpu_key`. That one is normalized -- lowercased, prefix
    stripped -- because it names a lock file, and two spellings of one card
    have to collapse to one lock. The CUDA driver is a different consumer: it
    resolves the uuid as the driver itself spells it, so this returns
    nvidia-smi's own string untouched.

    An index would be the obvious alternative and is wrong for the same reason
    the lock is not keyed on one: it is not stable under a remap, so pinning
    "0" means "whichever card this process happens to see first" -- the guess
    the pin exists to eliminate.

    Returns None when no uuid is available. `gpu_name_index_fallback` is good
    enough to key a lock, since any shared string serializes the card, but it
    is not something the driver can resolve. There is then nothing honest to
    pin to and the caller should leave the variable alone.

    Not MIG-aware. `--query-gpu=uuid` always reports the *physical* card's
    "GPU-<hex>"; on a MIG-partitioned card CUDA only accepts the instance's
    "MIG-<hex>" and a job pinned to the physical uuid sees no devices at all.
    Note that the degraded path above does not save us there -- a uuid *is*
    returned, just not one the driver will take. Nothing on this box uses MIG
    (`_PREFIX` accepts the spelling only because a lock key may arrive from
    another tool), so this is a note for whoever grows the fleet, not a bug
    with a victim today.
    """
    raw = gpu_uuid_from_nvidia_smi(index)
    return raw.strip() if raw and raw.strip() else None


def lock_filename(key: str) -> str:
    return _UNSAFE.sub("-", key) + ".lock"
