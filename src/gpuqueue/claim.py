"""The advisory lock and its claim record.

Enforcement is advisory because flock cannot be otherwise between
unprivileged processes. The claim record exists so that a human or agent
looking at a busy card learns *who* holds it without a running service.
"""
from __future__ import annotations

import getpass
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

from . import ledger
from .gpuid import gpu_key, total_vram_mb
from .ledger import ClaimBusy          # re-exported: callers import it here
from .procs import pid_alive           # re-exported for the same reason

DEFAULT_CLAIM_DIR = "/var/lock/gpu"
WAIT_POLL_S = 0.5


def claim_dir() -> Path:
    return Path(os.environ.get("GPU_CLAIM_DIR", DEFAULT_CLAIM_DIR))


def read_claim(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def job_orphaned(job_pid: int | None, runner_pid: int | None) -> bool:
    """True when a job is still running but the runner that started it is gone.

    Jobs run in their own session so the runner can kill a whole process
    group, which also means they survive a runner that dies abruptly. Such a
    job keeps running with nobody supervising it: no watchdog enforces its
    timeout and nothing will collect its result.

    Ownership is the signal, not the parent pid. Reparenting does not reliably
    land on init — any process marked a subreaper (a user systemd, a container
    init) adopts it instead, so "PPid is 1" is true on some hosts and false on
    others for the very same situation.

    A job with no recorded runner cannot be judged, so it is not reported: an
    unknown owner is not evidence of an absent one.
    """
    if not job_pid or not pid_alive(job_pid):
        return False
    if runner_pid is None:
        return False
    return not pid_alive(runner_pid)


def default_usable_mb() -> int | None:
    """What a standalone `gpu-claim` may admit against.

    None when the card cannot be queried, which `ledger.fits` turns into
    exclusive-only admission -- degraded, and the same posture preflight
    already takes when it cannot enumerate the card.
    """
    total = total_vram_mb()
    return None if total is None else total - ledger.DEFAULT_RESERVE_MB


def list_claims(directory: Path | None = None) -> list[tuple[Path, dict]]:
    d = Path(directory) if directory else claim_dir()
    return [(r.path, r.to_dict()) for r in ledger.all_records(d)]


def release_stale(directory: Path | None = None) -> list[dict]:
    """Remove records whose owning pid is gone. Returns what it freed."""
    d = Path(directory) if directory else claim_dir()
    released = []
    for r in ledger.all_records(d):
        if not pid_alive(r.pid):
            released.append(r.to_dict())
            ledger.remove(r)
    return released


@contextmanager
def gpu_claim(key: str | None = None, owner: str | None = None,
              cmd: list[str] | None = None, wait: bool = False,
              directory: Path | None = None, vram_mb: int | None = None,
              usable_mb: int | None = None, own_usage: bool = True):
    """Hold a share of the card. `vram_mb=None` means the whole of it.

    `own_usage=False` is for the runner, which takes the card before the
    job process exists and fills the usage pid in after launch.

    `wait` polls capacity rather than blocking on flock: the mutex is
    released the instant `acquire` returns, so there is no longer a kernel
    queue to wait in.
    """
    d = Path(directory) if directory else claim_dir()
    key = key or gpu_key()
    if usable_mb is None:
        usable_mb = default_usable_mb()
    while True:
        try:
            rec = ledger.acquire(
                key, vram_mb=vram_mb, owner=owner or _default_owner(),
                cmd=cmd, directory=d, usable_mb=usable_mb,
                usage_pid=os.getpid() if own_usage else None)
            break
        except ClaimBusy:
            if not wait:
                raise
            time.sleep(WAIT_POLL_S)
    try:
        yield rec
    finally:
        ledger.remove(rec)


def _default_owner() -> str:
    try:
        return getpass.getuser()
    except Exception:
        return f"uid{os.getuid()}"
