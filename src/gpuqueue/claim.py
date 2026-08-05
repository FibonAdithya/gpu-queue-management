"""The advisory lock and its claim file.

Enforcement is advisory because flock cannot be otherwise between
unprivileged processes. The claim file exists so that a human or agent
looking at a busy card learns *who* holds it without a running service.
"""
from __future__ import annotations

import errno
import fcntl
import json
import os
import getpass
from contextlib import contextmanager
from pathlib import Path

from .gpuid import gpu_key, lock_filename
from .spec import utcnow_iso

DEFAULT_CLAIM_DIR = "/var/lock/gpu"


class ClaimBusy(RuntimeError):
    """Another process holds the card."""


def claim_dir() -> Path:
    return Path(os.environ.get("GPU_CLAIM_DIR", DEFAULT_CLAIM_DIR))


def _paths(key: str, directory: Path) -> tuple[Path, Path]:
    lock = directory / lock_filename(key)
    return lock, lock.with_suffix(lock.suffix + ".json")


def read_claim(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError as e:
        return e.errno == errno.EPERM  # exists but not ours
    return True


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


def list_claims(directory: Path | None = None) -> list[tuple[Path, dict]]:
    d = Path(directory) if directory else claim_dir()
    if not d.is_dir():
        return []
    out = []
    for p in sorted(d.glob("*.lock.json")):
        body = read_claim(p)
        if body:
            out.append((p, body))
    return out


def release_stale(directory: Path | None = None) -> list[dict]:
    """Remove claim files whose owning pid is gone. Returns what it freed."""
    released = []
    for path, body in list_claims(directory):
        if not pid_alive(int(body.get("pid", -1))):
            released.append(body)
            path.unlink(missing_ok=True)
    return released


@contextmanager
def gpu_claim(key: str | None = None, owner: str | None = None,
              cmd: list[str] | None = None, wait: bool = False,
              directory: Path | None = None):
    d = Path(directory) if directory else claim_dir()
    d.mkdir(parents=True, exist_ok=True)
    key = key or gpu_key()
    lock_path, claim_path = _paths(key, d)

    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o666)
    flags = fcntl.LOCK_EX if wait else fcntl.LOCK_EX | fcntl.LOCK_NB
    try:
        fcntl.flock(fd, flags)
    except OSError:
        os.close(fd)
        held = read_claim(claim_path) or {}
        raise ClaimBusy(
            f"GPU {key} is held by pid {held.get('pid', '?')} "
            f"({held.get('owner', '?')}): {' '.join(held.get('cmd') or []) or '?'}"
        )

    body = {
        "pid": os.getpid(),
        "owner": owner or _default_owner(),
        "cmd": cmd or [],
        "started_at": utcnow_iso(),
        "key": key,
    }
    claim_path.write_text(json.dumps(body, indent=2) + "\n")
    try:
        yield body
    finally:
        claim_path.unlink(missing_ok=True)
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _default_owner() -> str:
    try:
        return getpass.getuser()
    except Exception:
        return f"uid{os.getuid()}"
