"""Refuse to start when someone else already holds the card.

This cannot stop a determined direct run — the lock is advisory. It
converts accidental contention into a fast, readable failure instead of a
CUDA OOM half an hour into a training run.
"""
from __future__ import annotations

import os
import subprocess
import sys

from .claim import list_claims
from .procs import descendants as _descendants, pid_alive


class PreflightFailed(RuntimeError):
    """Foreign CUDA processes hold the card."""


def _run(argv: list[str]) -> str:
    return subprocess.run(argv, check=True, capture_output=True,
                          text=True, timeout=15).stdout


def compute_apps() -> list[dict] | None:
    """None means we cannot see the process list, which is not the same
    as seeing that it is empty."""
    try:
        out = _run(["nvidia-smi",
                    "--query-compute-apps=pid,used_memory,process_name",
                    "--format=csv,noheader"])
    except Exception:
        return None
    if "not supported" in out.lower():
        return None
    apps = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3 or not parts[0].isdigit():
            continue
        mem = parts[1].replace("MiB", "").strip()
        apps.append({
            "pid": int(parts[0]),
            "used_mb": int(mem) if mem.isdigit() else None,
            "name": parts[2],
        })
    return apps


def own_pids() -> set[int]:
    """Every pid the claim protocol accounts for, including their children.

    A claim names the process that took the lock, but the process actually
    on the card is normally its child: `gpu-claim` runs the command as a
    subprocess, and the runner starts jobs the same way. Expanding only our
    own tree exempted a direct `gpu-claim` run's recorded pid while leaving
    the trainer underneath it covered by nothing — so `kill_orphan_cuda`
    SIGKILLed a legitimate run as an orphan, and preflight read it as
    foreign contention.

    Roots are collected before they are walked so that the common case —
    a claim whose pid *is* this process — does not walk the same tree twice.
    """
    roots = {os.getpid()}
    for _, body in list_claims():
        pid = int(body.get("pid", -1))
        if pid > 0 and pid_alive(pid):
            roots.add(pid)
    pids = roots | {os.getppid()}
    for root in roots:
        pids.update(_descendants(root))
    return pids


def _foreign(apps: list[dict], allow: set[int] | None) -> list[dict]:
    exempt = set(allow or set()) | own_pids()
    return [a for a in apps if a["pid"] not in exempt]


def foreign_processes(allow: set[int] | None = None) -> list[dict]:
    apps = compute_apps()
    if apps is None:
        return []
    return _foreign(apps, allow)


def preflight(allow: set[int] | None = None) -> None:
    # One query, not two: asking twice costs a second nvidia-smi and lets
    # the "can we see the list?" check and the "who is on the card?" check
    # disagree about what they saw.
    apps = compute_apps()
    if apps is None:
        print("gpu-claim: warning: cannot enumerate CUDA processes on this "
              "box; proceeding on the advisory lock alone", file=sys.stderr)
        return
    foreign = _foreign(apps, allow)
    if foreign:
        lines = [f"  pid {a['pid']:>7}  {a['used_mb'] or '?'} MiB  {a['name']}"
                 for a in foreign]
        raise PreflightFailed(
            "foreign CUDA processes hold this GPU:\n" + "\n".join(lines))
