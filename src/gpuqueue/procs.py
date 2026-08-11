"""Process liveness and process trees.

Split out of `claim` and `preflight` because `ledger` needs both and
`claim` imports `ledger`. This module imports nothing from gpuqueue, which
is what keeps that from being a cycle.
"""
from __future__ import annotations

import errno
import os
import subprocess


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError as e:
        return e.errno == errno.EPERM  # exists but not ours
    return True


def descendants(pid: int) -> set[int]:
    """Every process under `pid`, transitively.

    Used to decide who is exempt from the orphan sweep and who owns a CUDA
    process. Both callers need the whole tree: a claim names the process
    that took the lock, but the process on the card is normally its child,
    and a trainer's dataloader workers are children of that.
    """
    try:
        out = subprocess.run(["ps", "-o", "pid=", "--ppid", str(pid)],
                             check=True, capture_output=True, text=True,
                             timeout=15).stdout
    except Exception:
        return set()
    kids = {int(l) for l in out.split() if l.strip().isdigit()}
    for k in list(kids):
        kids |= descendants(k)
    return kids
