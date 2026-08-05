"""Reaping lives in the runner because it has to run when nothing else is
alive — which is exactly when a leaked job needs reaping."""
from __future__ import annotations

import os
import shutil
import signal
from pathlib import Path

from .claim import release_stale, pid_alive
from .config import RunnerConfig
from .preflight import compute_apps, own_pids
from .queue import QueueRoot

MAX_ATTEMPTS = 1


def _kill(pid: int) -> bool:
    try:
        os.kill(pid, signal.SIGKILL)
        return True
    except OSError:
        return False


def requeue_orphans(queue: QueueRoot,
                    active_ids: set[str] | None) -> tuple[list[str], list[str]]:
    """Requeue running jobs whose process is gone — once, then fail."""
    active = active_ids or set()
    requeued, failed = [], []
    for spec in queue.list_state("running"):
        if spec.id in active:
            continue
        if spec.pid and pid_alive(spec.pid):
            continue
        if spec.attempts < MAX_ATTEMPTS:
            queue.requeue(spec)
            requeued.append(spec.id)
        else:
            spec.error = (f"abandoned after {spec.attempts + 1} attempts; "
                          "the runner or the job died repeatedly")
            queue.finish(spec, ok=False)
            failed.append(spec.id)
    return requeued, failed


def kill_orphan_cuda(protect: set[int]) -> list[int]:
    apps = compute_apps()
    if apps is None:
        return []  # cannot see the list; killing blind is worse than leaking
    exempt = set(protect) | own_pids()
    killed = []
    for app in apps:
        if app["pid"] not in exempt and _kill(app["pid"]):
            killed.append(app["pid"])
    return killed


def clean_partials(queue: QueueRoot) -> list[str]:
    cleaned = []
    work = queue.root / "work"
    if not work.is_dir():
        return cleaned
    live = {s.id for s in queue.list_state("running")}
    for path in work.rglob("*.part"):
        # Never sweep inside a job that is still going. A .part file there is
        # that job's business, not debris. This matters because reaping now
        # runs on every tick rather than only between jobs.
        rel = path.relative_to(work)
        if rel.parts and rel.parts[0] in live:
            continue
        cleaned.append(str(path))
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)
    for d in work.iterdir():
        if d.is_dir() and d.name not in live and not any(d.iterdir()):
            d.rmdir()
    return cleaned


def reap(queue: QueueRoot, cfg: RunnerConfig,
         active_ids: set[str] | None = None,
         include_orphan_cuda: bool = True) -> dict:
    """Recover what a dead runner left behind.

    Split by cost. Releasing claims, requeueing abandoned jobs and removing
    debris are file operations, cheap enough to run on every tick — and they
    are the recovery path, so they should be. Killing orphaned CUDA processes
    shells out to nvidia-smi and walks the process tree with ps; it is a
    safety net with no latency requirement, so the runner puts it on a timer
    and passes include_orphan_cuda=False the rest of the time.
    """
    stale = release_stale(cfg.claim_dir)
    requeued, failed = requeue_orphans(queue, active_ids)
    killed = []
    if include_orphan_cuda and cfg.kill_orphan_cuda:
        protect = {s.pid for s in queue.list_state("running") if s.pid}
        killed = kill_orphan_cuda(protect)
    cleaned = clean_partials(queue)
    return {"stale_claims": stale, "requeued": requeued, "failed": failed,
            "killed_pids": killed, "cleaned_paths": cleaned}
