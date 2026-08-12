"""Reaping lives in the runner because it has to run when nothing else is
alive — which is exactly when a leaked job needs reaping."""
from __future__ import annotations

import os
import shutil
import signal
from pathlib import Path

from . import ledger
from .claim import release_stale, claim_dir
from .config import RunnerConfig
from .preflight import compute_apps, own_pids
from .procs import descendants, pid_alive
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


def kill_orphan_cuda(protect: set[int], records: list, apps: list[dict]) -> list[int]:
    """Kill CUDA processes no live claim accounts for.

    Takes `apps` rather than fetching them so one nvidia-smi call serves
    both this and the VRAM watchdog; takes `records` so ownership is
    decided by the same `ledger.attribute` preflight uses, rather than by
    a second, subtly different pid set.
    """
    _, unledgered = ledger.attribute(apps, records)
    exempt = set(protect) | own_pids()
    killed = []
    for app in unledgered:
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


WATCHDOG_STRIKES = 2


def _kill_tree(pid: int) -> bool:
    """Kill a holder's whole tree, by pid rather than by process group.

    killpg would be shorter and is wrong here: a `gpu-claim` launched from
    a script shares its group, so the group is not reliably the holder's
    own. Enumerating descendants kills exactly what the record is charged
    for and nothing else.
    """
    ok = False
    for p in {pid} | descendants(pid):
        ok = _kill(p) or ok
    return ok


def check_vram(records: list, apps: list[dict],
               strikes: dict[str, int]) -> list[dict]:
    """Convict holders using more than they declared.

    Attribution, not prevention. The victim of an overage OOMs in
    milliseconds and this convicts in up to two sweeps -- what it buys is
    that the failure is legible afterwards, instead of two jobs sharing a
    bare CUDA OOM with nothing to say whose fault it was.
    """
    owned, unledgered = ledger.attribute(apps, records)
    if unledgered and not owned and any(r.usage_pid for r in records):
        # Every visible process is unattributable while records claim to
        # own trees: the measurement is broken, not the box overrun. Under
        # MPS nvidia-smi reports the server rather than its clients, which
        # looks exactly like this. Convicting here would kill the box's
        # own work.
        strikes.clear()
        return []

    convicted, seen = [], set()
    for rec in records:
        if rec.vram_mb is None:
            continue  # declared the whole card, so it cannot exceed it
        # Keyed by full path, matching ledger.attribute. A bare filename
        # is not unique across <key>.lock.d directories, and these strikes
        # persist across sweeps -- a collision would charge one holder's
        # strikes to another and kill the wrong job.
        key = str(rec.path)
        seen.add(key)
        used = ledger.used_mb(owned.get(key, []))
        if used <= rec.vram_mb:
            strikes.pop(key, None)
            continue
        strikes[key] = strikes.get(key, 0) + 1
        if strikes[key] >= WATCHDOG_STRIKES:
            strikes.pop(key, None)
            convicted.append({"owner": rec.owner, "declared": rec.vram_mb,
                              "used": used, "usage_pid": rec.usage_pid,
                              "record": rec.name})
    for key in list(strikes):
        if key not in seen:
            strikes.pop(key)  # the holder is gone; its strikes go with it
    return convicted


def reap(queue: QueueRoot, cfg: RunnerConfig,
         active_ids: set[str] | None = None,
         include_orphan_cuda: bool = True,
         vram_strikes: dict[str, int] | None = None) -> dict:
    """Recover what a dead runner left behind.

    Split by cost. Releasing claims, requeueing abandoned jobs and removing
    debris are file operations, cheap enough to run on every tick — and they
    are the recovery path, so they should be. Killing orphaned CUDA processes
    and running the VRAM watchdog both shell out to nvidia-smi and walk the
    process tree with ps; they are a safety net with no latency requirement,
    so the runner puts them on a timer and passes include_orphan_cuda=False
    the rest of the time.
    """
    stale = release_stale(cfg.claim_dir)
    requeued, failed = requeue_orphans(queue, active_ids)
    killed, convicted = [], []
    # Both consumers below need one nvidia-smi call and one ledger scan.
    # Gate that shared cost on whether either consumer is switched on --
    # a box with kill_orphan_cuda and enforce_vram both off should pay for
    # neither on every timer tick.
    if include_orphan_cuda and (cfg.kill_orphan_cuda or cfg.enforce_vram):
        apps = compute_apps()
        d = cfg.claim_dir if cfg.claim_dir else claim_dir()
        records = ledger.live_records(ledger.all_records(d))
        if apps is not None:
            if cfg.kill_orphan_cuda:
                protect = {s.pid for s in queue.list_state("running") if s.pid}
                killed = kill_orphan_cuda(protect, records, apps)
            if cfg.enforce_vram and vram_strikes is not None:
                convicted = check_vram(records, apps, vram_strikes)
                for c in convicted:
                    if c["usage_pid"]:
                        _kill_tree(c["usage_pid"])
    cleaned = clean_partials(queue)
    return {"stale_claims": stale, "requeued": requeued, "failed": failed,
            "killed_pids": killed, "cleaned_paths": cleaned,
            "convicted": convicted}
