"""Refuse to start when someone else already holds the card.

This cannot stop a determined direct run — the lock is advisory. It
converts accidental contention into a fast, readable failure instead of a
CUDA OOM half an hour into a training run.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from . import cgroups
from . import ledger
# list_claims is re-exported, not used here; kept for callers.
from .claim import all_claim_dirs, claim_dir, list_claims   # noqa: F401
from .procs import descendants


class PreflightFailed(RuntimeError):
    """CUDA processes hold the card that no live ledger record claims."""


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


def own_pids(directory=None) -> set[int]:
    """Every pid the claim protocol accounts for, plus this process's own.

    Kept for `reaper.kill_orphan_cuda` and the tests covering it -- an
    earlier docstring justified it by callers outside this package, and
    grep finds none. The finer question -- which *record* owns a pid --
    is `ledger.attribute`. This one is deliberately coarser: it is the
    last exemption before a SIGKILL, so over-exempting is the safe way to
    be wrong.

    Bare, it reads *every* directory a claim on this box could be in, not
    just the one this process would write to. `claim_dir()` resolves
    `$GPU_CLAIM_DIR` in the calling process, and the reaper is a
    supervisor unit while the claim writer is an interactive shell that
    never inherits a unit's environment. Reading only our own left a
    hand-run `gpu-claim` invisible here and killed 48% of one session's
    runs (issue #19). See `claim.all_claim_dirs`.

    An explicit `directory=` still means exactly that one: a caller who
    names a directory is asking about that directory, and widening the
    answer would make the argument mean nothing.
    """
    pids = {os.getpid(), os.getppid()} | descendants(os.getpid())
    dirs = [Path(directory)] if directory else all_claim_dirs()
    for d in dirs:
        for rec in ledger.live_records(ledger.all_records(d)):
            pids.add(rec.pid)
            if rec.usage_pid is not None:
                pids.add(rec.usage_pid)
                pids.update(descendants(rec.usage_pid))
            else:
                pids.update(descendants(rec.pid))
    return pids


def own_scopes(directory=None) -> set[str]:
    """Every cgroup scope the claim protocol accounts for.

    The scope half of `own_pids`, deliberately built as a sibling rather
    than folded into it: `own_pids` stays exactly as issue #19 left it.
    What the two share is breadth, and that is the whole reason this
    exists. `kill_orphan_cuda` took its scope exemption from
    `ledger.attribute` alone, over records read from one directory
    (`cfg.claim_dir`), while its pid exemption came from a bare
    `own_pids()` reading every directory a claim could be in. So a
    `--scope-pid` claim written to a directory in `all_claim_dirs()` that
    is not `cfg.claim_dir` had its wrapper's pid tree spared and its
    *container* SIGKILLed -- while a plain `gpu-claim -- python train.py`
    in the same setup survived, because its trainer is a descendant.

    That divergence is not hypothetical. `cli_claim` already documents
    that the daemon reads `[queue].claim_dir` while "this process's
    environment cannot name it": an interactive shell and a supervisor
    unit systematically disagree about `$GPU_CLAIM_DIR`, which is issue
    #19. Without this, a correctly formed `gpu-claim --scope-pid $(docker
    inspect ...)` issued from a shell exempts nothing.

    `scope_is_live` is consulted, so a record whose anchor died or whose
    container restarted onto a fresh scope id exempts nothing. A void
    scope left standing is an amnesty for whatever the kernel puts at
    that path next -- the same unbounded window issue #21 was about, one
    mechanism over.

    An explicit `directory=` still means exactly that one, for the reason
    `own_pids` gives: a caller who names a directory is asking about that
    directory.

    Over-exempting is the safe way to be wrong here, same as `own_pids`:
    this is the last check before a SIGKILL.
    """
    scopes: set[str] = set()
    dirs = [Path(directory)] if directory else all_claim_dirs()
    for d in dirs:
        for rec in ledger.live_records(ledger.all_records(d)):
            if rec.scope_cgroup and ledger.scope_is_live(rec):
                scopes.add(rec.scope_cgroup)
    return scopes


def unledgered_processes(allow: set[int] | None = None,
                         directory=None) -> list[dict]:
    """CUDA processes no live ledger record accounts for.

    This replaces "foreign", which was the right question when the card
    admitted one job: with sharing, a declared co-tenant's process is
    someone else's and entirely legitimate. What is still contention is a
    process nobody has claimed capacity for.
    """
    apps = compute_apps()
    if apps is None:
        return []
    d = Path(directory) if directory else claim_dir()
    records = ledger.live_records(ledger.all_records(d))
    _, unledgered = ledger.attribute(apps, records)
    exempt = set(allow or set()) | {os.getpid(), os.getppid()}
    exempt |= descendants(os.getpid())
    return [a for a in unledgered if a["pid"] not in exempt]


# The old name, kept so nothing importing it breaks.
foreign_processes = unledgered_processes


def preflight(allow: set[int] | None = None, directory=None,
              scope: str | None = None) -> None:
    """Refuse to start when a CUDA process holds the card with no claim.

    `scope` is a cgroup the caller is about to claim; processes inside it
    are not contention.
    """
    # One query, not two: asking twice costs a second nvidia-smi and lets
    # the "can we see the list?" check and the "who is on the card?" check
    # disagree about what they saw.
    apps = compute_apps()
    if apps is None:
        print("gpu-claim: warning: cannot enumerate CUDA processes on this "
              "box; proceeding on the advisory lock alone", file=sys.stderr)
        return
    d = Path(directory) if directory else claim_dir()
    records = ledger.live_records(ledger.all_records(d))
    _, unledgered = ledger.attribute(apps, records)
    exempt = set(allow or set()) | {os.getpid(), os.getppid()}
    exempt |= descendants(os.getpid())
    stray = [a for a in unledgered if a["pid"] not in exempt]
    if scope:
        # A claim that has not been taken yet. `attribute` covers a scope
        # once the record is on disk, but this runs *before* that, so the
        # container's in-flight CUDA has nothing to be charged to and
        # would read as a stranger holding the card. Filtering here is
        # what lets a busy container be claimed at all.
        stray = [a for a in stray if not cgroups.in_scope(a["pid"], scope)]
    if stray:
        lines = [f"  pid {a['pid']:>7}  {a['used_mb'] or '?'} MiB  {a['name']}"
                 for a in stray]
        raise PreflightFailed(
            "CUDA processes hold this GPU with no claim on it:\n"
            + "\n".join(lines))
