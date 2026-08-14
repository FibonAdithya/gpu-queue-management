"""Reaping lives in the runner because it has to run when nothing else is
alive — which is exactly when a leaked job needs reaping."""
from __future__ import annotations

import os
import shutil
import signal
import time

from . import ledger
from .claim import release_stale, claim_dir
from .config import RunnerConfig
from .preflight import compute_apps, own_pids
from .procs import descendants, pid_alive
from .queue import QueueRoot

MAX_ATTEMPTS = 1


def _signal(pid: int, sig: int) -> bool:
    try:
        os.kill(pid, sig)
        return True
    except OSError:
        return False


def _kill(pid: int) -> bool:
    return _signal(pid, signal.SIGKILL)


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
    # Two directories on purpose, and the divergence is load-bearing.
    # `records` came from `cfg.claim_dir`; bare `own_pids()` reads
    # `$GPU_CLAIM_DIR`. That is safe only because `own_pids` is a strict
    # superset of what `attribute` owns, so disagreement can only add
    # exemptions, never remove one -- and this call is the only thing
    # standing between a direct `gpu-claim` user's trainer and SIGKILL when
    # the two paths genuinely differ.
    #
    # They still can. `bootstrap.sh` now templates GPU_CLAIM_DIR into the
    # generated gpuq.toml as well as the supervisor environment, so a
    # freshly bootstrapped box agrees -- but it writes that config once and
    # never overwrites it, so a box bootstrapped before that change, or any
    # hand-edited config, diverges exactly as before. `cli_runner` warns at
    # startup; it does not refuse.
    #
    # So do not "clean this up" by passing `cfg.claim_dir` into
    # `own_pids()`, and do not route it through `attribute()`. Either
    # removes that protection, and
    # `test_a_divergent_runner_claim_dir_still_spares_a_direct_run` is
    # there to catch it. Unifying the two becomes safe only once a
    # divergent config is unreachable, which is a bootstrap change, not a
    # directory plumbed through here.
    exempt = set(protect) | own_pids()
    killed = []
    for app in unledgered:
        if app["pid"] not in exempt and _kill(app["pid"]):
            killed.append(app["pid"])
    return killed


def _running_trees(queue: QueueRoot) -> set[int]:
    """Every pid under every running job, not just the job's own.

    The whole tree, because the parts of one `reap()` call would otherwise
    disagree about the same job. Supervisor restarts the runner
    while a GPU job is running; the job survives, since it was started with
    `start_new_session=True`. On the next tick `release_stale` deletes that
    job's ledger record -- the record carries the *dead runner's* pid, not
    the job's -- and `requeue_orphans` deliberately leaves the job alone,
    because `spec.pid` is still alive. The sweep then arrives at a job with
    no record to charge it to, and `spec.pid` is normally a venv or shell
    wrapper, a `torchrun`, a dataloader parent: the process actually on the
    card is its *child*. Protecting only `spec.pid` SIGKILLs a live job the
    same call just decided to spare.

    Costs one recursive `ps` per running job, bounded by `cpu_slots +
    gpu_max_jobs`, and only inside the timer-gated sweep -- not on the
    every-tick recovery path.
    """
    protect: set[int] = set()
    for spec in queue.list_state("running"):
        if spec.pid:
            protect.add(spec.pid)
            protect |= descendants(spec.pid)
    return protect


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


def _exited(pid: int) -> bool:
    """Gone, or a zombie the runner has not `wait`ed for yet.

    `pid_alive` is `kill(pid, 0)`, which a zombie answers. The runner is
    the parent of the job it convicts and does not reap it until
    `collect()`, a phase later, so without this every conviction would
    spend both grace periods below waiting for a process that has already
    exited. `executor._kill_group` does not need this because `proc.poll()`
    reaps its child; here there is no Popen to poll.
    """
    if not pid_alive(pid):
        return True
    try:
        with open(f"/proc/{pid}/stat") as fh:
            stat = fh.read()
    except FileNotFoundError:
        return True     # it exited between `pid_alive` and this read
    except OSError:
        # Unreadable, not absent -- a `hidepid` mount hides other users'
        # entries. `pid_alive` has already said this pid exists, and only
        # its *state* is in question here, so the safe answer is "still
        # alive": reading it as exited empties `_kill_tree`'s alive list
        # on the SIGKILL pass, and a convicted trainer that blocks SIGTERM
        # then survives the watchdog entirely.
        return False
    # The comm field can contain spaces and parentheses; state is the
    # first field after the last ')'.
    fields = stat.rpartition(")")[2].split()
    return bool(fields) and fields[0] == "Z"


def _kill_tree(pid: int) -> bool:
    """SIGTERM a holder's whole tree, then SIGKILL what survives, by pid
    rather than by process group.

    killpg would be shorter and is wrong here: a `gpu-claim` launched from
    a script shares its group, so the group is not reliably the holder's
    own. Enumerating descendants kills exactly what the record is charged
    for and nothing else.

    The grace period is not decoration: a convicted trainer that gets only
    SIGKILL flushes no logs and writes no checkpoint, so the operator loses
    the run *and* the evidence. Grace periods match
    `executor._kill_group`'s.

    This blocks the runner's single thread, so the total is bounded at
    10s + 5s per conviction and cannot grow: the tree is enumerated once,
    up front, and both loops exit early once everything in it has exited.
    A card still held by a dying trainer is not the moment to admit more
    work anyway -- the same trade `_kill_group` documents.

    Returns whether the tree is off the card, which is not the same as
    whether a signal landed. A holder that had already exited gets no
    signal, and reporting that as failure told the runner the over-user
    was still running: it logged `COULD NOT KILL` over a dead process and,
    far worse, skipped stamping `_last_conviction`, so the co-tenant that
    OOMed *because* of the overage failed `_hit_by_a_convicted_co_tenant`
    and was never requeued. That is not a corner case -- an over-using
    trainer typically OOMs itself within milliseconds of its victim, i.e.
    right around the sweep that convicts it. False is reserved for the
    case that actually needs it: a tree still alive after SIGKILL, which
    on this shared claim directory means another user's process and an
    EPERM `_signal` swallowed.
    """
    tree = {pid} | descendants(pid)
    for sig, grace in ((signal.SIGTERM, 10.0), (signal.SIGKILL, 5.0)):
        alive = [p for p in tree if not _exited(p)]
        if not alive:
            return True
        for p in alive:
            _signal(p, sig)
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            if all(_exited(p) for p in tree):
                return True
            time.sleep(0.1)
    return all(_exited(p) for p in tree)


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
            # No "record" field: it carried `rec.name`, the bare filename
            # this function has just finished explaining is ambiguous, and
            # nothing consumed it. `owner` is what identifies the holder.
            convicted.append({"owner": rec.owner, "declared": rec.vram_mb,
                              "used": used, "usage_pid": rec.usage_pid})
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
        if apps is None:
            # A sweep that cannot see the process list measured nothing, so
            # it must not leave a strike banked. `WATCHDOG_STRIKES` counts
            # *consecutive* sweeps over the declaration -- both branches
            # that do run (`strikes.pop` under the limit, `strikes.clear()`
            # on a broken measurement) forget, and a blind sweep is the
            # blindest of the three. Left banked, a job that spikes once
            # now and once an hour later, with nvidia-smi unavailable in
            # between, is SIGKILLed on what is effectively one sample.
            if vram_strikes is not None:
                vram_strikes.clear()
        else:
            # Inside the guard: with no visible process list neither
            # consumer runs, and walking the claim directory to build
            # records nothing will read is pure cost on every sweep of a
            # box where nvidia-smi is broken.
            d = cfg.claim_dir if cfg.claim_dir else claim_dir()
            records = ledger.live_records(ledger.all_records(d))
            if cfg.kill_orphan_cuda:
                protect = _running_trees(queue)
                killed = kill_orphan_cuda(protect, records, apps)
            if cfg.enforce_vram and vram_strikes is not None:
                convicted = check_vram(records, apps, vram_strikes)
                for c in convicted:
                    # Whether the kill landed is not a detail the caller
                    # can infer: this directory is shared with hand-run
                    # `gpu-claim` jobs, so a holder can belong to another
                    # user, `_signal` swallows the EPERM, and that holder
                    # goes on over-using the card. Reporting the
                    # conviction alone would have the runner log `killed`
                    # over a process that is still running.
                    c["killed"] = bool(c["usage_pid"]) and _kill_tree(
                        c["usage_pid"])
    cleaned = clean_partials(queue)
    return {"stale_claims": stale, "requeued": requeued, "failed": failed,
            "killed_pids": killed, "cleaned_paths": cleaned,
            "convicted": convicted}
