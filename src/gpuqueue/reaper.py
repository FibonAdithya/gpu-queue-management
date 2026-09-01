"""Reaping lives in the runner because it has to run when nothing else is
alive — which is exactly when a leaked job needs reaping."""
from __future__ import annotations

import os
import shutil
import signal
import time
from pathlib import Path

from . import cgroups
from . import killlog
from . import ledger
from .claim import sweep_stale, claim_dir, all_claim_dirs
from .config import RunnerConfig
from .preflight import compute_apps, own_pids, own_scopes
from .procs import descendants, pid_alive
from .queue import QueueRoot

MAX_ATTEMPTS = 1

# Shorter than `_kill_tree`'s 10s on purpose. A convicted holder is one
# we want to checkpoint; an unledgered process is contending for a card
# someone may be blocked on, and this only has to be long enough for a
# SIGTERM handler to write a line. Paid once per `orphan_cuda_interval_s`
# inside the timer-gated sweep, not on every tick.
ORPHAN_TERM_GRACE_S = 5.0


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


def kill_orphan_cuda(protect: set[int], records: list,
                     apps: list[dict]) -> list[dict]:
    """Kill CUDA processes no live claim accounts for.

    Takes `apps` rather than fetching them so one nvidia-smi call serves
    both this and the VRAM watchdog; takes `records` so ownership is
    decided by the same `ledger.attribute` preflight uses, rather than by
    a second, subtly different pid set.
    """
    _, unledgered = ledger.attribute(apps, records)
    # Two directories on purpose, and the divergence is load-bearing.
    # `records` came from `cfg.claim_dir`; bare `own_pids()` reads every
    # directory a claim on this box could be in. This call is the only
    # thing standing between a direct `gpu-claim` user's trainer and
    # SIGKILL when the two paths differ.
    #
    # This used to be justified by `own_pids` being a strict *superset* of
    # what `attribute` owns, so that disagreement could only add
    # exemptions. That was false, and it was the bug in issue #19. It held
    # only while the claim writer and the reaper resolved `$GPU_CLAIM_DIR`
    # to the same value, and they systematically do not: the daemon gets
    # the variable from a supervisor unit, and an interactive shell never
    # inherits a unit's environment. The two sets were then *disjoint*
    # rather than nested, the hand-run claim was invisible, and 48% of one
    # session's runs died on `exit -9` with an empty stderr.
    #
    # The superset property is now built rather than assumed:
    # `claim.all_claim_dirs()` names every directory a claim could have
    # landed in, whichever environment wrote it, and `own_pids` reads them
    # all. Over-exempting is the safe way to be wrong here -- it is the
    # last check before a SIGKILL.
    #
    # So still do not "clean this up" by passing `cfg.claim_dir` into
    # `own_pids()`, and do not route it through `attribute()`. Either
    # narrows the exemption back to one directory chosen by one process,
    # which is what killed those runs.
    # `test_a_divergent_runner_claim_dir_still_spares_a_direct_run` catches
    # the config-vs-environment split;
    # `test_a_claim_the_daemons_environment_cannot_see_is_still_spared`
    # catches the two-environment one that a single-process test cannot
    # express.
    #
    # `own_scopes()` is the same argument one mechanism over, and it is
    # here for the same reason it is bare. The scope exemption used to
    # live *only* inside `ledger.attribute` above, over `records` from a
    # single directory, while the pid exemption beside it read every
    # directory a claim could be in. A `--scope-pid` claim written to a
    # directory in `all_claim_dirs()` that is not `cfg.claim_dir` then had
    # its wrapper's pid tree spared and its container SIGKILLed, while a
    # plain `gpu-claim -- python train.py` in the same setup survived
    # because its trainer is a descendant. Two exemptions of different
    # breadth is what issue #19 was; making them the same breadth is what
    # fixed it, and a scope is not exempt from that.
    exempt = set(protect) | own_pids()
    scopes = own_scopes()
    victims = [a for a in unledgered
               if a["pid"] not in exempt
               and not any(cgroups.in_scope(a["pid"], s) for s in scopes)]
    if not victims:
        return []
    # Read before signalling: /proc/<pid>/cgroup is gone the moment the
    # process is, and this field is the one that tells the victim's
    # operator it was their container and not their algorithm.
    killed = [{"pid": a["pid"], "name": a.get("name"),
               "used_mb": a.get("used_mb"),
               "cgroup": cgroups.cgroup_of(a["pid"]),
               # Filled in below. Present from the start so every record
               # this writes has the field, whether or not the ladder got
               # that far.
               "sigkilled": False}
              for a in victims]
    # SIGTERM everything, then one shared grace, then SIGKILL what is
    # left. Batched rather than per-victim: a grace each would stall the
    # runner tick by N x grace, and there is no reason the second
    # victim's grace should start after the first's has finished.
    #
    # The ladder at all because a SIGKILLed process writes no stderr. Its
    # caller sees `exit -9` with an empty message and reads it as its own
    # bug -- on 2026-09-01 an agent rewrote a correct index and submitted
    # a worse method on that reading (issue #24). `_kill_tree` has had
    # this since it was written; the orphan sweep never did.
    #
    # Only what was actually signalled is returned, and `signalled` is its
    # own name because the grace loop below reassigns `alive` down to the
    # survivors. `_signal` swallows two failures that mean nothing was
    # done: ESRCH, where the victim exited on its own between the
    # nvidia-smi sample and the SIGTERM, and EPERM, where it belongs to
    # another user -- this claim directory is shared with hand-run
    # `gpu-claim` jobs -- and goes on holding the card. Neither is
    # escalated to SIGKILL either, since both drop out of the list here.
    #
    # Reporting them anyway put them in `kills.jsonl`, and
    # `skills/gpu-jobs/SKILL.md` tells an agent that a pid there was killed
    # by the queue: a process that crashed on its own would be named as a
    # queue kill and the agent would stop debugging its real crash. That is
    # issue #24's misdiagnosis with the arrow reversed, so `killed` stays
    # purely the pre-signal cgroup snapshot and this is the answer.
    signalled = [d for d in killed if _signal(d["pid"], signal.SIGTERM)]
    alive = list(signalled)
    if alive:
        deadline = time.monotonic() + ORPHAN_TERM_GRACE_S
        while time.monotonic() < deadline:
            alive = [d for d in alive if not _exited(d["pid"])]
            if not alive:
                break
            time.sleep(0.1)
        for d in alive:
            # Recorded per victim so the runner's line can say what the
            # ladder actually did rather than describing both rungs
            # whenever it kills anything.
            d["sigkilled"] = _kill(d["pid"])
    return signalled


def _running_trees(queue: QueueRoot) -> set[int]:
    """Every pid under every running job, not just the job's own.

    The whole tree, because the parts of one `reap()` call would otherwise
    disagree about the same job. Supervisor restarts the runner
    while a GPU job is running; the job survives, since it was started with
    `start_new_session=True`. On the next tick the claim sweep deletes that
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



def _swept_dirs(attributed_from) -> list[Path]:
    """Every claim directory this sweep is responsible for.

    One list, two uses, and that is the point. It is what `reap` releases
    stale records from and what `_consulted_dirs` names as the ledgers a
    claim would have spared a process from -- so a directory that can grant
    an exemption is a directory something sweeps. Issue #21 was those two
    answers drifting apart: `own_pids` read both of `all_claim_dirs()`
    while the sweep read `cfg.claim_dir` alone, so dead records under the
    other one were removed by nothing, accumulated for the life of the box,
    and each sat there as an exemption waiting for the kernel to reuse its
    pid.

    `attributed_from` is a third directory only when `[queue].claim_dir`
    and the reaper's own `$GPU_CLAIM_DIR` diverge -- a split `cli_runner`
    warns about and permits -- and on every ordinary box it dedups away.

    Appended, deduped on `.resolve()`, and falling back to the literal
    path on OSError -- all of that is `all_claim_dirs`'s to do, and it
    does it here through the `extra` argument rather than in a second copy
    of the loop. `gpu-claim --reap` builds its set the same way from the
    same function: two spellings of "every directory a claim could be in"
    are two lists that drift, and this whole module exists because two of
    them did.
    """
    return all_claim_dirs(attributed_from)


def _consulted_dirs(attributed_from) -> list[str]:
    """Every claim directory a live claim in which would have spared a
    process from this sweep.

    Two mechanisms, one answer. `own_pids()` exempts a pid claimed under
    any of `claim.all_claim_dirs()`; `ledger.attribute` never puts a pid
    claimed under `attributed_from` into `unledgered` in the first place.
    An operator reading the kill line is asking "where would a claim have
    saved me", and both spare by the same amount, so both belong.

    `attributed_from` is a third directory only when `[queue].claim_dir`
    and the reaper's own `$GPU_CLAIM_DIR` diverge -- a split `cli_runner`
    warns about and permits -- and on every ordinary box it dedups away.
    Left out, the kill line asserted that a claim outside `all_claim_dirs()`
    was invisible to the sweep, which on exactly that box was false and
    sent the one reader already debugging a kill after the wrong
    divergence. Chasing the wrong divergence is the whole of issue #19.

    The same list `_swept_dirs` builds, spelled for the log line: what
    spares a process and what gets swept must not be two lists that can
    drift apart.
    """
    return [str(d) for d in _swept_dirs(attributed_from)]


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
    # One directory list for the sweep and for the kill line below, so a
    # ledger that can grant an exemption is a ledger something sweeps.
    claims = cfg.claim_dir if cfg.claim_dir else claim_dir()
    stale, stuck = sweep_stale(_swept_dirs(claims))
    requeued, failed = requeue_orphans(queue, active_ids)
    killed, convicted = [], []
    # Records that claim a scope which no longer holds -- the anchor
    # died, or the container restarted and got a fresh scope id. Reported
    # beside `stale_claims` and `stuck_claims` because a claim that has
    # quietly stopped covering anything is the same class of silent
    # failure issue #24 is about.
    #
    # None, not `[]`: "not measured this tick" is a different fact from
    # "measured, and there are none", and only the timer-gated branch
    # below measures. `runner._reap` runs on every tick and change-gates
    # its warning on the previous tick's set, so an empty list here would
    # clear that memo on every non-sweep tick and the next sweep would
    # report every void scope as new -- once per `orphan_cuda_interval_s`,
    # forever, which is exactly what the gating exists to prevent.
    # `stuck_claims` needs no such distinction only because `sweep_stale`
    # runs unconditionally at the top of this function.
    void_scopes: list[str] | None = None
    # Where a claim would have spared a process from `kill_orphan_cuda`,
    # for the log line that follows a kill. See `_consulted_dirs`: both
    # the exemption ledgers and the one `attribute` read, because both
    # spare by the same amount and the operator is asking about the
    # outcome, not the mechanism.
    #
    # Populated only where it is true: the sweep also runs for
    # `enforce_vram`, which consults no exemption at all, and a sweep that
    # cannot see the process list examined nothing. Naming ledgers on
    # either would point the next operator at a read that never happened.
    exemption_dirs: list[str] = []
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
            records = ledger.live_records(ledger.all_records(claims))
            # Gated on `scope_cgroup`, not `scope_pid`: the condition being
            # reported is "this record claims a scope, and the scope no
            # longer holds" -- a record with no scope at all was never
            # making that claim and has nothing to go void.
            void_scopes = [str(r.path) for r in records
                          if r.scope_cgroup is not None
                          and not ledger.scope_is_live(r)]
            if cfg.kill_orphan_cuda:
                protect = _running_trees(queue)
                # What the log names, so it names what was consulted
                # rather than what we assume it consulted.
                exemption_dirs = _consulted_dirs(claims)
                killed = kill_orphan_cuda(protect, records, apps)
                # Written here rather than in `kill_orphan_cuda` because
                # that function does not know the queue root, and giving
                # it one would tie the kill decision to the queue's
                # layout. `exemption_dirs` is already in hand on this
                # line, which is the whole reason the record can name
                # what was consulted.
                killlog.append(queue.root, killed, exemption_dirs)
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
    return {"stale_claims": stale, "stuck_claims": stuck,
            "requeued": requeued, "failed": failed,
            # Two views of one list so they cannot drift. `killed_pids`
            # is what `runner.py` has always logged and what the suite
            # asserts on; `killed_details` is what the kill record needs.
            "killed_pids": [d["pid"] for d in killed],
            "killed_details": killed,
            "cleaned_paths": cleaned,
            "convicted": convicted, "exemption_dirs": exemption_dirs,
            "void_scopes": void_scopes}
