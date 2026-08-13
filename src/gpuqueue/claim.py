"""The advisory lock and its claim record.

Enforcement is advisory because flock cannot be otherwise between
unprivileged processes. The claim record exists so that a human or agent
looking at a busy card learns *who* holds it without a running service.
"""
from __future__ import annotations

import getpass
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from . import config, ledger
from .gpuid import gpu_key, total_vram_mb
# re-exported: callers use these
from .ledger import ClaimBusy, MutexTimeout, CannotEverFit
from .procs import pid_alive           # re-exported for the same reason

DEFAULT_CLAIM_DIR = "/var/lock/gpu"
WAIT_POLL_S = 0.5

# `usable_mb` omitted, as distinct from an explicit None. See gpu_claim.
_ASK_THE_CARD = object()

# The same distinction for `max_holders`: omitted means "read the policy",
# an explicit None means "no cap" and must survive being passed in.
_ASK_THE_CONFIG = object()

# A MutexTimeout means an old-style gpu-claim is holding LOCK_EX for its
# whole run, which can be hours -- a different condition from "the card is
# full," which clears in the time a job takes to exit. Retrying at the
# capacity cadence would mean polling flock at _MUTEX_POLL_S under the
# hood every WAIT_POLL_S, for as long as that run lasts. Its own, longer
# interval is what keeps that from being a busy-wait.
MUTEX_WAIT_POLL_S = 5.0


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


def default_usable_mb(index: int = 0) -> int | None:
    """What a standalone `gpu-claim` may admit against.

    The runner's answer to the same question about the same card, which is
    the point: the two of them share one `<key>.lock.d`, and a capacity
    they disagree about is a card they will double-book. So this reads
    `[queue].gpu_vram_mb`/`gpu_vram_reserve_mb` for the same reason
    `Runner._usable_mb` does -- see `config.vram_policy`, which explains
    why it does not go through `load_config`.

    `index` follows `gpu-claim --gpu-index`, which keys the ledger on that
    card and pins the child to it; sizing every claim against card 0 admits
    a 16 GB claim onto the 8 GB card beside it. A declared `gpu_vram_mb`
    describes the card the runner manages -- card 0, the only one
    `gpu_key()` ever names -- so it is not applied to any other index. The
    reserve is a headroom policy rather than a fact about one card, so it
    applies to all of them.

    None when the card cannot be queried, which `ledger.fits` turns into
    exclusive-only admission -- degraded, and the same posture preflight
    already takes when it cannot enumerate the card. Also None when the
    card's total is smaller than the reserve: an uncapped subtraction
    would go negative, and `want_mb > usable_mb` in
    `ledger.fits`/`exceeds_capacity` is then true for every claim,
    silently admitting nothing instead of degrading the same way a
    query failure does.
    """
    declared, reserve = config.vram_policy()
    total = declared if declared is not None and index == 0 else \
        total_vram_mb(index)
    if total is None:
        return None
    usable = total - reserve
    return usable if usable > 0 else None


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


def _refuse_if_too_big(key: str, vram_mb: int | None,
                       usable_mb: int | None) -> None:
    """A declaration larger than the whole card, refused rather than waited
    on: every waiter in `gpu_claim` polls for room to appear, and room for
    a claim this size never does. `_take_card` makes the same call for the
    same reason."""
    if ledger.exceeds_capacity(vram_mb, usable_mb):
        raise ledger.CannotEverFit(
            f"GPU {key}: declared {vram_mb} MiB but only {usable_mb} MiB is "
            "usable on this card; it can never be admitted")


@contextmanager
def gpu_claim(key: str | None = None, owner: str | None = None,
              cmd: list[str] | None = None, wait: bool = False,
              directory: Path | None = None, vram_mb: int | None = None,
              usable_mb: int | None | object = _ASK_THE_CARD,
              own_usage: bool = True,
              max_holders: int | None | object = _ASK_THE_CONFIG):
    """Hold a share of the card. `vram_mb=None` means the whole of it.

    `own_usage=False` is for the runner, which takes the card before the
    job process exists and fills the usage pid in after launch.

    `usable_mb` distinguishes three cases, which is why its default is a
    sentinel rather than None. Omitted means "you work out the capacity"
    and we ask the card. An int is that capacity. None means "I asked and
    the card would not say" -- `ledger.fits` reads that as exclusive-only
    admission, and it must survive being passed in. The runner hands us
    `Runner._usable_mb()` straight through, so conflating its None with
    "omitted" would have the runner logging exclusive-only while this
    function went on sharing the card against a rediscovered capacity that
    ignores the configured reserve.

    `max_holders` is the card's latency budget, `[queue].gpu_max_jobs`.
    Omitted, it is read from the config for the same reason `usable_mb` is
    -- this participant shares one `<key>.lock.d` with the runner, and a
    cap only one of them applies is not a cap on the card. The runner
    passes its own loaded value straight through instead, so it does not
    re-read the file once per admit.

    `wait` polls capacity rather than blocking on flock: the mutex is
    released the instant `acquire` returns, so there is no longer a kernel
    queue to wait in.

    A `MutexTimeout` -- an old-style gpu-claim holding the mutex itself,
    not the card being full -- is a different wait with a different
    cause, so it gets its own cadence and a one-time explanation instead
    of silently retrying at the capacity rate for however long that run
    lasts.
    """
    # The same rule `JobSpec.validate` applies on the submit path, applied
    # here because this is the other way into the ledger. A declaration
    # that is not a positive int does not merely fail to be admitted: it is
    # *summed* by `ledger.fits`, so a holder that declared -5000 subtracts
    # 5 GB from the accounted total and the next claimant is admitted past
    # the end of the card. Zero is admitted without limit, which an
    # unbounded number of co-tenants reaches the same way.
    if vram_mb is not None and (not isinstance(vram_mb, int)
                                or isinstance(vram_mb, bool) or vram_mb <= 0):
        raise ValueError(
            f"vram_mb must be a positive int or None (meaning the whole "
            f"card), got {vram_mb!r}")
    d = Path(directory) if directory else claim_dir()
    key = key or gpu_key()
    ours_to_ask = usable_mb is _ASK_THE_CARD
    if ours_to_ask:
        usable_mb = default_usable_mb()
    if max_holders is _ASK_THE_CONFIG:
        max_holders = config.max_holders()
    _refuse_if_too_big(key, vram_mb, usable_mb)
    warned = False
    while True:
        try:
            rec = ledger.acquire(
                key, vram_mb=vram_mb, owner=owner or _default_owner(),
                cmd=cmd, directory=d, usable_mb=usable_mb,
                usage_pid=os.getpid() if own_usage else None,
                max_holders=max_holders)
            break
        except MutexTimeout:
            if not wait:
                raise
            if not warned:
                print("gpu-claim: warning: an older gpu-claim is holding "
                      "this card exclusively; this may last as long as its "
                      "run", file=sys.stderr)
                warned = True
            time.sleep(MUTEX_WAIT_POLL_S)
        except ClaimBusy:
            if not wait:
                raise
            time.sleep(WAIT_POLL_S)
        if ours_to_ask and usable_mb is None:
            # A capacity query that failed is not latched for the whole
            # wait. `default_usable_mb` returns None for a card that would
            # not answer, and `ledger.fits` reads that as exclusive-only:
            # the waiter then polls until the card is completely empty,
            # which on a shared box is hours, over a subprocess hiccup that
            # cleared on the next poll. `Runner._usable_mb` declines to
            # cache a failed query for exactly this reason; a wait loop
            # that asked once, before the loop, latched it just as hard.
            #
            # Only when we were the ones who asked: an explicit None from
            # the caller means "I asked and the card would not say", and
            # re-deriving a capacity that ignores their configured reserve
            # is the double-booking the sentinel exists to prevent.
            usable_mb = default_usable_mb()
            _refuse_if_too_big(key, vram_mb, usable_mb)
    try:
        yield rec
    finally:
        ledger.remove(rec)


def _default_owner() -> str:
    try:
        return getpass.getuser()
    except Exception:
        return f"uid{os.getuid()}"
