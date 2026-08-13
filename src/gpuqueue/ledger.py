"""The claim ledger: who holds this card, and for how much VRAM.

`flock` cannot be a counting semaphore, so it changes role. `<key>.lock`
is taken only for the milliseconds needed to read the holders, decide, and
write a record -- it guards the accounting, not the card. Holders live one
file per holder under `<key>.lock.d/`, which is what keeps `ls` able to
show who is on the card and `rm` able to clear one wedged holder. A single
mutated document would give both up exactly when something is stuck, since
a torn write blinds every participant at once.

`vram_mb = None` means exclusive: the whole card. It fits only into an
empty ledger, and nothing fits alongside it. That one rule is what makes
an undeclared claim behave as it did before any of this existed.
"""
from __future__ import annotations

import fcntl
import json
import os
import secrets
import time
from dataclasses import dataclass
from pathlib import Path

from .gpuid import lock_filename
from .procs import descendants, pid_alive
from .spec import utcnow_iso

# Held back from admission. Two processes that each fit exactly still have
# their allocators fragmenting the same heap.
DEFAULT_RESERVE_MB = 512


class ClaimBusy(RuntimeError):
    """The card has no room for this claim."""


class MutexTimeout(ClaimBusy):
    """Could not take the ledger mutex itself -- not a capacity refusal.

    A subclass of `ClaimBusy`, not a sibling: every existing `except
    ClaimBusy` (the runner, `gpu-claim`, callers not yet written) must
    keep catching this without being told about it. What a caller who
    *does* care about the distinction gets is the option to add a more
    specific `except MutexTimeout` before the general one -- which is
    exactly what `gpu_claim`'s `wait=True` path does, since "an old
    gpu-claim is holding LOCK_EX for its whole run" can last hours and
    is worth explaining, where "the card is full" is not.
    """


class CannotEverFit(ClaimBusy):
    """The declaration exceeds the whole card -- permanent, not busy.

    A subclass for the same reason `MutexTimeout` is one: every existing
    `except ClaimBusy` must keep catching it. The distinction it offers a
    caller who asks for it is permanence. "Busy" invites waiting, and every
    waiter here polls for room to appear; room for a claim larger than the
    card never does, so a caller told only "busy" hangs forever. `gpu_claim`
    raises this before its wait loop rather than inside it, and `gpu-claim`
    reports it as unavailable (69) rather than a temporary failure (75).
    """


@dataclass
class Record:
    path: Path
    pid: int              # whose liveness governs this record
    usage_pid: int | None  # whose process tree is charged to it
    vram_mb: int | None    # None = exclusive
    owner: str
    cmd: list[str]
    started_at: str
    key: str

    @property
    def name(self) -> str:
        return self.path.name

    def to_dict(self) -> dict:
        return {"pid": self.pid, "usage_pid": self.usage_pid,
                "vram_mb": self.vram_mb, "owner": self.owner,
                "cmd": self.cmd, "started_at": self.started_at,
                "key": self.key}


def mutex_path(key: str, directory) -> Path:
    return Path(directory) / lock_filename(key)


def ledger_dir(key: str, directory) -> Path:
    return Path(str(mutex_path(key, directory)) + ".d")


def _legacy_path(key: str, directory) -> Path:
    return Path(str(mutex_path(key, directory)) + ".json")


def _load(path: Path) -> Record | None:
    try:
        d = json.loads(path.read_text())
        return Record(
            path=path, pid=int(d["pid"]),
            usage_pid=(int(d["usage_pid"])
                       if d.get("usage_pid") is not None else None),
            vram_mb=(int(d["vram_mb"])
                     if d.get("vram_mb") is not None else None),
            owner=d.get("owner", "?"), cmd=list(d.get("cmd") or []),
            started_at=d.get("started_at", ""), key=d.get("key", ""))
    except Exception:
        return None  # a garbage record must not blind us to the good ones


def _load_legacy(path: Path) -> Record | None:
    """A `<key>.lock.json` from a gpu-claim that predates the ledger.

    It took the whole card and the process on the card is normally the pid
    it recorded. Reading it as exclusive is what stops the reaper treating
    an old holder's trainer as unledgered and killing it mid-upgrade.
    """
    rec = _load(path)
    if rec is None:
        return None
    rec.vram_mb = None
    if rec.usage_pid is None:
        rec.usage_pid = rec.pid
    return rec


def records_for(key: str, directory) -> list[Record]:
    d = Path(directory)
    out = []
    ldir = ledger_dir(key, d)
    if ldir.is_dir():
        out.extend(r for r in (_load(p) for p in sorted(ldir.glob("*.json")))
                   if r is not None)
    legacy = _load_legacy(_legacy_path(key, d))
    if legacy is not None:
        out.append(legacy)
    return out


def all_records(directory) -> list[Record]:
    d = Path(directory)
    if not d.is_dir():
        return []
    out = []
    for ldir in sorted(d.glob("*.lock.d")):
        out.extend(r for r in (_load(p) for p in sorted(ldir.glob("*.json")))
                   if r is not None)
    for p in sorted(d.glob("*.lock.json")):
        legacy = _load_legacy(p)
        if legacy is not None:
            out.append(legacy)
    return out


def live_records(records: list[Record]) -> list[Record]:
    return [r for r in records if pid_alive(r.pid)]


def write_record(rec: Record) -> None:
    """Atomic: preflight and the reaper read records without the mutex, so
    a half-written file must never be observable."""
    rec.path.parent.mkdir(parents=True, exist_ok=True)
    tmp = rec.path.with_suffix(".json.part")
    tmp.write_text(json.dumps(rec.to_dict(), indent=2) + "\n")
    os.replace(tmp, rec.path)


def set_usage_pid(rec: Record, pid: int | None) -> None:
    rec.usage_pid = pid
    write_record(rec)


def remove(rec: Record) -> None:
    rec.path.unlink(missing_ok=True)


def exceeds_capacity(want_mb: int | None, usable_mb: int | None) -> bool:
    """A declaration that can never be admitted, however empty the card.

    The runner needs this apart from `fits` so it can fail such a job
    instead of leaving it pending forever.
    """
    if want_mb is None or usable_mb is None:
        return False
    return want_mb > usable_mb


def fits(records: list[Record], want_mb: int | None,
         usable_mb: int | None, max_holders: int | None = None) -> bool:
    """`usable_mb is None` means the card could not be queried, so every
    claim is treated as exclusive -- degraded, and the same posture
    preflight already takes when it cannot enumerate the card.

    `max_holders` is the latency budget, and it is checked here rather than
    only in `Runner._capacity` because it is a property of the card. VRAM
    accounting alone admits sixteen 500 MiB claims onto an 8 GB card, all
    time-slicing; a cap the runner applies to its own lane leaves that
    reachable by hand, and with independent submitters the cost lands on a
    stranger. None means uncapped, which is what a caller that has no
    policy to read should pass rather than guessing a number.
    """
    if max_holders is not None and len(records) >= max_holders:
        return False
    if any(r.vram_mb is None for r in records):
        return False
    if want_mb is None or usable_mb is None:
        return not records
    if want_mb > usable_mb:
        return False
    return sum(r.vram_mb or 0 for r in records) + want_mb <= usable_mb


def free_mb(records: list[Record], usable_mb: int | None) -> int:
    if usable_mb is None or any(r.vram_mb is None for r in records):
        return 0
    return max(0, usable_mb - sum(r.vram_mb or 0 for r in records))


def busy_message(key: str, records: list[Record], want_mb: int | None,
                 usable_mb: int | None, max_holders: int | None = None) -> str:
    want = "the whole card" if want_mb is None else f"{want_mb} MiB"
    if max_holders is not None and len(records) >= max_holders:
        # Named separately because the VRAM wording is actively misleading
        # here: refused for the count with 7 GB of the card free, "need 500
        # MiB, 6676 MiB free" sends the reader hunting for a memory problem
        # that does not exist. This is a queueing decision, not a capacity
        # one, and it clears when a holder exits rather than when memory
        # frees up.
        head = (f"GPU {key}: at the {max_holders}-job limit for this card "
                f"(gpu_max_jobs), with {len(records)} on it. Holders:")
    else:
        head = (f"GPU {key}: need {want}, "
                f"{free_mb(records, usable_mb)} MiB free"
                + (f" of {usable_mb}" if usable_mb is not None else "")
                + ". Holders:")
    lines = [
        f"  pid {r.pid:>7}  {r.owner:<24} "
        f"{'exclusive' if r.vram_mb is None else str(r.vram_mb) + ' MiB':>12}"
        f"  {' '.join(r.cmd) or '?'}"
        for r in records
    ] or ["  (none -- this claim does not fit the card at all)"]
    return "\n".join([head, *lines])


# Bound on the mutex wait, not on holding the card. A participant only
# ever holds this flock for a directory read and one rename, so anything
# longer means the holder predates this ledger entirely.
MUTEX_WAIT_S = 10.0
_MUTEX_POLL_S = 0.05


def _take_mutex(fd: int, timeout_s: float) -> None:
    """Bounded, because a participant only holds this for a directory read
    and one rename. Waiting longer than that means the holder is not
    playing by these rules -- in practice a gpu-claim from before the
    ledger, which takes LOCK_EX for the whole run and would hang us until
    its training finished."""
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            # Only "someone else holds it". A bare `except OSError` also
            # swallowed EBADF, ENOLCK and EINTR and reported them below as
            # "an older gpu-claim is holding this card -- wait for it",
            # sending an operator to wait hours on a condition that will
            # never clear. Those are bugs or a broken filesystem and must
            # come out as themselves.
            if time.monotonic() >= deadline:
                raise MutexTimeout(
                    f"could not take the ledger mutex within {timeout_s:g}s: "
                    "an older gpu-claim is holding this card exclusively for "
                    "the whole of its run. Wait for it, or upgrade it.")
            time.sleep(_MUTEX_POLL_S)


def acquire(key: str, *, vram_mb: int | None, owner: str,
            cmd: list[str] | None, directory, usable_mb: int | None,
            usage_pid: int | None = None,
            max_holders: int | None = None) -> Record:
    """Take a share of the card, or raise ClaimBusy.

    Non-blocking on capacity by design: the caller decides whether to wait,
    and the runner must not, because a single-threaded loop that waits here
    stalls the CPU lane behind whoever holds the card.

    `max_holders` is counted over the same `live_records` the VRAM is summed
    over, so a wedged holder whose pid is gone does not spend a slot -- it
    already does not spend its VRAM.
    """
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    ldir = ledger_dir(key, d)
    ldir.mkdir(parents=True, exist_ok=True)

    fd = os.open(mutex_path(key, d), os.O_CREAT | os.O_RDWR, 0o666)
    try:
        _take_mutex(fd, MUTEX_WAIT_S)
        live = live_records(records_for(key, d))
        if not fits(live, vram_mb, usable_mb, max_holders):
            raise ClaimBusy(busy_message(key, live, vram_mb, usable_mb,
                                         max_holders))
        # A token, not just the pid: the runner holds one record per GPU
        # job and every one of them carries the runner's pid.
        rec = Record(path=ldir / f"{os.getpid()}.{secrets.token_hex(3)}.json",
                     pid=os.getpid(), usage_pid=usage_pid, vram_mb=vram_mb,
                     owner=owner, cmd=list(cmd or []),
                     started_at=utcnow_iso(), key=key)
        write_record(rec)
        return rec
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def attribute(apps: list[dict],
              records: list[Record]) -> tuple[dict[str, list[dict]], list[dict]]:
    """Charge every visible CUDA process to the record that owns it.

    Returns (owned, unledgered), keyed by the record's full path. Three
    callers need this answer -- preflight, the orphan reaper and the VRAM
    watchdog -- and they share one implementation so they cannot disagree
    about who owns a pid, which is the disagreement that gets a
    legitimate job killed.

    Keyed by `str(path)` rather than `Record.name`: `all_records()` spans
    every `<key>.lock.d/` directory, so a bare filename is unique only
    within one directory, not across the whole ledger. Keying by name
    would let a second record's tree silently overwrite a first's, and
    the first holder's live process would then read as a stranger.

    A record with no `usage_pid` has been admitted but not launched. It
    owns nothing, and must not adopt a stranger's process.
    """
    # `records` is not filtered on `usage_pid` liveness, and need not be:
    # for gpu-claim and legacy records `pid == usage_pid`, so `live_records`
    # has already dropped them; for a runner record `_settle` removes the
    # record in the same tick the process is reaped, so a stale one can
    # accrue at most one of the two strikes a conviction needs.
    trees = {str(r.path): {r.usage_pid} | descendants(r.usage_pid)
             for r in records if r.usage_pid is not None}
    owned: dict[str, list[dict]] = {}
    unledgered: list[dict] = []
    for app in apps:
        # First match wins. If two records' trees overlap (a holder that
        # forked another holder), the process is charged to whichever
        # record sorts first and the other reads `used=0` -- under its
        # declaration, so the overlap can only fail to convict, never
        # convict the wrong holder.
        for path, tree in trees.items():
            if app["pid"] in tree:
                owned.setdefault(path, []).append(app)
                break
        else:
            unledgered.append(app)
    return owned, unledgered


def used_mb(apps: list[dict]) -> int:
    """nvidia-smi reports [N/A] for a process it can see but not measure;
    counting that as zero under-reports, which is the safe direction for a
    watchdog that kills."""
    return sum(a.get("used_mb") or 0 for a in apps)
