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
from .procs import pid_alive
from .spec import utcnow_iso

# Held back from admission. Two processes that each fit exactly still have
# their allocators fragmenting the same heap.
DEFAULT_RESERVE_MB = 512


class ClaimBusy(RuntimeError):
    """The card has no room for this claim."""


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
         usable_mb: int | None) -> bool:
    """`usable_mb is None` means the card could not be queried, so every
    claim is treated as exclusive -- degraded, and the same posture
    preflight already takes when it cannot enumerate the card."""
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
                 usable_mb: int | None) -> str:
    want = "the whole card" if want_mb is None else f"{want_mb} MiB"
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
        except OSError:
            if time.monotonic() >= deadline:
                raise ClaimBusy(
                    f"could not take the ledger mutex within {timeout_s:g}s: "
                    "an older gpu-claim is holding this card exclusively for "
                    "the whole of its run. Wait for it, or upgrade it.")
            time.sleep(_MUTEX_POLL_S)


def acquire(key: str, *, vram_mb: int | None, owner: str,
            cmd: list[str] | None, directory, usable_mb: int | None,
            usage_pid: int | None = None) -> Record:
    """Take a share of the card, or raise ClaimBusy.

    Non-blocking on capacity by design: the caller decides whether to wait,
    and the runner must not, because a single-threaded loop that waits here
    stalls the CPU lane behind whoever holds the card.
    """
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    ldir = ledger_dir(key, d)
    ldir.mkdir(parents=True, exist_ok=True)

    fd = os.open(mutex_path(key, d), os.O_CREAT | os.O_RDWR, 0o666)
    try:
        _take_mutex(fd, MUTEX_WAIT_S)
        live = live_records(records_for(key, d))
        if not fits(live, vram_mb, usable_mb):
            raise ClaimBusy(busy_message(key, live, vram_mb, usable_mb))
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
