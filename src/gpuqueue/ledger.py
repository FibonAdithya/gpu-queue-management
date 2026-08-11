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

import json
import os
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
            usage_pid=int(d["usage_pid"]) if d.get("usage_pid") else None,
            vram_mb=int(d["vram_mb"]) if d.get("vram_mb") else None,
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
