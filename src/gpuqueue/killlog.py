"""What the orphan sweep killed, where its victim's operator can read it.

The runner log already names every kill. That was not enough: it is on
the box, the killed process cannot read it, and its owner does not think
to check it -- because from their side the failure looks like their own
crash (issue #24). This is the same fact in a place `gpuq kills` can
print, so an agent that sees `killed by signal 9` has something to run.
"""
from __future__ import annotations

import json
from pathlib import Path

from .spec import utcnow_iso

KILLS_FILENAME = "kills.jsonl"

# Kills are rare and each line is small, but "rare" is not "bounded": a
# box left up for months with a misconfigured claim directory writes one
# per sweep forever. Keeping the newest N is what makes this a record
# rather than a slow leak.
MAX_ENTRIES = 1000


def _path(queue_root) -> Path:
    return Path(queue_root) / KILLS_FILENAME


def append(queue_root, entries: list[dict], consulted: list[str]) -> None:
    """Record a sweep's kills. A sweep that killed nothing writes nothing.

    `consulted` is the ledger list the sweep built its exemptions from --
    `reaper._consulted_dirs`, the same list the log line names. It is the
    first thing to check when a kill looks wrong, because a claim written
    outside those directories is invisible to the sweep and that is
    exactly what issue #19 was.
    """
    if not entries:
        return
    ts = utcnow_iso()
    lines = [json.dumps({**e, "ts": ts,
                         "reason": "orphan_sweep_unledgered",
                         "ledgers_consulted": list(consulted)})
             for e in entries]
    p = _path(queue_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    existing = p.read_text().splitlines() if p.exists() else []
    kept = (existing + lines)[-MAX_ENTRIES:]
    tmp = p.with_suffix(".jsonl.part")
    tmp.write_text("\n".join(kept) + "\n")
    tmp.replace(p)


def _read_all(queue_root) -> list[dict]:
    """Every recorded kill, oldest first, one disk read. A corrupt line is
    skipped, not fatal.

    Same posture as `ledger._load`: whoever is reading this is mid-
    incident, and one bad line must not hide the record they came for.
    """
    p = _path(queue_root)
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def _apply_limit(out: list[dict], limit: int | None) -> list[dict]:
    if limit is None:
        return out
    # Not `out[-limit:] if limit else out`: `--limit 0` falls to the
    # falsy branch and prints everything, and `out[-0:]` is the whole
    # list anyway, so even an `is not None` guard reads zero as "no
    # limit". This is the one place that contract is decided; every
    # caller -- `read` and `read_with_total` alike -- goes through it
    # rather than re-deciding it.
    return out[-limit:] if limit > 0 else []


def read(queue_root, limit: int | None = None) -> list[dict]:
    """Recorded kills, oldest first, limited per `_apply_limit`."""
    return _apply_limit(_read_all(queue_root), limit)


def read_with_total(queue_root,
                     limit: int | None = None) -> tuple[list[dict], int]:
    """`(limited entries, total recorded)` from a single disk read.

    `append` swaps the file atomically (`tmp.replace`), so two separate
    reads -- one for the shown entries, one for the total -- can straddle
    a concurrent sweep's append and disagree about what "total" means.
    A caller reporting a truncation window (`gpuq kills`'s "showing the
    most recent N of M") needs N and M to come from the same snapshot,
    or the window it describes may never have existed.
    """
    out = _read_all(queue_root)
    return _apply_limit(out, limit), len(out)
