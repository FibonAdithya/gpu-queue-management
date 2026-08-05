"""File a bug against gpuq on GitHub, deduplicated and throttled.

Everything with a side effect lives here; the taxonomy lives in
bugreport.py. The one thing this module must never do is fail loudly: a box
with no `gh`, no token or no network runs jobs exactly as it does without
autofix, so callers go through Runner._report_bug, which swallows GhError.

The token this reaches for is scoped to `issues: write` on this repository
and explicitly not `contents`. Nothing here pushes; its worst case is issue
spam.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .config import AutofixConfig

log = logging.getLogger("gpuqueue.bugfiler")

GH_TIMEOUT_S = 30

# The runner applies these; the workflow dispatches on them.
AUTO_LABEL = "gpuq-auto"           # structural evidence, dispatches at once
REPORTED_LABEL = "gpuq-reported"   # agent prose, waits for `fix-me`
THROTTLED_LABEL = "throttled"      # filed as evidence, deliberately not run


class GhError(RuntimeError):
    """The gh CLI was missing, unauthorised, or refused."""


def _gh(cfg: AutofixConfig, args: list[str], stdin: str | None = None) -> str:
    """The single subprocess call site, and the seam the tests replace.

    The PAT goes in through the environment. argv is world-readable in
    /proc, and a token in a process listing is a token on a shared box.
    """
    env = dict(os.environ)
    token = os.environ.get(cfg.token_env)
    if token:
        env["GH_TOKEN"] = token
    try:
        proc = subprocess.run(["gh", *args], input=stdin, env=env, text=True,
                              capture_output=True, timeout=GH_TIMEOUT_S)
    except OSError as e:
        # FileNotFoundError, PermissionError and their OSError siblings all
        # mean the same thing to a caller: gh did not run. Keep the message
        # honest about which -- "not installed" and "not executable" call
        # for different fixes from whoever reads the runner log.
        raise GhError(f"cannot run gh: {e}") from e
    except subprocess.TimeoutExpired as e:
        raise GhError(f"gh {args[0]} timed out after {GH_TIMEOUT_S}s") from e
    if proc.returncode != 0:
        raise GhError(f"gh {' '.join(args)} failed ({proc.returncode}): "
                      f"{proc.stderr.strip()}")
    return proc.stdout


def _search(cfg: AutofixConfig, kind: str, state: str, query: str,
            fields: str = "number,body") -> list[dict]:
    out = _gh(cfg, [kind, "list", "--repo", cfg.repo, "--state", state,
                    "--search", query, "--json", fields, "--limit", "10"])
    try:
        return json.loads(out or "[]")
    except json.JSONDecodeError:
        return []


def _exact(rows: list[dict], sig: str) -> list[dict]:
    """gh's search is full text and returns near misses. Match the literal
    line we wrote, or two unrelated bugs end up sharing one issue."""
    return [r for r in rows if f"sig: {sig}" in (r.get("body") or "")]


def find_open_issue(cfg: AutofixConfig, sig: str) -> dict | None:
    rows = _exact(_search(cfg, "issue", "open", f"sig: {sig} in:body"), sig)
    return rows[0] if rows else None


def find_open_pr(cfg: AutofixConfig, sig: str) -> int | None:
    """A bug that fails every job must not spawn one fix run per job. This
    is the lookup that prevents that."""
    rows = _exact(_search(cfg, "pr", "open", f"{sig} in:body"), sig)
    return rows[0]["number"] if rows else None


def find_recent_closed(cfg: AutofixConfig, sig: str) -> int | None:
    """A previous fix that did not hold is the most useful context the next
    attempt can have, and it costs nothing to pass on."""
    since = (datetime.now(timezone.utc)
             - timedelta(days=cfg.closed_lookback_days)).strftime("%Y-%m-%d")
    rows = _exact(_search(cfg, "issue", "closed",
                          f"sig: {sig} in:body closed:>={since}"), sig)
    return rows[0]["number"] if rows else None


def gpuq_commit() -> str:
    """Which gpuq the box is actually running. 'unknown' off a git checkout."""
    repo = Path(__file__).resolve().parents[2]
    try:
        proc = subprocess.run(["git", "-C", str(repo), "rev-parse",
                               "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10)
    except Exception:
        return "unknown"
    return proc.stdout.strip() if proc.returncode == 0 else "unknown"


_PRUNE_AFTER = timedelta(days=2)


def _load_dispatches(cfg: AutofixConfig) -> list[datetime]:
    """Timestamps of past auto-dispatches. Unreadable means empty.

    This file guards a budget, not queue state. Refusing to file because we
    cannot read it would lose evidence to protect a counter, which is the
    wrong way round.
    """
    try:
        raw = json.loads(Path(cfg.state_file).read_text())["dispatches"]
    except Exception:
        return []
    out = []
    for stamp in raw:
        try:
            out.append(datetime.fromisoformat(stamp))
        except (TypeError, ValueError):
            continue
    return out


def _save_dispatches(cfg: AutofixConfig, stamps: list[datetime]) -> None:
    path = Path(cfg.state_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.part")
    tmp.write_text(json.dumps(
        {"dispatches": [s.isoformat() for s in stamps]}, indent=2) + "\n")
    os.rename(tmp, path)


def dispatches_in_last_day(cfg: AutofixConfig, now: datetime) -> int:
    cutoff = now - timedelta(days=1)
    return sum(1 for s in _load_dispatches(cfg) if s > cutoff)


def may_dispatch(cfg: AutofixConfig, now: datetime) -> bool:
    return dispatches_in_last_day(cfg, now) < cfg.max_dispatches_per_day


def record_dispatch(cfg: AutofixConfig, now: datetime) -> None:
    kept = [s for s in _load_dispatches(cfg) if s > now - _PRUNE_AFTER]
    _save_dispatches(cfg, [*kept, now])
