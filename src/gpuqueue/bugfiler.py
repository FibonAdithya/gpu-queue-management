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
import re
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .bugreport import (build_report, bump_body, is_gpuq_fault, issue_body,
                        issue_title)
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
    """Timestamps of past auto-dispatches. Unreadable or malformed means
    empty.

    This file guards a budget, not queue state. Refusing to file because we
    cannot read it would lose evidence to protect a counter, which is the
    wrong way round -- so every failure here, including a `dispatches` that
    isn't a list at all (a hand-edited file, a future writer's bug), must
    fail towards zero recorded dispatches rather than raise.
    """
    try:
        raw = json.loads(Path(cfg.state_file).read_text())["dispatches"]
        if not isinstance(raw, list):
            return []
    except Exception:
        return []
    out = []
    for stamp in raw:
        try:
            when = datetime.fromisoformat(stamp)
        except (TypeError, ValueError):
            continue
        if when.tzinfo is None:
            # A naive timestamp -- hand-edited, or a future writer that
            # passed a naive `now` -- must not blow up the aware/naive
            # comparison in dispatches_in_last_day. Assume UTC, since that
            # is the only zone _save_dispatches ever writes.
            when = when.replace(tzinfo=timezone.utc)
        out.append(when)
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


_ISSUE_NUMBER = re.compile(r"/issues/(\d+)")

# Names of labels confirmed present, tracked one at a time rather than one
# all-or-nothing flag. `gh issue create --label` fails outright on a label
# that does not exist, and a transient failure on a single label (an auth
# hiccup, a rate limit) must not be remembered as success for the whole
# set -- otherwise every later file_bug/file_agent_report in this process's
# life raises GhError on a label gh never actually created, and since the
# runner swallows GhError, filing silently stops for good. Only genuinely
# confirmed labels are skipped; anything that failed is retried next call.
_labels_ensured: set[str] = set()

_LABEL_SPECS = (
    (AUTO_LABEL, "B60205", "filed by the gpuq runner from its own traceback"),
    (REPORTED_LABEL, "FBCA04", "filed by an agent; needs `fix-me` to run"),
    (THROTTLED_LABEL, "666666", "filed as evidence; deliberately not run"),
    ("fix-me", "0E8A16", "owner authorises an autofix run"),
)


def ensure_labels(cfg: AutofixConfig) -> None:
    """Make the labels this module needs once per process rather than
    documenting a manual step a rebuilt box will skip."""
    for name, colour, desc in _LABEL_SPECS:
        if name in _labels_ensured:
            continue
        try:
            _gh(cfg, ["label", "create", name, "--repo", cfg.repo,
                      "--color", colour, "--description", desc, "--force"])
        except GhError as e:
            log.warning("could not ensure label %s: %s", name, e)
        else:
            _labels_ensured.add(name)


def _create_issue(cfg: AutofixConfig, title: str, body: str,
                  label: str) -> int | None:
    ensure_labels(cfg)
    out = _gh(cfg, ["issue", "create", "--repo", cfg.repo, "--title", title,
                    "--label", label, "--body-file", "-"], stdin=body)
    match = _ISSUE_NUMBER.search(out or "")
    return int(match.group(1)) if match else None


def file_bug(cfg: AutofixConfig, exc: BaseException, phase: str, *,
             spec=None, queue_counts: dict[str, int] | None = None,
             now: datetime | None = None) -> str:
    """File, comment or stay silent. Returns what it did.

    Raises GhError if GitHub is unreachable; the runner's _report_bug is
    what makes that harmless.
    """
    if not cfg.enabled or not cfg.repo:
        return "disabled"
    if not is_gpuq_fault(exc):
        return "caller-fault"

    now = now or datetime.now(timezone.utc)
    report = build_report(exc, phase, spec=spec, queue_counts=queue_counts,
                          gpuq_commit=gpuq_commit(),
                          occurred_at=now.strftime("%Y-%m-%dT%H:%M:%SZ"))

    # 1. an open issue with this signature: record the recurrence, run nothing.
    existing = find_open_issue(cfg, report.sig)
    if existing:
        _gh(cfg, ["issue", "comment", str(existing["number"]),
                  "--repo", cfg.repo, "--body-file", "-"],
            stdin=f"Seen again at {report.occurred_at} "
                  f"(job {report.job['id'] if report.job else 'n/a'}).")
        bumped = bump_body(existing.get("body") or "", report.occurred_at)
        if bumped != (existing.get("body") or ""):
            _gh(cfg, ["issue", "edit", str(existing["number"]),
                      "--repo", cfg.repo, "--body-file", "-"], stdin=bumped)
        return "commented-issue"

    # 2. an open PR already addressing it: one run per bug, not per job.
    pr = find_open_pr(cfg, report.sig)
    if pr is not None:
        _gh(cfg, ["pr", "comment", str(pr), "--repo", cfg.repo,
                  "--body-file", "-"],
            stdin=f"Still failing on the box at {report.occurred_at} "
                  f"(sig: {report.sig}).")
        return "commented-pr"

    body = issue_body(report)

    # 3. fixed before and back again: the previous attempt is the best
    #    context the next one can have.
    closed = find_recent_closed(cfg, report.sig)
    if closed is not None:
        body += (f"\nPreviously fixed in #{closed}; that fix did not hold.\n")

    throttled = not may_dispatch(cfg, now)
    label = THROTTLED_LABEL if throttled else AUTO_LABEL
    _create_issue(cfg, issue_title(report), body, label)
    if throttled:
        # Evidence is never lost; budget cannot run away.
        log.warning("autofix throttled: filed %s with no run", report.sig)
        return "filed-throttled"
    record_dispatch(cfg, now)
    return "filed"


def file_agent_report(cfg: AutofixConfig, title: str, body: str) -> int | None:
    """The `gpuq bug` path. Prose and unreliable blame, so it dispatches
    nothing until the owner adds `fix-me`."""
    text = ("Filed by an agent with `gpuq bug`. This is prose, not a "
            "traceback: the blame in it is a guess and there is no "
            "signature to deduplicate on. Nothing runs until the owner "
            "adds `fix-me`.\n\n---\n\n" + body)
    return _create_issue(cfg, f"[gpuq] {title}", text, REPORTED_LABEL)
