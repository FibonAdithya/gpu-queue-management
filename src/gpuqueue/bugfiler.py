"""File a bug against gpuq on GitHub, deduplicated and throttled.

Everything with a side effect lives here; the taxonomy lives in
bugreport.py. The one thing this module must never do is fail loudly: a box
with no `gh`, no token or no network runs jobs exactly as it does without
autofix, so callers go through Runner._report_bug, which swallows GhError.

The token this reaches for is scoped to `issues: write` on this repository
and explicitly not `contents`. Nothing here pushes, but the issue body it
files is also read as a prompt downstream, so its worst case is issue spam
plus an attacker-influenced prompt, not issue spam alone -- see
docs/specs/2026-08-05-autofix-design.md §1.
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

    An unset `cfg.token_env` means *no credentials*, not "whatever else
    this box is authenticated as". gh reads GH_TOKEN and GITHUB_TOKEN from
    the environment on its own, and falls back to `gh auth login`'s stored
    credentials after that -- so on a box where anyone had ever run
    `gh auth login`, or exported a full-scope GH_TOKEN for some other tool,
    autofix would quietly file as that identity with that identity's
    permissions. The whole security argument for this module is that its
    token is scoped to `issues: write` and cannot push; inheriting a
    different one makes the token the operator verified in
    docs/deploying.md not the token in use. Scrub both, so an
    unconfigured box fails closed into the no-token path (GhError, logged,
    queue unaffected) rather than succeeding as the wrong principal.
    """
    env = dict(os.environ)
    token = os.environ.get(cfg.token_env)
    if token:
        env["GH_TOKEN"] = token
    else:
        env.pop("GH_TOKEN", None)
        env.pop("GITHUB_TOKEN", None)
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


# The space after `sig:` in every query below is load-bearing, not
# formatting. `sig:` is not a GitHub search qualifier; with the space it is
# dropped as noise and the hex matches as a term (verified against the live
# API: `sig: <term> in:body` returns the same rows as `<term> in:body`).
# Written `sig:<hex>`, GitHub parses it as an unknown qualifier and matches
# nothing at all, silently -- every lookup returns empty, dedup never fires
# and each occurrence files a new issue.
#
# What this cannot fix: GitHub's search index is eventually consistent, so
# two occurrences inside the indexing lag both miss and file twice. The
# open-issue lookup avoids the index entirely by listing on the label
# instead; the PR and closed-issue lookups still search, where a duplicate
# is the worst case rather than a missed fix-in-flight.


def find_open_issue(cfg: AutofixConfig, sig: str) -> dict | None:
    """Is this bug already filed and open?

    Listed on the label rather than searched: this is the lookup that
    decides whether an occurrence is a recurrence, and a stale search index
    here means a duplicate issue for a bug that already has one. Every
    issue this module files carries AUTO_LABEL, so the label is a complete
    index of its own work, and `_exact` does the precise matching locally.
    """
    out = _gh(cfg, ["issue", "list", "--repo", cfg.repo, "--state", "open",
                    "--label", AUTO_LABEL, "--json", "number,body",
                    "--limit", "100"])
    try:
        rows = json.loads(out or "[]")
    except json.JSONDecodeError:
        rows = []
    rows = _exact(rows, sig)
    return rows[0] if rows else None


def find_open_pr(cfg: AutofixConfig, sig: str) -> int | None:
    """A bug that fails every job must not spawn one fix run per job. This
    is the lookup that prevents that.

    Matches the other two lookups: the workflow prompt tells the fixer to
    copy the issue's `sig: <hex>` line verbatim into the PR body, so
    searching for exactly that line (not just the bare hex, which could
    appear anywhere in a PR unrelated to this bug) is what makes this
    lookup precise rather than a coincidence of gh's full-text search.
    """
    rows = _exact(_search(cfg, "pr", "open", f"sig: {sig} in:body"), sig)
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


def refund_dispatch(cfg: AutofixConfig, now: datetime) -> None:
    """Undo one recorded dispatch, for a run that turned out impossible.

    The budget is debited before the issue exists, deliberately (see
    `file_bug`). That ordering makes an unwritable counter fail towards
    no-dispatch, but it leaves the opposite hole open: a debit for a run
    nothing can ever trigger, because `issue create` failed and there is no
    issue. Across a GitHub outage that spends the day's budget on nothing.

    Only ever narrows what was just written, and only on the path where we
    know nothing was authorised -- so unlike the debit, failing here is
    harmless and is swallowed rather than raised over the GhError the
    caller actually needs to see.
    """
    try:
        stamps = _load_dispatches(cfg)
        if now in stamps:
            stamps.remove(now)
            _save_dispatches(cfg, stamps)
    except Exception as e:
        log.warning("autofix: could not refund a dispatch: %s", e)


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
    documenting a manual step a rebuilt box will skip.

    Deliberately not `--force`: that flag *edits* a label that already
    exists, so an owner who recoloured `throttled` or rewrote its
    description had it reset on every runner restart, which reads as the
    queue fighting them over their own triage furniture. Idempotence comes
    from treating "already exists" as the success it is instead.
    """
    for name, colour, desc in _LABEL_SPECS:
        if name in _labels_ensured:
            continue
        try:
            _gh(cfg, ["label", "create", name, "--repo", cfg.repo,
                      "--color", colour, "--description", desc])
        except GhError as e:
            if "already exists" in str(e).lower():
                # The steady state of every box after its first bug. Not a
                # failure, and must not be retried four times per bug for
                # the life of the process.
                _labels_ensured.add(name)
            else:
                log.warning("could not ensure label %s: %s", name, e)
        else:
            _labels_ensured.add(name)


def _create_issue(cfg: AutofixConfig, title: str, body: str,
                  labels: str | list[str]) -> int | None:
    ensure_labels(cfg)
    if isinstance(labels, str):
        labels = [labels]
    label_args = []
    for label in labels:
        label_args += ["--label", label]
    out = _gh(cfg, ["issue", "create", "--repo", cfg.repo, "--title", title,
                    *label_args, "--body-file", "-"], stdin=body)
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
    if not throttled:
        # Record the dispatch *before* the issue exists, not after. This
        # is a two-step operation that cannot be made atomic, so the
        # survivable failure must go second: if `record_dispatch` raises
        # (state_file's directory unwritable, disk full) before any issue
        # exists, nothing has been authorised yet and we can still fall
        # back to filing as throttled below. The old order created the
        # issue first -- with the `gpuq-auto` label, already authorising a
        # run -- and only then tried to debit the budget; if that write
        # failed, the run was authorised and the budget never moved,
        # i.e. an unwritable state file silently disabled the cap forever
        # (verified: a mode-0500 state dir let five faults file five
        # `gpuq-auto` issues in a row). A broken counter must degrade to
        # no-dispatch, never to infinite-dispatch -- so on failure here we
        # fall through and file as throttled instead of dispatching.
        try:
            record_dispatch(cfg, now)
        except Exception as e:
            log.warning("autofix: could not record a dispatch, filing "
                        "throttled instead: %s", e)
            throttled = True
    # Both labels when throttled: AUTO_LABEL alone would make a
    # `label:gpuq-auto` triage query miss every throttled bug, since the
    # bug is structural evidence just the same as a dispatched one -- it
    # was only the budget, not the classification, that held it back.
    labels = [AUTO_LABEL, THROTTLED_LABEL] if throttled else [AUTO_LABEL]
    try:
        number = _create_issue(cfg, issue_title(report), body, labels)
    except Exception:
        # No issue means no run: give the budget back before the error goes
        # up, or an outage during `issue create` spends the day's cap on
        # dispatches that cannot happen. Safe in a way the debit is not --
        # this only narrows, and only where nothing was authorised.
        if not throttled:
            refund_dispatch(cfg, now)
        raise
    if number is None:
        # gh returns the new issue's URL on stdout; not finding a number in
        # it means gh changed its output, not that filing failed. Worth a
        # line in the log, since every dedup lookup from here on depends on
        # the issue we can no longer name.
        log.warning("autofix: filed %s but could not read an issue number "
                    "out of gh's output", report.sig)
    if throttled:
        # Evidence is never lost; budget cannot run away.
        log.warning("autofix throttled: filed %s with no run", report.sig)
        return "filed-throttled"
    log.info("autofix dispatched: filed %s and recorded the dispatch",
             report.sig)
    return "filed"


def file_agent_report(cfg: AutofixConfig, title: str, body: str) -> int | None:
    """The `gpuq bug` path. Prose and unreliable blame, so it dispatches
    nothing until the owner adds `fix-me`."""
    text = ("Filed by an agent with `gpuq bug`. This is prose, not a "
            "traceback: the blame in it is a guess and there is no "
            "signature to deduplicate on. Nothing runs until the owner "
            "adds `fix-me`.\n\n---\n\n" + body)
    return _create_issue(cfg, f"[gpuq] {title}", text, REPORTED_LABEL)
