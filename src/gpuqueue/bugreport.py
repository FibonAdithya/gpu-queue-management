"""Whose fault was a failure, and what is its fingerprint?

One question is answered here: did gpuq's own code raise this? It is a
filter in code, not a model's judgment about blame -- the agent that just
failed is the least reliable narrator of whether the queue is broken or its
own request was.

Nothing in this module does I/O. Filing lives in bugfiler.py.
"""
from __future__ import annotations

import errno as _errno
import hashlib
import json
import re
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .executor import StartFailed

# The phase a failure came out of. Part of the signature, so a git failure
# during checkout and the same git failure during artifact collection are
# two bugs rather than one.
PHASES = ("preflight", "checkout", "execute", "artifacts", "reap", "admit")

# StartFailed is genuinely ambiguous: gpuq raises it, but usually because
# the caller's `-- command` names something that is not there. Only these
# errnos mean "the thing you asked to run does not exist or will not run".
# Anything else -- a bad working directory, a broken pipe -- is ours.
_CALLER_EXEC_ERRNOS = frozenset({
    _errno.ENOENT, _errno.EACCES, _errno.EPERM, _errno.ENOEXEC, _errno.EISDIR,
})


class CallerError(RuntimeError):
    """A gpuq exception that carries the caller's mistake, not gpuq's.

    Raised where gpuq's own code detects the request was wrong -- a declared
    artifact the job never produced. It surfaces as a gpuq traceback and
    would otherwise file a bug against gpuq for the caller's error, which is
    the one case the classifier cannot infer.
    """


def is_gpuq_fault(exc: BaseException) -> bool:
    """True if this exception should file a bug against gpuq.

    Unrecognised exceptions default to gpuq's fault. A novel exception class
    out of gpuq's own stack is precisely what nobody is watching for.
    """
    if isinstance(exc, CallerError):
        return False
    if isinstance(exc, StartFailed):
        return getattr(exc, "errno", None) not in _CALLER_EXEC_ERRNOS
    return True


def signature(exc: BaseException, phase: str) -> str:
    """A stable fingerprint: phase, exception type, gpuqueue frame names.

    Function names rather than line numbers, so an unrelated edit above the
    raise does not make the same bug look new and file a second issue.
    Frames outside the package are dropped: a caller's stack is not part of
    gpuq's identity, and including it would give one bug a signature per
    project on the box.
    """
    if phase not in PHASES:
        raise ValueError(f"unknown phase {phase!r}; expected one of {PHASES}")
    frames = [f.name for f in traceback.extract_tb(exc.__traceback__)
              if _is_gpuqueue_frame(f.filename)]
    payload = "|".join([phase, type(exc).__name__, *frames])
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def _is_gpuqueue_frame(filename: str) -> bool:
    return Path(filename).parent.name == "gpuqueue"


_OCCURRENCES = re.compile(r"^occurrences: (\d+)$", re.MULTILINE)
_LAST_SEEN = re.compile(r"^last seen: .*$", re.MULTILINE)

_PREAMBLE = (
    "gpuq's own code raised this. Filed automatically by the runner on the "
    "box; **no agent wrote any part of this issue** and no one has judged "
    "the blame — the classifier only knows the exception came out of "
    "`gpuqueue`'s own stack.\n"
)


@dataclass
class BugReport:
    sig: str
    phase: str
    exc_type: str
    message: str
    traceback_text: str
    queue_counts: dict[str, int] = field(default_factory=dict)
    job: dict | None = None
    gpuq_commit: str = "unknown"
    occurred_at: str = ""


def build_report(exc: BaseException, phase: str, *, spec=None,
                 queue_counts: dict[str, int] | None = None,
                 gpuq_commit: str = "unknown",
                 occurred_at: str | None = None) -> BugReport:
    return BugReport(
        sig=signature(exc, phase),
        phase=phase,
        exc_type=type(exc).__name__,
        message=str(exc),
        traceback_text="".join(traceback.format_exception(
            type(exc), exc, exc.__traceback__)),
        queue_counts=dict(queue_counts or {}),
        job=spec.to_dict() if spec is not None else None,
        gpuq_commit=gpuq_commit,
        occurred_at=occurred_at or datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
    )


def issue_title(report: BugReport) -> str:
    first_line = report.message.strip().splitlines()[0] if report.message else ""
    return f"[gpuq] {report.phase}: {report.exc_type}: {first_line}"[:200] \
           + f" [{report.sig}]"


def issue_body(report: BugReport) -> str:
    """The issue body *is* the prompt, so it carries facts and nothing else."""
    counts = ", ".join(f"{state} {n}"
                       for state, n in sorted(report.queue_counts.items())) \
        or "not recorded"
    job = json.dumps(report.job, indent=2) if report.job else \
        "no job — this failure was not attributable to one queued job"
    return "\n".join([
        _PREAMBLE,
        f"sig: {report.sig}",
        f"phase: {report.phase}",
        f"gpuq commit: {report.gpuq_commit}",
        "occurrences: 1",
        f"first seen: {report.occurred_at}",
        f"last seen: {report.occurred_at}",
        "",
        "## Traceback",
        "",
        "```",
        report.traceback_text.rstrip(),
        "```",
        "",
        "## JobSpec",
        "",
        "```json",
        job,
        "```",
        "",
        "## Queue state at the time",
        "",
        counts,
        "",
    ])


def bump_body(body: str, occurred_at: str) -> str:
    """Record a recurrence in an existing issue body.

    Returns the body untouched if it does not carry the lines we wrote --
    an owner may have rewritten it by hand, and mangling that is worse than
    losing a count.
    """
    match = _OCCURRENCES.search(body)
    if not match:
        return body
    body = _OCCURRENCES.sub(f"occurrences: {int(match.group(1)) + 1}", body,
                            count=1)
    return _LAST_SEEN.sub(f"last seen: {occurred_at}", body, count=1)
