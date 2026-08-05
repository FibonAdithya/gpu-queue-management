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
import traceback
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
