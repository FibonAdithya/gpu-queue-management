"""Job specification: the unit the queue moves between directories."""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

LANES = ("cpu", "gpu")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
# A commit is caller-supplied, reaches git's argv, and -- when git fails on
# it -- ends up verbatim inside a ``` fenced block in a bug report that a
# headless agent reads as its prompt. A newline and a closing fence escape
# that block. Wide enough for anything git takes as a rev (sha, tag,
# `origin/main`, `HEAD~1`, `v1.2.3^{commit}`), narrow enough to have no
# newlines, quotes, backticks or shell metacharacters in it.
_SAFE_REV = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/~^{}-]*$")


class SpecError(ValueError):
    """A job spec is malformed or unsafe."""


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class JobSpec:
    id: str
    lane: str
    project: str
    commit: str
    branch: str
    cmd: list[str]
    artifacts: list[str] = field(default_factory=list)
    timeout_s: int = 3600
    # None means "the whole card": admit alone, exclude everything else.
    # That is the default because a job that has not said what it needs
    # cannot be admitted alongside anything safely -- and it is what keeps
    # every spec written before this field behaving exactly as it did.
    vram_mb: int | None = None
    attempts: int = 0
    dedupe_key: str | None = None
    submitted_at: str = field(default_factory=utcnow_iso)
    pid: int | None = None
    runner_pid: int | None = None
    exit_code: int | None = None
    error: str | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "JobSpec":
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(d) - known
        if unknown:
            raise SpecError(f"unknown fields: {sorted(unknown)}")
        return cls(**d)

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> None:
        if not _SAFE_ID.match(self.id or ""):
            raise SpecError(f"id must match {_SAFE_ID.pattern}, got {self.id!r}")
        if self.lane not in LANES:
            raise SpecError(f"lane must be one of {LANES}, got {self.lane!r}")
        if not self.project:
            raise SpecError("project is required")
        if not self.commit:
            raise SpecError("commit is required; a branch alone is not reproducible")
        if not _SAFE_REV.match(self.commit):
            raise SpecError(
                f"commit must match {_SAFE_REV.pattern}, got {self.commit!r}")
        if not self.cmd or not all(isinstance(a, str) for a in self.cmd):
            raise SpecError("cmd must be a non-empty list of strings")
        if not isinstance(self.timeout_s, int) or self.timeout_s <= 0:
            raise SpecError(f"timeout_s must be a positive int, got {self.timeout_s!r}")
        if self.vram_mb is not None and (not isinstance(self.vram_mb, int)
                                         or isinstance(self.vram_mb, bool)
                                         or self.vram_mb <= 0):
            raise SpecError(
                f"vram_mb must be a positive int or None (meaning the whole "
                f"card), got {self.vram_mb!r}")
        if self.attempts < 0:
            raise SpecError("attempts must be >= 0")
        for a in self.artifacts:
            if a.startswith("/") or ".." in a.split("/"):
                raise SpecError(f"artifact path must be relative and contained: {a!r}")
