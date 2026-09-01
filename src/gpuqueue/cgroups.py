"""Cgroup membership, for a claim that covers a tree it did not spawn.

Imports nothing from gpuqueue. That is the rule `procs.py` follows and
the reason this is not a cycle: `ledger` needs this, and `claim` imports
`ledger`.

Only the reverse direction is implemented -- pid to cgroup path, read
from `/proc/<pid>/cgroup`, which is world-readable. The forward direction
(cgroup to pid set) is the obvious implementation and is deliberately not
used: it means reading `/sys/fs/cgroup`, which needs that mount visible,
and nothing here does. The reaper on the deployment box is a host process
and would manage either; a less privileged one would not.

Why cgroups rather than pids at all: a `docker exec`'d process is a child
of the containerd-shim, not of container init, so `procs.descendants` of
a container's init pid misses it. Measured 2026-09-01; see the spec.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

# Refused scopes that deserve a better sentence than the depth rule
# gives. Every path here is one component deep, so `refuse_reason`'s
# depth rule already refuses each of them and this table adds no
# refusals -- it exists so `--scope-pid 1` says "the whole box" rather
# than "fewer than 2 components". Deleting an entry costs a good message
# and never an exemption; `test_refuse_reason_rejects_top_level_scopes_
# without_the_message_table` is what holds that apart.
NAMED_SCOPES = {
    "/": "the whole box",
    "/init.scope": "pid 1's own scope",
    "/system.slice": "every system service on this box",
    "/user.slice": "every logged-in user's session",
}

# systemd's session containers. Never a workload, and the shape every
# host shell pid resolves to -- which is why this cannot be left to the
# depth rule: `/user.slice/user-0.slice/session-1848.scope` is three
# components deep and passes it.
_SESSION = re.compile(r"^(session-[^/]+\.scope|user-[^/]+\.slice)$")


def cgroup_of(pid: int, proc_root: str = "/proc") -> str | None:
    """The unified-hierarchy cgroup path for `pid`, or None.

    None means three different things -- the pid is gone, /proc is not
    readable, or the box is cgroup v1 and has no `0::` line -- and the
    callers that care distinguish them themselves (`cli_claim` reports a
    dead pid differently from a v1 box). What this must never do is guess
    a path: a v1 line's path is not comparable to a v2 one, so returning
    the first line regardless of prefix would produce a scope that
    matches nothing and reads as a working claim.
    """
    try:
        text = Path(proc_root, str(pid), "cgroup").read_text()
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith("0::"):
            return line[3:].strip() or None
    return None


def refuse_reason(scope: str) -> str | None:
    """Why this scope may not be claimed, or None if it may.

    A mistyped `--scope-pid` is not a failed claim, it is a claim that
    silently disables orphan protection for the whole card. Pid 1
    resolves to `/`; any host shell pid resolves to a login session.
    """
    if not scope or not scope.startswith("/"):
        return f"{scope!r} is not an absolute cgroup path"
    parts = [c for c in scope.split("/") if c]
    if len(parts) < 2:
        what = NAMED_SCOPES.get(scope, "a top-level slice")
        return (f"cgroup {scope!r} is {what}; name a pid inside the "
                f"workload you mean")
    if _SESSION.match(parts[-1]):
        return (f"cgroup {scope!r} is a login session, not a workload; "
                f"name a pid inside the container or service you mean")
    return None


def in_scope(pid: int, scope: str, proc_root: str = "/proc") -> bool:
    """True when `pid` is in `scope` or in a cgroup nested under it.

    Prefix match with the separator restored, so `/a/bc` is not inside
    `/a/b`. Nesting matters because a container may create sub-cgroups of
    its own, and those processes are still the container's.

    A scope this module would have refused matches nothing, rather than
    matching everything. `docs/design.md` makes hand-repair of a record
    supported, so a `"scope_cgroup": "/"` can reach this function without
    ever having passed `refuse_reason` at claim time.
    """
    if not scope or refuse_reason(scope) is not None:
        return False
    cg = cgroup_of(pid, proc_root)
    if cg is None:
        return False
    return cg == scope or cg.startswith(scope.rstrip("/") + "/")


def scope_process_count(scope: str, proc_root: str = "/proc") -> int | None:
    """How many live processes are in `scope`, best effort.

    Informational, and taken once at claim time: it is the operator's
    check that they named a container and not the box -- "1 live process"
    against "247" is the difference. None when the walk fails, because a
    count we could not take is not a count of zero.
    """
    try:
        entries = os.listdir(proc_root)
    except OSError:
        return None
    return sum(1 for name in entries
               if name.isdigit() and in_scope(int(name), scope, proc_root))
