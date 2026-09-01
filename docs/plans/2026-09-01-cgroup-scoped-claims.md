# Cgroup-scoped claims — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a `gpu-claim` cover CUDA processes running in a container it did not spawn, by naming a cgroup instead of a pid tree — and make the orphan sweep's kill legible to its victim.

**Architecture:** A new dependency-free `gpuqueue/cgroups.py` answers "is this pid in that cgroup" by reading `/proc/<pid>/cgroup`. `ledger.Record` gains `scope_pid` + `scope_cgroup`, and `ledger.attribute` — the one function preflight, the reaper and the VRAM watchdog all share — consults them. Because ownership changes in `attribute`, all three consumers follow without knowing scopes exist. Separately the orphan sweep gains a batched SIGTERM→grace→SIGKILL ladder and writes a `kills.jsonl` that `gpuq kills` prints.

**Tech Stack:** Python 3, stdlib only (`pathlib`, `re`, `json`, `signal`). pytest. No new dependencies.

**Spec:** `docs/specs/2026-09-01-cgroup-scoped-claims-design.md`

## Global Constraints

- **No new dependencies.** stdlib only, as the rest of `src/gpuqueue/` is.
- **`gpuqueue/cgroups.py` imports nothing from `gpuqueue`.** This is the rule `procs.py` follows and the reason it is not an import cycle: `ledger` needs this module, and `claim` imports `ledger`.
- **Never read `/sys/fs/cgroup`.** Only the reverse direction (pid → cgroup path via `/proc/<pid>/cgroup`) is implemented, so this works without the cgroup mount visible.
- **Every new `Record` field is read with `.get(...)`.** `_load` returns `None` on any exception; a `d["scope_pid"]` would make every pre-upgrade record on disk unreadable, blinding the reaper to live claims. That is issue #19's failure with a new cause.
- **`preflight.own_pids` is not modified.** Its pid-tree exemption stays exactly as issue #19 left it.
- **Run the full suite before every commit:** `.venv/bin/python -m pytest -q`. Baseline on `main` at 2026-09-01 is **546 passed**. There is no bare `python` on PATH, so the venv interpreter is not optional.
- **Mutation-check each new test** before committing the task: break the code under test, confirm the test fails, restore. The plan names the mutation for each test.

---

### Task 1: The scope module

**Files:**
- Create: `src/gpuqueue/cgroups.py`
- Test: `tests/test_cgroups.py`
- Modify: `docs/design.md:24`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `cgroup_of(pid: int, proc_root: str = "/proc") -> str | None`
  - `in_scope(pid: int, scope: str, proc_root: str = "/proc") -> bool`
  - `refuse_reason(scope: str) -> str | None`
  - `scope_process_count(scope: str, proc_root: str = "/proc") -> int | None`
  - `NAMED_SCOPES: dict[str, str]`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cgroups.py`:

```python
"""The cgroup membership test behind `gpu-claim --scope-pid`.

`proc_root` is a parameter on every function here so these run against a
real fixture directory rather than a monkeypatched `cgroup_of`. A suite
that stubbed the parser at every call site would still pass with the
parser deleted, which is the one thing these tests exist to prevent.
"""
import pytest

from gpuqueue import cgroups


def _proc(tmp_path, mapping):
    """A stand-in /proc: {pid: contents of its cgroup file}."""
    root = tmp_path / "proc"
    for pid, text in mapping.items():
        d = root / str(pid)
        d.mkdir(parents=True)
        (d / "cgroup").write_text(text)
    return str(root)


DOCKER = "/system.slice/docker-43faa0ee4d16.scope"


def test_cgroup_of_reads_the_unified_line(tmp_path):
    root = _proc(tmp_path, {42: f"0::{DOCKER}\n"})
    assert cgroups.cgroup_of(42, root) == DOCKER


def test_cgroup_of_is_none_on_a_cgroup_v1_box(tmp_path):
    # v1 has one line per controller and no `0::`. Returning line 1
    # regardless of prefix would hand back "/docker/abc" here, which is a
    # v1 path that means nothing to `in_scope`'s v2 comparison.
    root = _proc(tmp_path, {42: "12:pids:/docker/abc\n"
                                "11:memory:/docker/abc\n"
                                "0:name=systemd:/user.slice\n"})
    assert cgroups.cgroup_of(42, root) is None


def test_cgroup_of_is_none_for_a_pid_that_is_gone(tmp_path):
    root = _proc(tmp_path, {42: f"0::{DOCKER}\n"})
    assert cgroups.cgroup_of(99999, root) is None


def test_in_scope_covers_a_nested_cgroup(tmp_path):
    # A container that makes its own sub-cgroups is still inside it.
    root = _proc(tmp_path, {42: f"0::{DOCKER}/worker\n"})
    assert cgroups.in_scope(42, DOCKER, root) is True


def test_in_scope_rejects_a_sibling_sharing_a_prefix(tmp_path):
    # `/a/bc` is not inside `/a/b`. Bare `startswith` says it is.
    root = _proc(tmp_path, {42: "0::/system.slice/bc\n"})
    assert cgroups.in_scope(42, "/system.slice/b", root) is False


def test_in_scope_is_false_for_a_scope_that_would_be_refused(tmp_path):
    # `docs/design.md` makes hand-repair of a record supported, so a
    # scope of "/" can reach this function even though `refuse_reason`
    # would never have let it be claimed. It must not then match the box.
    root = _proc(tmp_path, {42: "0::/system.slice/anything.scope\n"})
    assert cgroups.in_scope(42, "/", root) is False


def test_refuse_reason_admits_a_container_scope():
    assert cgroups.refuse_reason(DOCKER) is None


def test_refuse_reason_rejects_a_login_session():
    # Three components deep, so a depth-only check passes it -- and this
    # is the shape EVERY host shell pid resolves to, so it is the
    # likeliest accident rather than the least.
    reason = cgroups.refuse_reason(
        "/user.slice/user-0.slice/session-1848.scope")
    assert reason is not None
    assert "session" in reason


def test_refuse_reason_rejects_a_user_slice():
    assert cgroups.refuse_reason("/user.slice/user-0.slice") is not None


@pytest.mark.parametrize("scope", ["/", "/init.scope", "/system.slice",
                                   "/user.slice"])
def test_refuse_reason_rejects_top_level_scopes_without_the_message_table(
        monkeypatch, scope):
    # The refusal must come from the depth rule, not from NAMED_SCOPES.
    # That table exists only so `--scope-pid 1` says "the whole box"
    # instead of "fewer than 2 components"; emptying it must cost a good
    # message and never an exemption.
    monkeypatch.setattr(cgroups, "NAMED_SCOPES", {})
    assert cgroups.refuse_reason(scope) is not None


def test_refuse_reason_rejects_a_relative_path():
    assert cgroups.refuse_reason("system.slice/x.scope") is not None


def test_scope_process_count_counts_only_the_scope(tmp_path):
    root = _proc(tmp_path, {
        11: f"0::{DOCKER}\n",
        12: f"0::{DOCKER}/worker\n",
        13: "0::/user.slice/user-0.slice/session-1.scope\n",
    })
    assert cgroups.scope_process_count(DOCKER, root) == 2


def test_scope_process_count_is_none_when_proc_is_unreadable(tmp_path):
    # None is not zero. A count we could not take must not read as "you
    # named an empty cgroup".
    assert cgroups.scope_process_count(DOCKER, str(tmp_path / "absent")) is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_cgroups.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'gpuqueue.cgroups'`

- [ ] **Step 3: Write the implementation**

Create `src/gpuqueue/cgroups.py`:

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_cgroups.py -q`
Expected: PASS (16 tests — the parametrized case is four)

- [ ] **Step 5: Mutation-check the two load-bearing tests**

1. In `cgroup_of`, replace the loop body with `return text.splitlines()[0].strip()`. Run `.venv/bin/python -m pytest tests/test_cgroups.py -q`. Expected: `test_cgroup_of_is_none_on_a_cgroup_v1_box` FAILS. Restore.
2. In `in_scope`, change the last line to `return cg.startswith(scope)`. Run again. Expected: `test_in_scope_rejects_a_sibling_sharing_a_prefix` FAILS. Restore.
3. In `refuse_reason`, delete the `_SESSION` branch. Run again. Expected: `test_refuse_reason_rejects_a_login_session` FAILS. Restore.

- [ ] **Step 6: Correct the stale deployment claim in design.md**

`docs/design.md:24` currently reads:

```
- The target box is an **unprivileged container** (a hosted PyTorch image). No
  Docker-in-Docker, no kernel modules, no sysctls. Long-running processes are
  managed by **supervisor**.
```

Replace with:

```
- The target box runs the queue as an **unprivileged host process** under
  **supervisor**. No kernel modules, no sysctls, and nothing here may require
  root. It was originally a hosted PyTorch container; on `tig-gpu` as of
  2026-09-01 the runner is a host process in the root cgroup namespace
  (`/system.slice/supervisor.service`), and GPU work of its *users* is what
  now runs in containers. `cgroups.py` depends on that direction only: it
  reads `/proc/<pid>/cgroup`, never `/sys/fs/cgroup`, so it degrades to
  "covers nothing" rather than mis-exempting if the runner is ever
  namespaced again.
```

- [ ] **Step 7: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS, with 14 more tests than before.

- [ ] **Step 8: Commit**

```bash
git add src/gpuqueue/cgroups.py tests/test_cgroups.py docs/design.md
git commit -m "feat: cgroup membership for claims that cover a foreign tree

A docker exec'd process is a child of the containerd-shim, not of
container init, so descendants() of a container's init pid misses it.
The cgroup is the boundary that matches what an operator means by 'this
container'; the pid tree is not.

Reads /proc/<pid>/cgroup only, never /sys/fs/cgroup, so it needs no
mount visibility. Refuses the root cgroup, top-level slices and login
sessions -- the shapes a mistyped --scope-pid resolves to."
```

---

### Task 2: The record carries a scope

**Files:**
- Modify: `src/gpuqueue/ledger.py` (`Record` at :80, `to_dict` at :94, `_load` at :113, `acquire` at :334)
- Test: `tests/test_ledger.py`

**Interfaces:**
- Consumes: `cgroups.cgroup_of` from Task 1.
- Produces:
  - `Record.scope_pid: int | None`, `Record.scope_cgroup: str | None`
  - `ledger.scope_is_live(rec: Record) -> bool`
  - `acquire(..., scope_pid: int | None = None, scope_cgroup: str | None = None)`

**Import style matters:** add `from . import cgroups` to `ledger.py`, not `from .cgroups import cgroup_of`. Tests monkeypatch `cgroups.cgroup_of`, and a name bound at import would not see the patch.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_ledger.py`:

```python
def test_a_record_written_without_scope_keys_still_loads(tmp_path):
    # Every record already on disk at upgrade time lacks these keys. A
    # `d["scope_pid"]` would send `_load` down its `except` and return
    # None for all of them -- and a record the reaper cannot read is a
    # claim it will not spare. That is issue #19 with a new cause.
    p = tmp_path / "old.json"
    p.write_text(json.dumps({"pid": 4321, "usage_pid": 4321,
                             "vram_mb": 512, "owner": "someone",
                             "cmd": ["train"], "started_at": "",
                             "key": "k"}))
    rec = lg._load(p)
    assert rec is not None
    assert rec.pid == 4321
    assert rec.scope_pid is None and rec.scope_cgroup is None


def test_acquire_records_the_scope(tmp_path):
    rec = lg.acquire("k", vram_mb=512, owner="me", cmd=["x"],
                         directory=tmp_path, usable_mb=8000,
                         usage_pid=os.getpid(),
                         scope_pid=4321,
                         scope_cgroup="/system.slice/docker-abc.scope")
    on_disk = json.loads(rec.path.read_text())
    assert on_disk["scope_pid"] == 4321
    assert on_disk["scope_cgroup"] == "/system.slice/docker-abc.scope"
    assert lg._load(rec.path).scope_cgroup == \
        "/system.slice/docker-abc.scope"


def _scoped(tmp_path, scope_pid, scope_cgroup):
    return lg.Record(
        path=tmp_path / "r.json", pid=os.getpid(), usage_pid=os.getpid(),
        vram_mb=512, owner="me", cmd=[], started_at="", key="k",
        scope_pid=scope_pid, scope_cgroup=scope_cgroup)


def test_scope_is_live_when_the_anchor_still_sits_where_it_did(
        tmp_path, monkeypatch):
    monkeypatch.setattr(lg.cgroups, "cgroup_of",
                        lambda pid, proc_root="/proc": "/system.slice/d.scope")
    rec = _scoped(tmp_path, os.getpid(), "/system.slice/d.scope")
    assert lg.scope_is_live(rec) is True


def test_scope_is_dead_when_the_anchor_is_gone(tmp_path, monkeypatch):
    # A dead anchor whose pid the kernel later reuses would otherwise
    # drift the exemption onto an unrelated process's cgroup.
    monkeypatch.setattr(lg.cgroups, "cgroup_of",
                        lambda pid, proc_root="/proc": "/system.slice/d.scope")
    rec = _scoped(tmp_path, 999999, "/system.slice/d.scope")
    assert lg.scope_is_live(rec) is False


def test_scope_is_dead_when_the_anchor_moved_cgroup(tmp_path, monkeypatch):
    # The container restarted: same pid alive, new docker scope id. A
    # check of `pid_alive` alone would honour the stale path.
    monkeypatch.setattr(lg.cgroups, "cgroup_of",
                        lambda pid, proc_root="/proc": "/system.slice/NEW.scope")
    rec = _scoped(tmp_path, os.getpid(), "/system.slice/OLD.scope")
    assert lg.scope_is_live(rec) is False


def test_a_record_with_no_scope_is_not_live_scoped(tmp_path):
    assert lg.scope_is_live(_scoped(tmp_path, None, None)) is False
```

`tests/test_ledger.py` already imports `json`, `os` and binds the module as `from gpuqueue import ledger as lg` — use that alias, do not add a second name for the same module.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_ledger.py -q -k "scope"`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'scope_pid'` and `AttributeError: module 'gpuqueue.ledger' has no attribute 'scope_is_live'`

- [ ] **Step 3: Write the implementation**

In `src/gpuqueue/ledger.py`, add to the imports at :25-27:

```python
from . import cgroups
```

Extend the `Record` dataclass (:80-98). The two new fields get defaults so
every existing construction site keeps working:

```python
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
    # A tree this record is charged for that it did not spawn. `scope_pid`
    # is the anchor the claim named; `scope_cgroup` is where that anchor
    # sat when the claim was taken. Both, because either alone is wrong:
    # the pid alone drifts onto whatever the kernel recycles it to, and
    # the path alone outlives the container that owned it.
    scope_pid: int | None = None
    scope_cgroup: str | None = None

    @property
    def name(self) -> str:
        return self.path.name

    def to_dict(self) -> dict:
        return {"pid": self.pid, "usage_pid": self.usage_pid,
                "vram_mb": self.vram_mb, "owner": self.owner,
                "cmd": self.cmd, "started_at": self.started_at,
                "key": self.key, "scope_pid": self.scope_pid,
                "scope_cgroup": self.scope_cgroup}
```

Extend `_load` (:113-125). Both reads are `.get`:

```python
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
            started_at=d.get("started_at", ""), key=d.get("key", ""),
            # `.get`, not `d[...]`: every record written before this
            # field existed is still on disk, and `_load` returns None on
            # any exception. A KeyError here would make the reaper unable
            # to read live claims across an upgrade -- issue #19's
            # failure reached by a different door.
            scope_pid=(int(d["scope_pid"])
                       if d.get("scope_pid") is not None else None),
            scope_cgroup=d.get("scope_cgroup"))
    except Exception:
        return None  # a garbage record must not blind us to the good ones
```

Add `scope_is_live` immediately after `set_usage_pid` (:185-187):

```python
def scope_is_live(rec: Record) -> bool:
    """True when a record's scope still names what it named at claim time.

    Both halves are load-bearing, and each covers the other's blind spot.
    `pid_alive` alone lets the kernel recycle `scope_pid` onto an
    unrelated process and drift the exemption onto *its* cgroup. The path
    alone outlives the container: a restart gives docker a new scope id,
    so the recorded path would name a cgroup that no longer exists, or
    worse, one reissued to something else.

    Disagreement means the scope covers nothing, rather than covering
    something the claimant did not name. `reap()` reports which records
    went void, because a claim that has quietly stopped covering anything
    is the same silent failure issue #24 is about.
    """
    if rec.scope_pid is None or rec.scope_cgroup is None:
        return False
    if not pid_alive(rec.scope_pid):
        return False
    return cgroups.cgroup_of(rec.scope_pid) == rec.scope_cgroup
```

Extend `acquire`'s signature (:334-337) and its `Record(...)` construction (:364-367):

```python
def acquire(key: str, *, vram_mb: int | None, owner: str,
            cmd: list[str] | None, directory, usable_mb: int | None,
            usage_pid: int | None = None,
            max_holders: int | None = None,
            scope_pid: int | None = None,
            scope_cgroup: str | None = None) -> Record:
```

```python
        rec = Record(path=ldir / f"{os.getpid()}.{secrets.token_hex(3)}.json",
                     pid=os.getpid(), usage_pid=usage_pid, vram_mb=vram_mb,
                     owner=owner, cmd=list(cmd or []),
                     started_at=utcnow_iso(), key=key,
                     scope_pid=scope_pid, scope_cgroup=scope_cgroup)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_ledger.py -q`
Expected: PASS

- [ ] **Step 5: Mutation-check**

1. In `_load`, change `scope_pid=(int(d["scope_pid"]) if d.get(...))` to `scope_pid=int(d["scope_pid"])`. Run `.venv/bin/python -m pytest tests/test_ledger.py -q`. Expected: `test_a_record_written_without_scope_keys_still_loads` FAILS. Restore.
2. In `scope_is_live`, delete the `pid_alive` check. Expected: `test_scope_is_dead_when_the_anchor_is_gone` FAILS. Restore.
3. In `scope_is_live`, replace the final line with `return True`. Expected: `test_scope_is_dead_when_the_anchor_moved_cgroup` FAILS. Restore.

- [ ] **Step 6: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS. Nothing else reads these fields yet, so no existing test should change behaviour.

- [ ] **Step 7: Commit**

```bash
git add src/gpuqueue/ledger.py tests/test_ledger.py
git commit -m "feat: claim records carry a cgroup scope

scope_pid is the anchor the claim named; scope_cgroup is where that
anchor sat when the claim was taken. Both, because either alone is
wrong: the pid drifts onto whatever the kernel recycles it to, and the
path outlives the container that owned it. Disagreement voids the
scope rather than exempting a stranger.

Both fields load with .get, so records written before this change stay
readable -- a record the reaper cannot read is a claim it will not
spare."
```

---

### Task 3: `attribute` charges a scoped process to its claim

**Files:**
- Modify: `src/gpuqueue/ledger.py:377-417` (`attribute`)
- Test: `tests/test_ledger.py`

**Interfaces:**
- Consumes: `cgroups.in_scope` (Task 1), `ledger.scope_is_live` (Task 2).
- Produces: no signature change. `attribute(apps, records)` keeps returning `(owned, unledgered)` keyed by `str(record.path)`.

This is the task that fixes issue #24. Everything before it is scaffolding and everything after is surface.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_ledger.py`:

```python
SCOPE = "/system.slice/docker-43faa0ee.scope"


def _rec(tmp_path, name, *, usage_pid=None, scope_pid=None,
         scope_cgroup=None):
    return lg.Record(
        path=tmp_path / name, pid=os.getpid(), usage_pid=usage_pid,
        vram_mb=512, owner="me", cmd=[], started_at="", key="k",
        scope_pid=scope_pid, scope_cgroup=scope_cgroup)


def test_a_process_in_a_claimed_cgroup_is_charged_to_that_record(
        tmp_path, monkeypatch):
    # Issue #24 at unit scale: the CUDA process is not a descendant of
    # anything the claim could name, and the claim covers it anyway.
    monkeypatch.setattr(lg.cgroups, "cgroup_of",
                        lambda pid, proc_root="/proc": SCOPE)
    rec = _rec(tmp_path, "a.json", usage_pid=os.getpid(),
               scope_pid=os.getpid(), scope_cgroup=SCOPE)
    apps = [{"pid": 2791919, "used_mb": 900, "name": "tig-runtime"}]
    owned, unledgered = lg.attribute(apps, [rec])
    assert unledgered == []
    assert owned[str(rec.path)] == apps


def test_a_process_outside_every_scope_is_unledgered(tmp_path, monkeypatch):
    # The other half: the feature must not exempt the whole box.
    monkeypatch.setattr(
        lg.cgroups, "cgroup_of",
        lambda pid, proc_root="/proc":
            SCOPE if pid == os.getpid() else "/system.slice/other.scope")
    rec = _rec(tmp_path, "a.json", usage_pid=os.getpid(),
               scope_pid=os.getpid(), scope_cgroup=SCOPE)
    apps = [{"pid": 2791919, "used_mb": 900, "name": "stranger"}]
    owned, unledgered = lg.attribute(apps, [rec])
    assert unledgered == apps
    assert owned == {}


def test_a_dead_anchor_charges_nothing_to_its_recorded_cgroup(
        tmp_path, monkeypatch):
    # Honouring scope_cgroup without re-checking the anchor would exempt
    # whatever now lives at that path.
    monkeypatch.setattr(lg.cgroups, "cgroup_of",
                        lambda pid, proc_root="/proc": SCOPE)
    rec = _rec(tmp_path, "a.json", usage_pid=os.getpid(),
               scope_pid=999999, scope_cgroup=SCOPE)
    apps = [{"pid": 2791919, "used_mb": 900, "name": "tig-runtime"}]
    owned, unledgered = lg.attribute(apps, [rec])
    assert unledgered == apps


def test_pid_tree_ownership_is_tested_before_scope_ownership(
        tmp_path, monkeypatch):
    # Two records could both claim this process: one owns its pid tree,
    # the other's cgroup contains it. The tree is the more specific
    # answer and must win, or a co-tenant's VRAM is charged to the
    # container's declaration and neither reads correctly.
    monkeypatch.setattr(lg.cgroups, "cgroup_of",
                        lambda pid, proc_root="/proc": SCOPE)
    monkeypatch.setattr(lg, "descendants", lambda pid: {2791919})
    by_tree = _rec(tmp_path, "tree.json", usage_pid=os.getpid())
    # usage_pid=None on purpose: `descendants` is stubbed for every pid,
    # so giving this record a usage_pid too would put the app in BOTH
    # records' pid trees, and the test would then turn on dict order
    # rather than on the tree-before-scope rule it claims to check.
    by_scope = _rec(tmp_path, "scope.json", usage_pid=None,
                    scope_pid=os.getpid(), scope_cgroup=SCOPE)
    apps = [{"pid": 2791919, "used_mb": 900, "name": "x"}]
    owned, unledgered = lg.attribute(apps, [by_scope, by_tree])
    assert str(by_tree.path) in owned
    assert str(by_scope.path) not in owned
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_ledger.py -q -k "charged or unledgered or anchor or tested_before"`
Expected: FAIL — the scoped process lands in `unledgered`.

- [ ] **Step 3: Write the implementation**

Replace `attribute`'s body (`ledger.py:401-417`) — keep the existing docstring and the comment above `trees`, and add to the docstring:

```python
    A record may also name a *scope*: a cgroup it is charged for but did
    not spawn, which is how a container's CUDA process gets an owner at
    all (issue #24). Scopes are consulted only after every pid tree has
    been tried, so the tree stays the more specific answer.
```

Then:

```python
    trees = {str(r.path): {r.usage_pid} | descendants(r.usage_pid)
             for r in records if r.usage_pid is not None}
    # Resolved once per call, not once per app. `scope_is_live` reads
    # /proc for the anchor, and there are always more visible apps than
    # scoped records.
    scopes = {str(r.path): r.scope_cgroup
              for r in records if scope_is_live(r)}
    owned: dict[str, list[dict]] = {}
    unledgered: list[dict] = []
    for app in apps:
        path = _owner_of(app["pid"], trees, scopes)
        if path is None:
            unledgered.append(app)
        else:
            owned.setdefault(path, []).append(app)
    return owned, unledgered


def _owner_of(pid: int, trees: dict[str, set[int]],
              scopes: dict[str, str]) -> str | None:
    """Which record owns `pid`, or None.

    First match wins, and pid trees are tried before scopes. If two
    records' trees overlap (a holder that forked another holder) the
    process is charged to whichever sorts first and the other reads
    `used=0` -- under its declaration, so an overlap can only fail to
    convict, never convict the wrong holder.

    Trees before scopes for the same reason: a scope is the coarser
    claim, covering a whole container, and a co-tenant that took its own
    claim inside that container has named itself more precisely. Charging
    it to the container's declaration instead would leave both readings
    wrong -- the co-tenant's record showing `used=0` and the container's
    inflated by a process that declared separately.
    """
    for path, tree in trees.items():
        if pid in tree:
            return path
    for path, scope in scopes.items():
        if cgroups.in_scope(pid, scope):
            return path
    return None
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_ledger.py -q`
Expected: PASS

- [ ] **Step 5: Mutation-check**

1. Delete the `scopes` loop from `_owner_of`. Expected: `test_a_process_in_a_claimed_cgroup_is_charged_to_that_record` FAILS. Restore.
2. Move the `scopes` loop above the `trees` loop. Expected: `test_pid_tree_ownership_is_tested_before_scope_ownership` FAILS. Restore.
3. Change `scopes` to include every record with a `scope_cgroup` regardless of `scope_is_live`. Expected: `test_a_dead_anchor_charges_nothing_to_its_recorded_cgroup` FAILS. Restore.

- [ ] **Step 6: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS. `preflight`, `kill_orphan_cuda` and `check_vram` all route through `attribute`, so this is where a regression in any of them would show.

- [ ] **Step 7: Commit**

```bash
git add src/gpuqueue/ledger.py tests/test_ledger.py
git commit -m "feat: attribute a scoped process to the claim that named it

Fixes the ownership half of #24. attribute() is the one function
preflight, the orphan reaper and the VRAM watchdog share -- its
docstring says why they must -- so teaching it about scopes makes a
container's CUDA process owned rather than unledgered for all three,
without any of them knowing scopes exist.

Pid trees are tested before scopes: a co-tenant that claimed inside the
container named itself more precisely than the container did."
```

---

### Task 4: `gpu-claim --scope-pid`

**Files:**
- Modify: `src/gpuqueue/claim.py:260-330` (`gpu_claim`)
- Modify: `src/gpuqueue/preflight.py:113-141` (`preflight`)
- Modify: `src/gpuqueue/cli_claim.py` (parser at :27-45, `main` at :148)
- Modify: `tests/test_cli_claim.py:12-20` (the `fake_gpu` fixture)
- Modify: `docs/design.md` §Lock protocol
- Test: `tests/test_cli_claim.py`, `tests/test_preflight.py`

**Interfaces:**
- Consumes: `cgroups.*` (Task 1), `ledger.acquire(scope_pid=, scope_cgroup=)` (Task 2).
- Produces:
  - `claim.gpu_claim(..., scope_pid: int | None = None, scope_cgroup: str | None = None)`
  - `preflight.preflight(allow=None, directory=None, scope: str | None = None)`

**Breaking-stub warning:** `tests/test_cli_claim.py:16` stubs preflight as `lambda allow=None, directory=None: None`. Once `main` calls `preflight(scope=...)` that stub raises `TypeError`. Step 1 updates it; do not skip that or every test in the file fails for the wrong reason.

- [ ] **Step 1: Update the existing preflight stub**

In `tests/test_cli_claim.py`, change line 16 from:

```python
    monkeypatch.setattr(cli_claim, "preflight", lambda allow=None, directory=None: None)
```

to:

```python
    monkeypatch.setattr(cli_claim, "preflight",
                        lambda allow=None, directory=None, scope=None: None)
```

Also update the two inline stubs `monkeypatch.setattr(cli_claim, "preflight", lambda: None)` (in `test_gpu_claim_passes_the_declaration_through` and its neighbour) to `lambda **kw: None`.

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_cli_claim.py`:

```python
DOCKER_SCOPE = "/system.slice/docker-43faa0ee4d16.scope"


def test_scope_pid_is_passed_into_the_claim(tmp_path, monkeypatch):
    from contextlib import contextmanager
    seen = {}

    @contextmanager
    def fake_claim(**kw):
        seen.update(kw)
        yield None

    monkeypatch.setattr(cli_claim, "gpu_claim", fake_claim)
    monkeypatch.setattr(cli_claim.cgroups, "cgroup_of",
                        lambda pid, proc_root="/proc": DOCKER_SCOPE)
    assert cli_claim.main(["--scope-pid", "2818873", "--", "true"]) == 0
    assert seen["scope_pid"] == 2818873
    assert seen["scope_cgroup"] == DOCKER_SCOPE


def test_scope_pid_prints_the_resolved_scope(monkeypatch, capsys):
    # The operator's sanity check that they named a container and not the
    # box. Without it a wrong --scope-pid looks exactly like a right one.
    monkeypatch.setattr(cli_claim.cgroups, "cgroup_of",
                        lambda pid, proc_root="/proc": DOCKER_SCOPE)
    monkeypatch.setattr(cli_claim.cgroups, "scope_process_count",
                        lambda scope, proc_root="/proc": 3)
    cli_claim.main(["--scope-pid", "2818873", "--", "true"])
    err = capsys.readouterr().err
    assert DOCKER_SCOPE in err
    # The whole phrase: DOCKER_SCOPE already contains a "3", so a bare
    # `"3" in err` passes with the count deleted entirely.
    assert "3 live processes" in err


def test_scope_pid_naming_the_whole_box_is_refused(monkeypatch, capsys):
    monkeypatch.setattr(cli_claim.cgroups, "cgroup_of",
                        lambda pid, proc_root="/proc": "/")
    assert cli_claim.main(["--scope-pid", "1", "--", "true"]) == 2
    assert "whole box" in capsys.readouterr().err


def test_scope_pid_naming_a_login_session_is_refused(monkeypatch, capsys):
    monkeypatch.setattr(
        cli_claim.cgroups, "cgroup_of",
        lambda pid, proc_root="/proc":
            "/user.slice/user-0.slice/session-1848.scope")
    assert cli_claim.main(["--scope-pid", "2838576", "--", "true"]) == 2
    assert "login session" in capsys.readouterr().err


def test_scope_pid_that_is_not_running_is_refused(monkeypatch, capsys):
    monkeypatch.setattr(cli_claim.cgroups, "cgroup_of",
                        lambda pid, proc_root="/proc": None)
    monkeypatch.setattr(cli_claim, "pid_alive", lambda pid: False)
    assert cli_claim.main(["--scope-pid", "999999", "--", "true"]) == 2
    assert "not a running process" in capsys.readouterr().err


def test_scope_pid_on_a_cgroup_v1_box_is_refused(monkeypatch, capsys):
    # A live pid with no unified path is a v1 box, and the operator's
    # next move is nothing like "fix the pid".
    monkeypatch.setattr(cli_claim.cgroups, "cgroup_of",
                        lambda pid, proc_root="/proc": None)
    monkeypatch.setattr(cli_claim, "pid_alive", lambda pid: True)
    assert cli_claim.main(["--scope-pid", "2818873", "--", "true"]) == 2
    assert "cgroup v2" in capsys.readouterr().err

```

Append to `tests/test_preflight.py`:

```python
def test_preflight_does_not_refuse_a_process_in_the_prospective_scope(
        tmp_path, monkeypatch):
    # preflight runs BEFORE the claim exists, so there is no record to
    # attribute the container's in-flight CUDA to. Without the
    # prospective scope, claiming a busy container is refused and
    # claiming an idle one races the next request -- the feature fails
    # exactly when it is needed.
    scope = "/system.slice/docker-43faa0ee.scope"
    monkeypatch.setattr(
        preflight, "compute_apps",
        lambda: [{"pid": 2791919, "used_mb": 900, "name": "tig-runtime"}])
    monkeypatch.setattr(pf.cgroups, "in_scope",
                        lambda pid, s, proc_root="/proc": s == scope)
    with pytest.raises(pf.PreflightFailed):
        pf.preflight(directory=tmp_path)
    pf.preflight(directory=tmp_path, scope=scope)  # must not raise
```

`tests/test_preflight.py` already imports `os` and `pytest` and binds the module as `from gpuqueue import preflight as pf` — use that alias.

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_cli_claim.py tests/test_preflight.py -q`
Expected: FAIL — `unrecognized arguments: --scope-pid`, and `preflight() got an unexpected keyword argument 'scope'`.

- [ ] **Step 4: Implement — `preflight`**

In `src/gpuqueue/preflight.py`, add `from . import cgroups` to the imports, and change `preflight` (:113):

```python
def preflight(allow: set[int] | None = None, directory=None,
              scope: str | None = None) -> None:
```

and after the `stray` list comprehension (:130), before the `if stray:`:

```python
    if scope:
        # A claim that has not been taken yet. `attribute` covers a scope
        # once the record is on disk, but this runs *before* that, so the
        # container's in-flight CUDA has nothing to be charged to and
        # would read as a stranger holding the card. Filtering here is
        # what lets a busy container be claimed at all.
        stray = [a for a in stray if not cgroups.in_scope(a["pid"], scope)]
```

Add to the docstring of `preflight`: `scope` is a cgroup the caller is about to claim; processes inside it are not contention.

- [ ] **Step 5: Implement — `gpu_claim`**

In `src/gpuqueue/claim.py`, extend the signature (:260-268):

```python
              max_holders: int | None | object = _ASK_THE_CONFIG,
              scope_pid: int | None = None,
              scope_cgroup: str | None = None):
```

and the `ledger.acquire` call (:320-325):

```python
            rec = ledger.acquire(
                key, vram_mb=vram_mb, owner=owner or _default_owner(),
                cmd=cmd, directory=d, usable_mb=usable_mb,
                usage_pid=os.getpid() if own_usage else None,
                max_holders=max_holders,
                scope_pid=scope_pid, scope_cgroup=scope_cgroup)
```

Add to the docstring:

```
    `scope_pid`/`scope_cgroup` charge this claim for a cgroup it did not
    spawn -- a container's CUDA work, which is not a descendant of
    anything a claim can name from the host shell (issue #24). `usage_pid`
    is still set alongside: ownership is the union of the pid tree and
    the scope, because the wrapped command may touch the card itself and
    over-exempting is the safe way to be wrong.
```

- [ ] **Step 6: Implement — `cli_claim`**

Add to the imports in `src/gpuqueue/cli_claim.py`:

```python
from . import cgroups
from .claim import pid_alive
```

Add to `build_parser`, after `--vram-mb` (:38-43):

```python
    p.add_argument("--scope-pid", dest="scope_pid", type=int, default=None,
                   help="claim on behalf of the cgroup this pid belongs "
                        "to, for CUDA that runs in a container rather "
                        "than in this command's own process tree. Name "
                        "any pid inside the target, e.g. --scope-pid "
                        "$(docker inspect -f '{{.State.Pid}}' <container>).")
```

Add a resolver above `main`:

```python
def _resolve_scope(scope_pid: int) -> str | None:
    """The cgroup `--scope-pid` names, or None after reporting why not.

    Resolution and refusal happen here, at claim time, rather than in the
    reaper an hour later: an over-broad scope does not fail, it silently
    disables orphan protection for the card, and the operator's only
    other signal would be a SIGKILL that never comes.
    """
    if scope_pid <= 0:
        print(f"gpu-claim: --scope-pid must be a pid, got {scope_pid}",
              file=sys.stderr)
        return None
    scope = cgroups.cgroup_of(scope_pid)
    if scope is None:
        # Two causes with different next moves: a pid that is gone is a
        # typo or a race the operator retries, a box with no unified
        # hierarchy cannot support this flag at all.
        if not pid_alive(scope_pid):
            print(f"gpu-claim: --scope-pid {scope_pid} is not a running "
                  f"process", file=sys.stderr)
        else:
            print(f"gpu-claim: pid {scope_pid} has no unified cgroup path; "
                  f"--scope-pid needs cgroup v2 and this box is not "
                  f"running it", file=sys.stderr)
        return None
    reason = cgroups.refuse_reason(scope)
    if reason is not None:
        print(f"gpu-claim: --scope-pid {scope_pid}: {reason}",
              file=sys.stderr)
        return None
    n = cgroups.scope_process_count(scope)
    where = "" if n is None else \
        f" ({n} live process{'' if n == 1 else 'es'})"
    print(f"gpu-claim: scope {scope}{where}", file=sys.stderr)
    return scope
```

In `main`, after the `--vram-mb` validation (:195-203) and before `gpu_key` (:206):

```python
    scope_cgroup = None
    if args.scope_pid is not None:
        scope_cgroup = _resolve_scope(args.scope_pid)
        if scope_cgroup is None:
            return 2
```

Change the preflight call (:213-217):

```python
    if not args.no_preflight:
        try:
            preflight(scope=scope_cgroup)
        except PreflightFailed as e:
            print(f"gpu-claim: {e}", file=sys.stderr)
            return EX_UNAVAILABLE
```

Change the `gpu_claim` call (:222-225):

```python
        with gpu_claim(key=key, owner=args.owner, cmd=cmd, wait=args.wait,
                       vram_mb=args.vram_mb,
                       usable_mb=default_usable_mb(args.gpu_index),
                       scope_pid=args.scope_pid,
                       scope_cgroup=scope_cgroup):
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_cli_claim.py tests/test_preflight.py -q`
Expected: PASS

- [ ] **Step 8: Mutation-check**

1. Delete the `if scope:` block from `preflight.py`. Run `.venv/bin/python -m pytest tests/test_preflight.py -q`. Expected: `test_preflight_does_not_refuse_a_process_in_the_prospective_scope` FAILS. Restore.
2. In `_resolve_scope`, drop the `refuse_reason` check. Expected: `test_scope_pid_naming_the_whole_box_is_refused` and `test_scope_pid_naming_a_login_session_is_refused` FAIL. Restore.
3. In `_resolve_scope`, collapse the two `scope is None` branches into one message. Expected: one of `test_scope_pid_that_is_not_running_is_refused` / `test_scope_pid_on_a_cgroup_v1_box_is_refused` FAILS. Restore.

- [ ] **Step 9: Document the surface in design.md**

In `docs/design.md`, §Lock protocol, after the claim-file shape, add:

```markdown
### Claiming for a container

A claim normally covers the claimant's own process tree. That cannot
express containerised CUDA: a `docker exec`'d process is a child of the
containerd-shim, not of container init, so it is outside the pid tree of
anything a claim can name from the host shell.

`gpu-claim --scope-pid <pid>` charges the claim for the **cgroup** that
pid belongs to, and everything nested under it:

    gpu-claim --vram-mb 3000 \
      --scope-pid $(docker inspect -f '{{.State.Pid}}' my-container) \
      -- ./run_experiment.py

The record then carries `scope_pid` and `scope_cgroup`, and the scope is
honoured only while the anchor is alive *and* still in the recorded
cgroup — so a container restart voids the scope rather than drifting it
onto whatever the kernel gives that pid next.

Refused: the root cgroup, top-level slices, and login sessions. Those
are what a mistyped `--scope-pid` resolves to, and each would disable
orphan protection for the whole card while looking like a normal claim.
```

- [ ] **Step 10: Run the full suite and commit**

```bash
.venv/bin/python -m pytest -q
git add src/gpuqueue/cli_claim.py src/gpuqueue/claim.py \
        src/gpuqueue/preflight.py tests/test_cli_claim.py \
        tests/test_preflight.py docs/design.md
git commit -m "feat: gpu-claim --scope-pid claims for a container's cgroup

Closes the interface half of #24: there was no way to claim for a tree
you did not spawn, so containerised CUDA was unledgerable by
construction rather than by anyone's mistake.

preflight takes the prospective scope too. It runs before the claim
exists, so without it a container already serving a request reads as a
stranger on the card -- claiming a busy container would be refused and
claiming an idle one would race the next request."
```

---

### Task 5: The orphan sweep stops going straight to SIGKILL

**Files:**
- Modify: `src/gpuqueue/reaper.py:54-99` (`kill_orphan_cuda`), `:395`, `:410-412` (`reap`'s return)
- Test: `tests/test_reaper.py`

**Interfaces:**
- Consumes: `cgroups.cgroup_of` (Task 1).
- Produces:
  - `kill_orphan_cuda(protect, records, apps) -> list[dict]` with keys `pid`, `name`, `used_mb`, `cgroup`
  - `reaper.ORPHAN_TERM_GRACE_S = 5.0`
  - `reap()` returns **both** `killed_pids: list[int]` (derived, unchanged type) and `killed_details: list[dict]`

**Why both keys:** `killed_pids` is asserted as a list of ints in twelve places across `tests/test_reaper.py` and `tests/test_runner.py`, and read at `runner.py:258`. One producer, two views — `reap` derives `killed_pids` from the details — so they cannot drift and no existing assertion or log line changes.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_reaper.py`:

```python
def test_orphan_sweep_sigterms_before_it_sigkills(monkeypatch):
    # A SIGKILLed process writes no stderr, so its caller sees `exit -9`
    # and an empty message and reads it as its own bug. That is what cost
    # the diagnosis in #24. SIGTERM first gives a handler the chance to
    # say what happened; the watchdog's _kill_tree has had this since it
    # was written and the orphan sweep never did.
    sent = []
    monkeypatch.setattr(rp, "_signal",
                        lambda pid, sig: sent.append((pid, sig)) or True)
    monkeypatch.setattr(rp, "_exited", lambda pid: True)
    monkeypatch.setattr(rp.cgroups, "cgroup_of",
                        lambda pid, proc_root="/proc": None)
    apps = [{"pid": 4321, "used_mb": 900, "name": "x"}]
    killed = rp.kill_orphan_cuda(set(), [], apps)
    assert [s for _, s in sent][0] == signal.SIGTERM
    assert [d["pid"] for d in killed] == [4321]


def test_orphan_sweep_sigkills_what_survives_the_grace(monkeypatch):
    sent = []
    monkeypatch.setattr(rp, "_signal",
                        lambda pid, sig: sent.append((pid, sig)) or True)
    monkeypatch.setattr(rp, "_exited", lambda pid: False)
    monkeypatch.setattr(rp, "ORPHAN_TERM_GRACE_S", 0.0)
    monkeypatch.setattr(rp.cgroups, "cgroup_of",
                        lambda pid, proc_root="/proc": None)
    apps = [{"pid": 4321, "used_mb": 900, "name": "x"}]
    rp.kill_orphan_cuda(set(), [], apps)
    assert [s for _, s in sent] == [signal.SIGTERM, signal.SIGKILL]


def test_every_victim_is_sigtermed_before_any_is_sigkilled(monkeypatch):
    # This is what "batched" means, and it is observable in the signal
    # order alone -- no clock control needed. A per-victim grace would
    # interleave TERM, KILL, TERM, KILL...; one shared grace sends every
    # TERM first. The difference matters because a per-victim ladder
    # stalls the runner tick by N x grace instead of by one.
    sent = []
    monkeypatch.setattr(rp, "_signal",
                        lambda pid, sig: sent.append((pid, sig)) or True)
    monkeypatch.setattr(rp, "_exited", lambda pid: False)
    monkeypatch.setattr(rp, "ORPHAN_TERM_GRACE_S", 0.0)
    monkeypatch.setattr(rp.cgroups, "cgroup_of",
                        lambda pid, proc_root="/proc": None)
    apps = [{"pid": p, "used_mb": 1, "name": "x"} for p in (2791919, 2792864, 2765642, 2761761)]
    rp.kill_orphan_cuda(set(), [], apps)
    sigs = [s for _, s in sent]
    assert sigs == [signal.SIGTERM] * 4 + [signal.SIGKILL] * 4


def test_a_kill_records_the_victims_cgroup(monkeypatch):
    # The field that tells an operator it was their container rather than
    # their algorithm. Read before signalling: /proc/<pid>/cgroup is gone
    # the moment the process is.
    monkeypatch.setattr(rp, "_signal", lambda pid, sig: True)
    monkeypatch.setattr(rp, "_exited", lambda pid: True)
    monkeypatch.setattr(
        rp.cgroups, "cgroup_of",
        lambda pid, proc_root="/proc": "/system.slice/docker-abc.scope")
    apps = [{"pid": 4321, "used_mb": 900, "name": "tig-runtime"}]
    killed = rp.kill_orphan_cuda(set(), [], apps)
    assert killed[0]["cgroup"] == "/system.slice/docker-abc.scope"
    assert killed[0]["name"] == "tig-runtime"


def test_an_exempt_process_is_never_signalled(monkeypatch):
    sent = []
    monkeypatch.setattr(rp, "_signal",
                        lambda pid, sig: sent.append((pid, sig)) or True)
    monkeypatch.setattr(rp, "_exited", lambda pid: True)
    monkeypatch.setattr(rp.cgroups, "cgroup_of",
                        lambda pid, proc_root="/proc": None)
    apps = [{"pid": 4321, "used_mb": 900, "name": "x"}]
    assert rp.kill_orphan_cuda({4321}, [], apps) == []
    assert sent == []
```

`tests/test_reaper.py` binds the module as `from gpuqueue import reaper as rp` — use that alias. It does **not** import `signal`; add `import signal` at the top.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_reaper.py -q -k "sigterm or sigkills or grace or cgroup or signalled"`
Expected: FAIL — `kill_orphan_cuda` returns ints and only ever sends SIGKILL.

- [ ] **Step 3: Write the implementation**

In `src/gpuqueue/reaper.py`, add `from . import cgroups` to the imports and a constant beside `MAX_ATTEMPTS`:

```python
# Shorter than `_kill_tree`'s 10s on purpose. A convicted holder is one
# we want to checkpoint; an unledgered process is contending for a card
# someone may be blocked on, and this only has to be long enough for a
# SIGTERM handler to write a line. Paid once per `orphan_cuda_interval_s`
# inside the timer-gated sweep, not on every tick.
ORPHAN_TERM_GRACE_S = 5.0
```

Replace the body of `kill_orphan_cuda` from `_, unledgered = ledger.attribute(...)` onward — **keep the entire existing comment block at :63-92 verbatim**, it is the issue #19 history and still true — and replace only the final four lines:

```python
    exempt = set(protect) | own_pids()
    victims = [a for a in unledgered if a["pid"] not in exempt]
    if not victims:
        return []
    # Read before signalling: /proc/<pid>/cgroup is gone the moment the
    # process is, and this field is the one that tells the victim's
    # operator it was their container and not their algorithm.
    killed = [{"pid": a["pid"], "name": a.get("name"),
               "used_mb": a.get("used_mb"),
               "cgroup": cgroups.cgroup_of(a["pid"])}
              for a in victims]
    # SIGTERM everything, then one shared grace, then SIGKILL what is
    # left. Batched rather than per-victim: a grace each would stall the
    # runner tick by N x grace, and there is no reason the second
    # victim's grace should start after the first's has finished.
    #
    # The ladder at all because a SIGKILLed process writes no stderr. Its
    # caller sees `exit -9` with an empty message and reads it as its own
    # bug -- on 2026-09-01 an agent rewrote a correct index and submitted
    # a worse method on that reading (issue #24). `_kill_tree` has had
    # this since it was written; the orphan sweep never did.
    alive = [d for d in killed if _signal(d["pid"], signal.SIGTERM)]
    if alive:
        deadline = time.monotonic() + ORPHAN_TERM_GRACE_S
        while time.monotonic() < deadline:
            alive = [d for d in alive if not _exited(d["pid"])]
            if not alive:
                break
            time.sleep(0.1)
        for d in alive:
            _kill(d["pid"])
    return killed
```

Update the type annotation on the `def` line:

```python
def kill_orphan_cuda(protect: set[int], records: list,
                     apps: list[dict]) -> list[dict]:
```

In `reap` (:395), rename the local and derive both views:

```python
                killed = kill_orphan_cuda(protect, records, apps)
```

stays as-is; change only the return dict (:410-412):

```python
    return {"stale_claims": stale, "stuck_claims": stuck,
            "requeued": requeued, "failed": failed,
            # Two views of one list so they cannot drift. `killed_pids`
            # is what `runner.py` has always logged and what the suite
            # asserts on; `killed_details` is what the kill record needs.
            "killed_pids": [d["pid"] for d in killed],
            "killed_details": killed,
            "cleaned_paths": cleaned,
            "convicted": convicted, "exemption_dirs": exemption_dirs}
```

`killed` is initialised as `killed, convicted = [], []` at :354, so the empty case yields `[]` for both keys.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_reaper.py -q`
Expected: PASS, including the twelve pre-existing `killed_pids == [...]` assertions.

- [ ] **Step 5: Mutation-check**

1. Replace the ladder with the old `if _kill(app["pid"])`. Expected: `test_orphan_sweep_sigterms_before_it_sigkills` FAILS. Restore.
2. Replace the batched ladder with a per-victim one (`for d in killed: _signal(TERM); sleep(grace); _kill(...)`). Expected: `test_every_victim_is_sigtermed_before_any_is_sigkilled` FAILS on the interleaved signal order. Restore.
3. Delete the `cgroup` key from the `killed` dicts. Expected: `test_a_kill_records_the_victims_cgroup` FAILS with `KeyError`. Restore.

- [ ] **Step 6: Run the full suite and commit**

```bash
.venv/bin/python -m pytest -q
git add src/gpuqueue/reaper.py tests/test_reaper.py
git commit -m "fix: give the orphan sweep a signal its victim can survive

A SIGKILLed process writes no stderr, so its caller sees exit -9 with an
empty message and reads it as its own bug. On 2026-09-01 an agent read
it that way, rewrote a correct IVF index and submitted a worse method
(#24). The VRAM watchdog's _kill_tree has had SIGTERM -> grace ->
SIGKILL since it was written for exactly this reason; the orphan sweep
went straight to SIGKILL.

Batched: SIGTERM all, one shared grace, SIGKILL the survivors, so the
tick stalls by one grace rather than by N. Each victim's cgroup is read
before signalling -- that is the field that tells an operator it was
their container and not their algorithm."
```

---

### Task 6: A kill record the victim's operator can read

**Files:**
- Create: `src/gpuqueue/killlog.py`
- Modify: `src/gpuqueue/reaper.py` (`reap`, after `kill_orphan_cuda`)
- Modify: `src/gpuqueue/cli_gpuq.py` (`_cmd_kills`, parser at :231-236)
- Modify: `skills/gpu-jobs/SKILL.md`
- Test: `tests/test_killlog.py`, `tests/test_cli_gpuq.py`

**Interfaces:**
- Consumes: `reap()`'s `killed_details` (Task 5), `_consulted_dirs` (existing, `reaper.py:311`).
- Produces:
  - `killlog.append(queue_root: Path, entries: list[dict], consulted: list[str]) -> None`
  - `killlog.read(queue_root: Path, limit: int | None = None) -> list[dict]`
  - `killlog.KILLS_FILENAME = "kills.jsonl"`, `killlog.MAX_ENTRIES = 1000`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_killlog.py`:

```python
import json

from gpuqueue import killlog


ENTRY = {"pid": 2791919, "name": "tig-runtime build-index",
         "used_mb": 900, "cgroup": "/system.slice/docker-43faa0ee.scope"}


def test_append_writes_a_readable_record(tmp_path):
    killlog.append(tmp_path, [ENTRY], ["/workspace/lock/gpu"])
    got = killlog.read(tmp_path)
    assert len(got) == 1
    assert got[0]["pid"] == 2791919
    assert got[0]["cgroup"] == "/system.slice/docker-43faa0ee.scope"
    assert got[0]["reason"] == "orphan_sweep_unledgered"
    assert got[0]["ledgers_consulted"] == ["/workspace/lock/gpu"]
    assert got[0]["ts"]


def test_append_is_capped(tmp_path):
    # A rare-event log with no bound is how a box fills its disk.
    # Seeded in one write rather than 1050 appends: each append rewrites
    # the whole file, so the loop form is quadratic for no extra cover.
    seed = [dict(ENTRY, pid=i) for i in range(killlog.MAX_ENTRIES)]
    killlog.append(tmp_path, seed, [])
    killlog.append(tmp_path, [dict(ENTRY, pid=999001)], [])
    got = killlog.read(tmp_path)
    assert len(got) == killlog.MAX_ENTRIES
    # The cap keeps the NEWEST, which is the half an operator is reading;
    # dropping from the wrong end would leave a log that never updates.
    assert got[-1]["pid"] == 999001
    assert got[0]["pid"] == 1


def test_read_of_an_absent_file_is_empty_not_an_error(tmp_path):
    assert killlog.read(tmp_path) == []


def test_a_corrupt_line_does_not_hide_the_good_ones(tmp_path):
    # Same posture as `lg._load`: garbage must not blind a reader to
    # the records around it.
    killlog.append(tmp_path, [ENTRY], [])
    p = tmp_path / killlog.KILLS_FILENAME
    p.write_text(p.read_text() + "{not json\n")
    killlog.append(tmp_path, [dict(ENTRY, pid=7)], [])
    assert [e["pid"] for e in killlog.read(tmp_path)] == [2791919, 7]


def test_read_honours_a_limit(tmp_path):
    for i in range(5):
        killlog.append(tmp_path, [dict(ENTRY, pid=i)], [])
    assert [e["pid"] for e in killlog.read(tmp_path, limit=2)] == [3, 4]


def test_append_of_nothing_creates_no_file(tmp_path):
    killlog.append(tmp_path, [], [])
    assert not (tmp_path / killlog.KILLS_FILENAME).exists()
```

Append to `tests/test_cli_gpuq.py`:

```python
def test_kills_prints_recent_kills(tmp_path, monkeypatch, capsys):
    # The thing an agent that sees `killed by signal 9` can be TOLD to
    # run. A file nobody thinks to open is barely better than the runner
    # log nobody thinks to open, which is #24's actual complaint.
    from gpuqueue import killlog
    monkeypatch.setenv("QUEUE_ROOT", str(tmp_path))
    killlog.append(tmp_path, [{"pid": 2791919, "name": "tig-runtime",
                               "used_mb": 900,
                               "cgroup": "/system.slice/docker-abc.scope"}],
                   ["/workspace/lock/gpu"])
    assert main(["kills"]) == 0
    out = capsys.readouterr().out
    assert "2791919" in out
    assert "docker-abc.scope" in out


def test_kills_with_no_kills_says_so(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("QUEUE_ROOT", str(tmp_path))
    assert main(["kills"]) == 0
    assert "no kills" in capsys.readouterr().out.lower()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_killlog.py tests/test_cli_gpuq.py -q -k "kill"`
Expected: FAIL — `ModuleNotFoundError: No module named 'gpuqueue.killlog'`

- [ ] **Step 3: Implement `killlog.py`**

```python
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


def read(queue_root, limit: int | None = None) -> list[dict]:
    """Recorded kills, oldest first. A corrupt line is skipped, not fatal.

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
    if limit is None:
        return out
    # Not `out[-limit:] if limit else out`: `--limit 0` falls to the
    # falsy branch and prints everything, and `out[-0:]` is the whole
    # list anyway, so even an `is not None` guard reads zero as "no
    # limit".
    return out[-limit:] if limit > 0 else []
```

- [ ] **Step 4: Wire it into `reap`**

In `src/gpuqueue/reaper.py`, add `from . import killlog` to the imports, and immediately after the `killed = kill_orphan_cuda(protect, records, apps)` line (:395):

```python
                # Written here rather than in `kill_orphan_cuda` because
                # that function does not know the queue root, and giving
                # it one would tie the kill decision to the queue's
                # layout. `exemption_dirs` is already in hand on this
                # line, which is the whole reason the record can name
                # what was consulted.
                killlog.append(queue.root, killed, exemption_dirs)
```

- [ ] **Step 5: Add the `gpuq kills` subcommand**

In `src/gpuqueue/cli_gpuq.py`, add `from . import killlog` to the imports, a command function beside `_cmd_bug`:

```python
def _cmd_kills(args) -> int:
    """What the orphan sweep killed, most recent last.

    Exists because #24's victim had no way to tell a queue kill from its
    own crash: SIGKILL writes no stderr, so the caller sees `exit -9` and
    an empty message. An agent that sees a signal death runs this.
    """
    q = _queue(args)
    entries = killlog.read(q.root, limit=args.limit)
    if not entries:
        print("no kills recorded")
        return 0
    total = len(killlog.read(q.root))
    if total > len(entries):
        # Say so rather than truncating quietly. An operator who sees
        # four kills and had five is chasing the wrong window.
        print(f"showing the most recent {len(entries)} of {total} "
              f"-- pass --limit {total} for all")
    for e in entries:
        print(f"{e.get('ts', '?')}  pid {e.get('pid')}  "
              f"{e.get('used_mb') or '?'} MiB  {e.get('name') or '?'}")
        print(f"    cgroup:  {e.get('cgroup') or '(none read)'}")
        print(f"    reason:  {e.get('reason')}")
        print(f"    ledgers: "
              f"{', '.join(e.get('ledgers_consulted') or ['(none)'])}")
    return 0
```

and register it after the `bug` parser (:236):

```python
    k = sub.add_parser("kills",
                       help="what the orphan sweep killed and why")
    k.add_argument("--limit", type=int, default=20)
    k.set_defaults(func=_cmd_kills)
```

- [ ] **Step 6: Point the skill at it**

In `skills/gpu-jobs/SKILL.md`, add a section:

```markdown
## A job died with signal 9 and no message

A `SIGKILL` writes no stderr, so an `exit -9` with an empty message is
not evidence of a bug in your own code. The queue's orphan sweep kills
CUDA processes that no live claim accounts for, and that is what it
looks like from the victim's side.

Run `gpuq kills` first. If your pid is there, the queue killed it and
the `cgroup` line names what it killed — check whether the work was
running somewhere your claim did not cover. CUDA inside a container is
the usual case, and the fix is to claim for it:

    gpu-claim --vram-mb <MiB> \
      --scope-pid $(docker inspect -f '{{.State.Pid}}' <container>) \
      -- <your command>

If your pid is *not* there, the queue did not kill it and the failure is
somewhere else.
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_killlog.py tests/test_cli_gpuq.py -q`
Expected: PASS

- [ ] **Step 8: Mutation-check**

1. Remove the `[-MAX_ENTRIES:]` slice. Expected: `test_append_is_capped` FAILS. Restore.
2. Change the `except ValueError: continue` in `read` to `raise`. Expected: `test_a_corrupt_line_does_not_hide_the_good_ones` FAILS. Restore.
3. Delete the `if not entries: return` guard. Expected: `test_append_of_nothing_creates_no_file` FAILS. Restore.

- [ ] **Step 9: Run the full suite and commit**

```bash
.venv/bin/python -m pytest -q
git add src/gpuqueue/killlog.py src/gpuqueue/reaper.py \
        src/gpuqueue/cli_gpuq.py tests/test_killlog.py \
        tests/test_cli_gpuq.py skills/gpu-jobs/SKILL.md
git commit -m "feat: gpuq kills, so a victim can tell a sweep from its own bug

Closes the visibility half of #24. The runner log already named every
kill, and that was not enough: it is on the box, the killed process
cannot read it, and its owner does not think to check it -- because from
their side it looks like their own crash.

A subcommand is the part that matters. An agent that sees `killed by
signal 9` can be told to run `gpuq kills`; a file nobody thinks to open
is barely better than a log nobody thinks to open. The skill now says
so, and the record names the victim's cgroup and the ledgers the sweep
consulted."
```

---

## Verification on the deployment box

After Task 6, before closing #24. `tig-gpu` is reachable over ssh and
`tig-pentesting-tig-scorer-1` is live.

### Already done, read-only, 2026-09-01

These were measured on the box without deploying anything or restarting
the runner. They confirm the design's premises; they do **not** replace
Steps 2-3, which are still outstanding.

- **Every deployment fact in the spec's §1 table still holds.**
  `/proc/1/cgroup` is `0::/init.scope` (cgroup v2 unified, root
  namespace); the runner is a **host** process (pid 2892850) in
  `/system.slice/supervisor.service`; the scorer is up at init pid
  2818873, cgroup `/system.slice/docker-43faa0ee4d16….scope` — the same
  pid and cgroup the spec recorded.
- **The premise re-measured against the live container.** A
  `docker exec`'d process had ppid **2818851**, the containerd-shim, not
  container init 2818873. `ps --ppid` descendants of init returned
  **nothing**; the cgroup contained **both** processes. This is exactly
  why a pid tree cannot express containerised CUDA.
- **`cgroups.py` answers correctly against a real `/proc`** (module
  copied to `/tmp` and run standalone — it imports nothing from
  `gpuqueue`, so this needed no install): `in_scope(exec'd, scope)` True,
  `in_scope(host shell, scope)` False, `refuse_reason(container scope)`
  None, `scope_process_count` 2, and both a login session and `/`
  refused with their intended messages.
- **The claim-directory divergence is LIVE on this box**, which is what
  makes `preflight.own_scopes()` load-bearing rather than defensive:

  | | `$GPU_CLAIM_DIR` | `all_claim_dirs()` |
  |---|---|---|
  | runner (supervisor unit) | `/workspace/lock/gpu` | `['/workspace/lock/gpu', '/var/lock/gpu']` |
  | interactive shell | *unset* | `['/var/lock/gpu']` |

  `gpuq.toml` sets no `[queue].claim_dir`. So a `--scope-pid` claim taken
  from a shell lands in `/var/lock/gpu`, while a scope exemption built
  from `cfg.claim_dir or claim_dir()` in the runner reads only
  `/workspace/lock/gpu`. Without `own_scopes()` reading every directory,
  that claim is invisible and the container is SIGKILLed anyway — issue
  #19's shape, reached through the new feature.

### Done on the box, 2026-09-01

All four steps ran against the live `tig-pentesting-tig-scorer-1` at
deployed commit `e72995e`. Recorded on #24; the summary is below and the
checkboxes are ticked in place.

`kill_orphan_cuda` is `false` on this box for the c004 experiment. It was
turned on for the verification window and restored afterwards — `diff`
against the pre-verification backup is empty.

- **The premise, re-measured.** A `docker exec`'d CUDA process had ppid
  **2818851**, the containerd-shim, not container init 2818873.
- **Step 2, no claim:** `16:30:22 orphan sweep SIGTERMed unledgered CUDA
  pid 2943713`, dead ~50s after starting. `gpuq kills` named the scorer's
  cgroup. The line says SIGTERMed with no escalation clause, so the victim
  took the SIGTERM inside the grace.
- **Step 3, scoped claim:** taken with `$GPU_CLAIM_DIR` unset, so the
  record landed in `/var/lock/gpu` against the runner's
  `/workspace/lock/gpu` — the divergent case, which is the only one where
  `own_scopes()` breadth matters. `gpu-claim` printed `scope
  /system.slice/docker-43faa0ee….scope (4 live processes)` and did **not**
  refuse, though the CUDA was already on the card before the claim: §5's
  trap. The holder survived **170s**, about three sweep intervals.
- **It discriminates.** Claim released; the same two pids were swept three
  minutes later, in one line — `SIGTERMed unledgered CUDA pids 2944556,
  2944895` — which is the batched ladder's one shared grace, observed live.

- [x] **Step 1: Deploy and restart the runner**

The deployed tree is an **editable install at
`/workspace/gpu-queue-management`** (on `main` at 9a5da12 as of
2026-09-01), with the venv at `/opt/gpuq/venv`. An earlier draft of this
plan said `/opt/gpuq/src`; **that path does not exist** and the command
failed at `cd`. Because the install is editable, no `pip install -e .` is
needed for a pure-Python change — checkout and restart is enough.

```bash
ssh tig-gpu 'cd /workspace/gpu-queue-management && git fetch && \
  git checkout <branch> && supervisorctl restart gpuq-runner'
```

- [x] **Step 2: Confirm the sweep kills an unclaimed container process**

Drive CUDA work through the scorer with no claim held, and watch:

```bash
ssh tig-gpu 'tail -f /workspace/queue/logs/runner.log'
```

Expected: an `orphan sweep SIGTERMed` line, then `gpuq kills` shows the
pid with the scorer's cgroup. This is the **discriminating** half — if
this does not kill, the test that follows proves nothing.

- [x] **Step 3: Confirm a scoped claim spares it**

```bash
ssh tig-gpu 'PID=$(docker inspect -f "{{.State.Pid}}" tig-pentesting-tig-scorer-1); \
  gpu-claim --vram-mb 3000 --scope-pid $PID -- sleep 300'
```

Drive the same CUDA work through the scorer while that claim is held,
across at least two sweep intervals (>120s at the default 60s).

Expected: no kill; `gpu-claim --status` shows the scope; `nvidia-smi`
shows the work completing.

Run this from a shell with `$GPU_CLAIM_DIR` **unset**, not exported to
match the runner's. The divergence measured above is the realistic case,
and a shell that borrows the runner's value would test the one
configuration the bug could not occur in.

- [x] **Step 4: Record the result on the issue**

Post both outcomes to #24 — the kill without a claim and the survival
with one. A single "it worked" does not distinguish a working exemption
from a sweep that never ran.

---

## Self-review notes

**Spec coverage:** §3 → Task 1. §4 → Task 2. §2 + the `attribute` change → Task 3. §5 → Task 4. §6 → Task 5. §7 → Task 6. §8 → the mutation-check step in every task plus the deployment verification. §1's `design.md:24` correction → Task 1 Step 6. §9's failure modes are behaviour, not code, and are covered by `scope_is_live`'s tests in Task 2.

**Not covered by any task, deliberately:** §Out of scope's three items — `--container` sugar, the `gpuq bug` / `[autofix].enabled` coupling, and cgroup enforcement.

**Type consistency:** `cgroup_of(pid, proc_root)` keeps that signature in Tasks 1–5. `refuse_reason` takes a resolved path (never None) — `cli_claim._resolve_scope` handles the None case before calling it. `kill_orphan_cuda` returns `list[dict]` from Task 5 onward, and `reap`'s `killed_pids` stays `list[int]` throughout so no pre-existing assertion changes.
