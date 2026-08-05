# gpu-queue-management Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the four components in `docs/design.md` — `gpu-claim`, `gpuq`, `gpuq-runner`, `bootstrap.sh` — so a shared single-GPU box serializes GPU work behind an advisory lock and runs queued jobs without anyone holding an ssh session.

**Architecture:** One Python package, `gpuqueue`, exposing three console scripts. State lives in a directory tree; transitions are `os.rename` within one filesystem. The GPU lock is `flock` on a file named by the normalized GPU UUID. The runner is a **single-threaded** process: one `tick()` starts subprocesses, polls the ones already running, performs *all* git operations, and reaps. Jobs run concurrently because they are separate *processes*, not because the runner has threads.

**Tech Stack:** Python 3.11+, standard library only at runtime (`fcntl`, `subprocess`, `tomllib`, `json`, `pathlib`), `pytest` for tests, `nvidia-smi` for GPU identity, supervisor for process management.

## Global Constraints

- **Python 3.11+, standard library only at runtime.** 3.11 is the floor because `tomllib` arrives there; a `tomli` backport would be a dependency, and this package installs onto boxes with arbitrary ML stacks where any dependency is a chance to conflict with theirs. `pytest` as a test-only extra is fine.
- **Never import `torch`.** Not even guarded, not even for GPU identity. A JAX or TensorFlow project must be able to use this lock, and importing torch inside the runner daemon costs seconds and initializes a CUDA context in the one process that should never touch the card. GPU identity comes from `nvidia-smi`. `normalize_gpu_uuid` still exists — see "Design gaps resolved", gap 2 — precisely because *other* implementations of the protocol will use torch.
- **The runner has no threads.** No `threading`, no `concurrent.futures`, no locks. `tick()` is one non-blocking pass and is the entire concurrency story; if a change seems to need a worker thread, it needs a redesign instead. Shared mutable state between a worker and the main loop is the failure mode this rules out by construction.
- **Every test must pass without a GPU.** CI and dev machines have none. `nvidia-smi` is shelled out through one seam per module (`_run`) that tests monkeypatch. A test requiring real CUDA is a broken test.
- **Unprivileged container.** No root, no Docker, no kernel modules, no sysctls. At runtime, never write outside `$QUEUE_ROOT`, `$GPU_CLAIM_DIR`, and configured checkouts. `bootstrap.sh` is the one exception and writes exactly two things beyond those: the supervisor program file, and the agent skill under `$GPUQ_SKILLS_DIR` (default `~/.claude/skills/gpu-jobs/`). Both are install-time, both are idempotent, and neither is touched once the runner is up.
- **Queue root is one filesystem.** `os.rename` is only atomic within a filesystem; the code must never rename across `$QUEUE_ROOT` and anywhere else.
- **Jobs never invoke git.** Every git subprocess call lives in `git_ops.py` and is made by the runner's `tick()`, between the poll of one job and the start of the next. A job's own `cmd` running git against a checkout it does not own is out of the runner's control, but nothing in this package may do it.
- **The lock file format is a pinned protocol**, not an implementation detail: lock path `$GPU_CLAIM_DIR/<normalized-uuid>.lock`, claim JSON alongside it with the keys `pid`, `owner`, `cmd`, `started_at`, `key`. A second implementation on the same box must interoperate, so changing any of these is a version bump documented in `docs/design.md`, not a refactor.
- **All timestamps are UTC ISO-8601** with a `Z` suffix, e.g. `2026-08-05T12:00:00Z`.
- **Per-job VRAM/memory limits are OUT OF SCOPE** for this plan, by explicit decision on 2026-08-05. The GPU lane is one slot; a job that holds the card holds all of it. Do not add a `vram_mb` field, a packing scheduler, or `PYTORCH_CUDA_ALLOC_CONF` manipulation. A later phase will revisit it.

## Design gaps resolved by this plan

Three things `docs/design.md` does not settle. All are decided here; if you disagree, raise it before implementing rather than improvising.

**1. Concurrent jobs at different pinned commits.** The design gives each project one checkout and pins `commit` per job. Two concurrent CPU jobs at different commits cannot share one working tree — the second job's checkout rewrites the tree under the first one's feet, mid-run. **Resolution:** `tick()` creates a per-job detached `git worktree` at the pinned commit under `$QUEUE_ROOT/work/<id>`, and removes it after artifacts are collected. The long-lived checkout remains the only place commits are made. Git stays serialized because only `tick()` calls it.

**2. GPU UUID normalization.** This package derives identity from `nvidia-smi --query-gpu=uuid`, which returns `GPU-<hex>`. But the design names `torch.cuda.get_device_properties(d).uuid` as the key derivation, and that returns a `uuid.UUID` whose `str()` is bare hyphenated hex. A torch-based producer and this one would take *different lock files for the same physical card* — the precise failure UUID-keying exists to prevent. **Resolution:** `normalize_gpu_uuid()` strips a leading `GPU-`/`MIG-`, lowercases, and strips surrounding whitespace; every producer goes through it before the filename is formed, and the normalized form is what the pinned protocol specifies. Task 4 tests that both spellings collapse to one key. This is the one place where not importing torch still obliges us to care what torch emits.

**3. Concurrency without threads.** Several jobs must run at once, and the obvious shape — a thread per job — puts a worker and the main loop on the same queue files, the same lane counters and the same `JobSpec` objects, with a lock around each. **Resolution:** the runner is single-threaded. `subprocess.Popen` already gives concurrency; `tick()` starts what the lanes allow, polls what is already running, and settles what has finished. Nothing is shared because nothing is concurrent inside the process. The GPU claim is entered non-blocking in `tick()` and held across ticks (`cm.__enter__()` / `cm.__exit__()` by hand) rather than wrapped around a blocking `wait()` — a `wait=True` claim in a single-threaded loop would deadlock the whole runner behind one card.

---

## File Structure

```
pyproject.toml                  package metadata, console_scripts, pytest config
src/gpuqueue/__init__.py
src/gpuqueue/spec.py            JobSpec dataclass: validation, JSON round-trip
src/gpuqueue/queue.py           QueueRoot: directory tree, atomic transitions, dedupe
src/gpuqueue/gpuid.py           GPU identity: derivation + normalization
src/gpuqueue/claim.py           flock acquire/release, claim file read/write
src/gpuqueue/preflight.py       foreign CUDA process detection
src/gpuqueue/config.py          TOML config: QueueConfig, ProjectConfig
src/gpuqueue/git_ops.py         clone, worktree add/remove, commit artifacts (runner loop only)
src/gpuqueue/executor.py        start / poll / kill one job's subprocess, with the watchdog
src/gpuqueue/reaper.py          dead claims, orphan CUDA procs, .part files, requeue-once
src/gpuqueue/runner.py          the single-threaded tick: reap, collect, admit
src/gpuqueue/cli_claim.py       `gpu-claim` entry point
src/gpuqueue/cli_gpuq.py        `gpuq` entry point
src/gpuqueue/cli_runner.py      `gpuq-runner` entry point
supervisor/gpuq-runner.conf     supervisor program file, shipped not hand-written
skills/gpu-jobs/SKILL.md        how an agent submits work instead of running it
bootstrap.sh                    bare box -> running runner, idempotent
tests/…                         one test module per source module
```

Each module has one responsibility and no upward dependencies: `spec` and `gpuid` depend on nothing; `queue`, `claim`, `preflight` depend on those; `runner` depends on everything and is depended on by nothing but its CLI. `git_ops` is imported by `runner` alone, which is what makes "only the loop runs git" checkable by reading the import graph rather than by trusting a comment.

---

### Task 1: Package scaffold and JobSpec

**Files:**
- Create: `pyproject.toml`, `src/gpuqueue/__init__.py`, `src/gpuqueue/spec.py`
- Test: `tests/test_spec.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `JobSpec` dataclass with fields `id: str`, `lane: str`, `project: str`, `commit: str`, `branch: str`, `cmd: list[str]`, `artifacts: list[str]`, `timeout_s: int`, `attempts: int`, `dedupe_key: str | None`, `submitted_at: str`, `pid: int | None`, `exit_code: int | None`, `error: str | None`. Methods `JobSpec.from_dict(d) -> JobSpec`, `JobSpec.to_dict() -> dict`, `JobSpec.validate() -> None` (raises `SpecError`). Exception `SpecError(ValueError)`. Constant `LANES = ("cpu", "gpu")`. Helper `utcnow_iso() -> str`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_spec.py
import pytest
from gpuqueue.spec import JobSpec, SpecError, utcnow_iso

def _minimal(**over):
    d = {
        "id": "glove-v0-train-01",
        "lane": "gpu",
        "project": "wgan-synthetic",
        "commit": "a1b2c3d",
        "branch": "ds/glove",
        "cmd": ["python", "-m", "src.train", "--config", "c.yaml"],
        "artifacts": ["runs/glove/v0/summary.json"],
        "timeout_s": 21600,
        "attempts": 0,
        "dedupe_key": "glove:v0:a1b2c3d",
    }
    d.update(over)
    return d

def test_round_trip_preserves_fields():
    spec = JobSpec.from_dict(_minimal())
    assert JobSpec.from_dict(spec.to_dict()) == spec

def test_defaults_are_filled():
    d = _minimal()
    del d["attempts"]
    spec = JobSpec.from_dict(d)
    assert spec.attempts == 0
    assert spec.pid is None
    assert spec.submitted_at.endswith("Z")

def test_unknown_lane_rejected():
    with pytest.raises(SpecError, match="lane"):
        JobSpec.from_dict(_minimal(lane="tpu")).validate()

def test_empty_cmd_rejected():
    with pytest.raises(SpecError, match="cmd"):
        JobSpec.from_dict(_minimal(cmd=[])).validate()

def test_nonpositive_timeout_rejected():
    with pytest.raises(SpecError, match="timeout_s"):
        JobSpec.from_dict(_minimal(timeout_s=0)).validate()

def test_id_with_path_separator_rejected():
    with pytest.raises(SpecError, match="id"):
        JobSpec.from_dict(_minimal(id="../escape")).validate()

def test_absolute_artifact_path_rejected():
    with pytest.raises(SpecError, match="artifact"):
        JobSpec.from_dict(_minimal(artifacts=["/etc/passwd"])).validate()

def test_utcnow_iso_format():
    assert utcnow_iso().endswith("Z")
    assert "T" in utcnow_iso()
```

`test_id_with_path_separator_rejected` and `test_absolute_artifact_path_rejected` are not decoration: the id becomes a filename under `$QUEUE_ROOT` and artifacts are copied out of a worktree. Both are traversal vectors.

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_spec.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gpuqueue'`

- [ ] **Step 3: Write the package scaffold**

```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "gpu-queue-management"
version = "0.1.0"
description = "Host-level GPU arbitration and job queueing for shared single-GPU boxes"
requires-python = ">=3.11"
dependencies = []

[project.optional-dependencies]
dev = ["pytest>=8"]

[project.scripts]
gpu-claim = "gpuqueue.cli_claim:main"
gpuq = "gpuqueue.cli_gpuq:main"
gpuq-runner = "gpuqueue.cli_runner:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
```

Leave `src/gpuqueue/__init__.py` empty.

- [ ] **Step 4: Write the minimal implementation**

```python
# src/gpuqueue/spec.py
"""Job specification: the unit the queue moves between directories."""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

LANES = ("cpu", "gpu")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


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
    attempts: int = 0
    dedupe_key: str | None = None
    submitted_at: str = field(default_factory=utcnow_iso)
    pid: int | None = None
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
        if not self.cmd or not all(isinstance(a, str) for a in self.cmd):
            raise SpecError("cmd must be a non-empty list of strings")
        if not isinstance(self.timeout_s, int) or self.timeout_s <= 0:
            raise SpecError(f"timeout_s must be a positive int, got {self.timeout_s!r}")
        if self.attempts < 0:
            raise SpecError("attempts must be >= 0")
        for a in self.artifacts:
            if a.startswith("/") or ".." in a.split("/"):
                raise SpecError(f"artifact path must be relative and contained: {a!r}")
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pip install -e . && pytest tests/test_spec.py -v`
Expected: 8 passed

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/gpuqueue/__init__.py src/gpuqueue/spec.py tests/test_spec.py
git commit -m "feat: package scaffold and JobSpec with validation"
```

---

### Task 2: Queue tree and atomic transitions

**Files:**
- Create: `src/gpuqueue/queue.py`
- Test: `tests/test_queue.py`

**Interfaces:**
- Consumes: `JobSpec`, `SpecError`, `utcnow_iso` from `gpuqueue.spec`.
- Produces: `QueueRoot(root: Path)` with `ensure_dirs() -> None`, `submit(spec: JobSpec) -> str` (returns the id actually queued — an existing id if deduped), `claim(job_id: str) -> JobSpec | None` (atomic pending→running; `None` if someone else won), `update(spec: JobSpec, state: str = "running") -> None` (rewrite a spec in place, atomically), `finish(spec: JobSpec, ok: bool) -> None` (running→done/failed), `requeue(spec: JobSpec) -> None` (running→pending, `attempts += 1`), `list_state(state: str) -> list[JobSpec]`, `find(job_id: str) -> tuple[str, JobSpec] | None`, `cancel(job_id: str) -> bool`, `path_for(state, job_id) -> Path`, `log_paths(job_id) -> tuple[Path, Path]`, `work_dir(job_id) -> Path`. Constant `STATES = ("pending", "running", "done", "failed")`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_queue.py
import json
import pytest
from gpuqueue.queue import QueueRoot, STATES
from gpuqueue.spec import JobSpec

def mkspec(job_id="j1", **over):
    d = dict(id=job_id, lane="cpu", project="p", commit="abc",
             branch="main", cmd=["true"], artifacts=[], timeout_s=60)
    d.update(over)
    return JobSpec.from_dict(d)

@pytest.fixture
def q(tmp_path):
    qr = QueueRoot(tmp_path / "queue")
    qr.ensure_dirs()
    return qr

def test_ensure_dirs_is_idempotent(q):
    q.ensure_dirs()
    for s in STATES:
        assert (q.root / s).is_dir()
    assert (q.root / "logs").is_dir()

def test_submit_writes_pending_and_is_valid_json(q):
    q.submit(mkspec())
    p = q.root / "pending" / "j1.json"
    assert json.loads(p.read_text())["id"] == "j1"

def test_claim_moves_pending_to_running(q):
    q.submit(mkspec())
    spec = q.claim("j1")
    assert spec is not None
    assert not (q.root / "pending" / "j1.json").exists()
    assert (q.root / "running" / "j1.json").exists()

def test_second_claim_returns_none(q):
    q.submit(mkspec())
    assert q.claim("j1") is not None
    assert q.claim("j1") is None

def test_finish_ok_moves_to_done(q):
    q.submit(mkspec())
    spec = q.claim("j1")
    spec.exit_code = 0
    q.finish(spec, ok=True)
    assert (q.root / "done" / "j1.json").exists()

def test_finish_not_ok_moves_to_failed_and_keeps_error(q):
    q.submit(mkspec())
    spec = q.claim("j1")
    spec.exit_code = 1
    spec.error = "boom"
    q.finish(spec, ok=False)
    body = json.loads((q.root / "failed" / "j1.json").read_text())
    assert body["error"] == "boom"

def test_update_rewrites_in_place_without_changing_state(q):
    q.submit(mkspec())
    spec = q.claim("j1")
    spec.pid = 4242
    q.update(spec)
    assert json.loads((q.root / "running" / "j1.json").read_text())["pid"] == 4242
    assert q.find("j1")[0] == "running"

def test_requeue_increments_attempts(q):
    q.submit(mkspec())
    spec = q.claim("j1")
    q.requeue(spec)
    assert json.loads((q.root / "pending" / "j1.json").read_text())["attempts"] == 1

def test_dedupe_returns_existing_id_for_pending(q):
    q.submit(mkspec("j1", dedupe_key="k"))
    assert q.submit(mkspec("j2", dedupe_key="k")) == "j1"
    assert not (q.root / "pending" / "j2.json").exists()

def test_dedupe_also_matches_running(q):
    q.submit(mkspec("j1", dedupe_key="k"))
    q.claim("j1")
    assert q.submit(mkspec("j2", dedupe_key="k")) == "j1"

def test_dedupe_does_not_match_done(q):
    q.submit(mkspec("j1", dedupe_key="k"))
    q.finish(q.claim("j1"), ok=True)
    assert q.submit(mkspec("j2", dedupe_key="k")) == "j2"

def test_no_dedupe_key_never_dedupes(q):
    q.submit(mkspec("j1"))
    assert q.submit(mkspec("j2")) == "j2"

def test_duplicate_id_rejected(q):
    q.submit(mkspec("j1"))
    with pytest.raises(FileExistsError):
        q.submit(mkspec("j1"))

def test_find_reports_state(q):
    q.submit(mkspec())
    assert q.find("j1")[0] == "pending"
    q.claim("j1")
    assert q.find("j1")[0] == "running"
    assert q.find("nope") is None

def test_cancel_pending_moves_to_failed(q):
    q.submit(mkspec())
    assert q.cancel("j1") is True
    assert (q.root / "failed" / "j1.json").exists()

def test_cancel_running_returns_false(q):
    q.submit(mkspec())
    q.claim("j1")
    assert q.cancel("j1") is False

def test_list_state_skips_corrupt_files(q):
    q.submit(mkspec())
    (q.root / "pending" / "garbage.json").write_text("{not json")
    assert [s.id for s in q.list_state("pending")] == ["j1"]
```

`test_list_state_skips_corrupt_files` matters because the design promises the queue stays inspectable and repairable by hand — a half-written file must not take down `gpuq list`.

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_queue.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gpuqueue.queue'`

- [ ] **Step 3: Write the minimal implementation**

```python
# src/gpuqueue/queue.py
"""The queue: a directory tree whose transitions are atomic renames."""
from __future__ import annotations

import fcntl
import json
import os
from contextlib import contextmanager
from pathlib import Path

from .spec import JobSpec

STATES = ("pending", "running", "done", "failed")
_ACTIVE = ("pending", "running")


class QueueRoot:
    def __init__(self, root: Path):
        self.root = Path(root)

    # --- layout -------------------------------------------------------
    def ensure_dirs(self) -> None:
        for d in (*STATES, "logs", "work"):
            (self.root / d).mkdir(parents=True, exist_ok=True)

    def path_for(self, state: str, job_id: str) -> Path:
        if state not in STATES:
            raise ValueError(f"unknown state {state!r}")
        return self.root / state / f"{job_id}.json"

    def log_paths(self, job_id: str) -> tuple[Path, Path]:
        return (self.root / "logs" / f"{job_id}.out",
                self.root / "logs" / f"{job_id}.err")

    def work_dir(self, job_id: str) -> Path:
        return self.root / "work" / job_id

    # --- io -----------------------------------------------------------
    def _write(self, path: Path, spec: JobSpec) -> None:
        """Write via a temp file in the same directory, then rename.

        A reader must never see a half-written spec; rename(2) is the only
        way to publish one atomically.
        """
        tmp = path.with_suffix(".json.part")
        tmp.write_text(json.dumps(spec.to_dict(), indent=2) + "\n")
        os.rename(tmp, path)

    def _read(self, path: Path) -> JobSpec:
        return JobSpec.from_dict(json.loads(path.read_text()))

    @contextmanager
    def _submit_lock(self):
        """Serialize dedupe-check-then-write across concurrent submitters."""
        self.root.mkdir(parents=True, exist_ok=True)
        lock = self.root / ".submit.lock"
        fd = os.open(lock, os.O_CREAT | os.O_RDWR, 0o666)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            os.close(fd)

    # --- transitions --------------------------------------------------
    def submit(self, spec: JobSpec) -> str:
        spec.validate()
        with self._submit_lock():
            if spec.dedupe_key:
                for state in _ACTIVE:
                    for other in self.list_state(state):
                        if other.dedupe_key == spec.dedupe_key:
                            return other.id
            dest = self.path_for("pending", spec.id)
            if any(self.path_for(s, spec.id).exists() for s in STATES):
                raise FileExistsError(f"job id already exists: {spec.id}")
            self._write(dest, spec)
            return spec.id

    def claim(self, job_id: str) -> JobSpec | None:
        """pending -> running. None if another thread got there first."""
        src = self.path_for("pending", job_id)
        dst = self.path_for("running", job_id)
        try:
            os.rename(src, dst)
        except FileNotFoundError:
            return None
        return self._read(dst)

    def update(self, spec: JobSpec, state: str = "running") -> None:
        """Rewrite a spec where it already is. The runner records a pid this
        way, so the reaper can tell a live job from an abandoned one."""
        self._write(self.path_for(state, spec.id), spec)

    def finish(self, spec: JobSpec, ok: bool) -> None:
        state = "done" if ok else "failed"
        self._write(self.path_for("running", spec.id), spec)
        os.rename(self.path_for("running", spec.id),
                  self.path_for(state, spec.id))

    def requeue(self, spec: JobSpec) -> None:
        spec.attempts += 1
        spec.pid = None
        self._write(self.path_for("running", spec.id), spec)
        os.rename(self.path_for("running", spec.id),
                  self.path_for("pending", spec.id))

    def cancel(self, job_id: str) -> bool:
        """Cancel a pending job. Running jobs are the runner's to stop."""
        src = self.path_for("pending", job_id)
        try:
            spec = self._read(src)
        except FileNotFoundError:
            return False
        spec.error = "cancelled"
        self._write(src, spec)
        try:
            os.rename(src, self.path_for("failed", job_id))
        except FileNotFoundError:
            return False
        return True

    # --- queries ------------------------------------------------------
    def list_state(self, state: str) -> list[JobSpec]:
        d = self.root / state
        if not d.is_dir():
            return []
        out = []
        for p in sorted(d.glob("*.json")):
            try:
                out.append(self._read(p))
            except Exception:
                continue  # a hand-repairable queue tolerates one bad file
        return out

    def find(self, job_id: str) -> tuple[str, JobSpec] | None:
        for state in STATES:
            p = self.path_for(state, job_id)
            if p.exists():
                try:
                    return state, self._read(p)
                except Exception:
                    return None
        return None
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_queue.py -v`
Expected: 17 passed

- [ ] **Step 5: Commit**

```bash
git add src/gpuqueue/queue.py tests/test_queue.py
git commit -m "feat: queue tree with atomic rename transitions and dedupe"
```

---

### Task 3: `gpuq` CLI

**Files:**
- Create: `src/gpuqueue/cli_gpuq.py`
- Test: `tests/test_cli_gpuq.py`

**Interfaces:**
- Consumes: `QueueRoot`, `JobSpec`, `SpecError`.
- Produces: `main(argv: list[str] | None = None) -> int`. Subcommands `submit`, `list`, `show`, `cancel`. Helper `generate_id(prefix: str) -> str` returning `<prefix>-<YYYYmmddTHHMMSSZ>-<6 hex>`. Queue root resolution order: `--queue-root`, then `$QUEUE_ROOT`, then `/workspace/queue`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli_gpuq.py
import json
import pytest
from gpuqueue.cli_gpuq import main, generate_id

@pytest.fixture
def root(tmp_path, monkeypatch):
    r = tmp_path / "queue"
    monkeypatch.setenv("QUEUE_ROOT", str(r))
    return r

def test_submit_creates_pending_job(root, capsys):
    rc = main(["submit", "--project", "p", "--commit", "abc", "--branch", "main",
               "--lane", "cpu", "--id", "j1", "--", "python", "-c", "print(1)"])
    assert rc == 0
    assert capsys.readouterr().out.strip() == "j1"
    body = json.loads((root / "pending" / "j1.json").read_text())
    assert body["cmd"] == ["python", "-c", "print(1)"]
    assert body["lane"] == "cpu"

def test_submit_generates_id_when_omitted(root, capsys):
    assert main(["submit", "--project", "p", "--commit", "abc",
                 "--branch", "main", "--", "true"]) == 0
    job_id = capsys.readouterr().out.strip()
    assert (root / "pending" / f"{job_id}.json").exists()

def test_submit_default_lane_is_cpu(root, capsys):
    main(["submit", "--project", "p", "--commit", "abc", "--branch", "main",
          "--id", "j1", "--", "true"])
    assert json.loads((root / "pending" / "j1.json").read_text())["lane"] == "cpu"

def test_submit_dedupe_prints_existing_id(root, capsys):
    args = ["submit", "--project", "p", "--commit", "abc", "--branch", "main",
            "--dedupe-key", "k", "--", "true"]
    main(args + ["--id", "j1"][:0])
    first = capsys.readouterr().out.strip()
    main(args)
    assert capsys.readouterr().out.strip() == first

def test_submit_invalid_lane_exits_nonzero(root, capsys):
    rc = main(["submit", "--project", "p", "--commit", "abc", "--branch", "main",
               "--lane", "tpu", "--", "true"])
    assert rc == 2
    assert "lane" in capsys.readouterr().err

def test_submit_requires_cmd(root, capsys):
    assert main(["submit", "--project", "p", "--commit", "abc",
                 "--branch", "main", "--"]) == 2

def test_list_json_output(root, capsys):
    main(["submit", "--project", "p", "--commit", "abc", "--branch", "main",
          "--id", "j1", "--", "true"])
    capsys.readouterr()
    assert main(["list", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows[0]["id"] == "j1" and rows[0]["state"] == "pending"

def test_list_filters_by_state(root, capsys):
    main(["submit", "--project", "p", "--commit", "abc", "--branch", "main",
          "--id", "j1", "--", "true"])
    capsys.readouterr()
    main(["list", "--state", "done", "--json"])
    assert json.loads(capsys.readouterr().out) == []

def test_show_prints_spec(root, capsys):
    main(["submit", "--project", "p", "--commit", "abc", "--branch", "main",
          "--id", "j1", "--", "true"])
    capsys.readouterr()
    assert main(["show", "j1"]) == 0
    assert json.loads(capsys.readouterr().out)["id"] == "j1"

def test_show_missing_job_exits_1(root, capsys):
    assert main(["show", "nope"]) == 1

def test_cancel_pending(root, capsys):
    main(["submit", "--project", "p", "--commit", "abc", "--branch", "main",
          "--id", "j1", "--", "true"])
    assert main(["cancel", "j1"]) == 0
    assert (root / "failed" / "j1.json").exists()

def test_cancel_unknown_exits_1(root):
    assert main(["cancel", "nope"]) == 1

def test_generate_id_is_unique_and_safe():
    a, b = generate_id("job"), generate_id("job")
    assert a != b
    assert "/" not in a and a.startswith("job-")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_cli_gpuq.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gpuqueue.cli_gpuq'`

- [ ] **Step 3: Write the minimal implementation**

```python
# src/gpuqueue/cli_gpuq.py
"""gpuq: submit, list, inspect and cancel jobs."""
from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
from datetime import datetime, timezone
from pathlib import Path

from .queue import QueueRoot, STATES
from .spec import JobSpec, SpecError

DEFAULT_QUEUE_ROOT = "/workspace/queue"


def generate_id(prefix: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{stamp}-{secrets.token_hex(3)}"


def _queue(args) -> QueueRoot:
    root = args.queue_root or os.environ.get("QUEUE_ROOT") or DEFAULT_QUEUE_ROOT
    q = QueueRoot(Path(root))
    q.ensure_dirs()
    return q


def _cmd_submit(args) -> int:
    q = _queue(args)
    spec = JobSpec(
        id=args.id or generate_id(args.project),
        lane=args.lane,
        project=args.project,
        commit=args.commit,
        branch=args.branch,
        cmd=args.cmd,
        artifacts=args.artifact,
        timeout_s=args.timeout_s,
        dedupe_key=args.dedupe_key,
    )
    try:
        job_id = q.submit(spec)
    except SpecError as e:
        print(f"gpuq: {e}", file=sys.stderr)
        return 2
    except FileExistsError as e:
        print(f"gpuq: {e}", file=sys.stderr)
        return 2
    print(job_id)
    return 0


def _cmd_list(args) -> int:
    q = _queue(args)
    states = [args.state] if args.state else list(STATES)
    rows = [{"state": s, **spec.to_dict()}
            for s in states for spec in q.list_state(s)]
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        for r in rows:
            print(f"{r['state']:<8} {r['lane']:<3} {r['id']:<40} {r['project']}")
    return 0


def _cmd_show(args) -> int:
    q = _queue(args)
    found = q.find(args.id)
    if not found:
        print(f"gpuq: no such job: {args.id}", file=sys.stderr)
        return 1
    state, spec = found
    out, err = q.log_paths(args.id)
    print(json.dumps({"state": state, **spec.to_dict(),
                      "stdout_log": str(out), "stderr_log": str(err)}, indent=2))
    return 0


def _cmd_cancel(args) -> int:
    q = _queue(args)
    if q.cancel(args.id):
        print(f"cancelled {args.id}")
        return 0
    print(f"gpuq: cannot cancel {args.id} (not pending)", file=sys.stderr)
    return 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gpuq")
    p.add_argument("--queue-root", default=None)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("submit", help="queue a job")
    s.add_argument("--project", required=True)
    s.add_argument("--commit", required=True,
                   help="exact commit; a branch alone is not reproducible")
    s.add_argument("--branch", required=True)
    s.add_argument("--lane", default="cpu", choices=["cpu", "gpu"])
    s.add_argument("--id", default=None)
    s.add_argument("--artifact", action="append", default=[])
    s.add_argument("--timeout-s", dest="timeout_s", type=int, default=3600)
    s.add_argument("--dedupe-key", dest="dedupe_key", default=None)
    s.add_argument("cmd", nargs=argparse.REMAINDER)
    s.set_defaults(func=_cmd_submit)

    l = sub.add_parser("list")
    l.add_argument("--state", choices=list(STATES), default=None)
    l.add_argument("--json", action="store_true")
    l.set_defaults(func=_cmd_list)

    sh = sub.add_parser("show")
    sh.add_argument("id")
    sh.set_defaults(func=_cmd_show)

    c = sub.add_parser("cancel")
    c.add_argument("id")
    c.set_defaults(func=_cmd_cancel)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "cmd", None) is not None:
        if args.cmd and args.cmd[0] == "--":
            args.cmd = args.cmd[1:]
        if not args.cmd:
            print("gpuq: a command is required after --", file=sys.stderr)
            return 2
    return args.func(args)
```

Note: `--lane tpu` is rejected by `argparse` with `SystemExit(2)`, not by `SpecError`. Adjust `test_submit_invalid_lane_exits_nonzero` to `pytest.raises(SystemExit)` and assert `.code == 2`, or drop `choices=` and let `validate()` raise. **Choose dropping `choices=`** so the error text comes from one place — `spec.validate()` — and matches the test's `"lane" in stderr` assertion.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_cli_gpuq.py -v`
Expected: 13 passed

- [ ] **Step 5: Commit**

```bash
git add src/gpuqueue/cli_gpuq.py tests/test_cli_gpuq.py
git commit -m "feat: gpuq submit/list/show/cancel"
```

---

### Task 4: GPU identity and normalization

**Files:**
- Create: `src/gpuqueue/gpuid.py`
- Test: `tests/test_gpuid.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `normalize_gpu_uuid(raw: str) -> str`, `gpu_uuid_from_nvidia_smi(index: int = 0) -> str | None`, `gpu_name_index_fallback(index: int = 0) -> str | None`, `gpu_key(index: int = 0) -> str` (uuid, then `name-index` fallback; raises `GpuIdError` if neither works), `lock_filename(key: str) -> str`. Exception `GpuIdError(RuntimeError)`.

No `gpu_uuid_from_torch`. The design names torch as the key derivation, and that remains true of the *protocol* — see design gap 2 — but this implementation reads `nvidia-smi` only. `normalize_gpu_uuid` is what makes the two agree.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gpuid.py
import pytest
from gpuqueue import gpuid
from gpuqueue.gpuid import normalize_gpu_uuid, gpu_key, lock_filename, GpuIdError

HEX = "4b8f2c1a-0000-0000-0000-000000000001"

def test_smi_and_torch_spellings_collapse_to_one_key():
    assert normalize_gpu_uuid(f"GPU-{HEX}") == normalize_gpu_uuid(HEX)

def test_normalize_is_lowercase():
    assert normalize_gpu_uuid(f"GPU-{HEX.upper()}") == HEX

def test_normalize_strips_mig_prefix_and_whitespace():
    assert normalize_gpu_uuid(f"  MIG-{HEX}\n") == HEX

def test_normalize_rejects_empty():
    with pytest.raises(GpuIdError):
        normalize_gpu_uuid("   ")

def test_gpu_key_normalizes_the_smi_spelling(monkeypatch):
    monkeypatch.setattr(gpuid, "gpu_uuid_from_nvidia_smi", lambda i=0: f"GPU-{HEX}")
    assert gpu_key() == HEX

def test_gpu_key_falls_back_to_name_index(monkeypatch):
    monkeypatch.setattr(gpuid, "gpu_uuid_from_nvidia_smi", lambda i=0: None)
    monkeypatch.setattr(gpuid, "gpu_name_index_fallback",
                        lambda i=0: "NVIDIA GeForce RTX 4060-0")
    assert gpu_key() == "nvidia geforce rtx 4060-0"

def test_gpu_key_raises_when_no_source_works(monkeypatch):
    monkeypatch.setattr(gpuid, "gpu_uuid_from_nvidia_smi", lambda i=0: None)
    monkeypatch.setattr(gpuid, "gpu_name_index_fallback", lambda i=0: None)
    with pytest.raises(GpuIdError):
        gpu_key()

def test_module_never_imports_torch():
    """A JAX project must be able to use this lock, and the runner daemon
    must not initialize a CUDA context just to read an identifier."""
    import inspect
    assert "import torch" not in inspect.getsource(gpuid)

def test_lock_filename_is_a_safe_basename():
    name = lock_filename(HEX)
    assert "/" not in name and name.endswith(".lock")

def test_lock_filename_sanitizes_fallback_keys():
    assert "/" not in lock_filename("NVIDIA GeForce RTX 4060/weird-0")

def test_nvidia_smi_parses_first_line(monkeypatch):
    monkeypatch.setattr(gpuid, "_run",
                        lambda argv: f"GPU-{HEX}\nGPU-other\n")
    assert gpuid.gpu_uuid_from_nvidia_smi(0) == f"GPU-{HEX}"

def test_nvidia_smi_returns_none_when_missing(monkeypatch):
    def boom(argv):
        raise FileNotFoundError("nvidia-smi")
    monkeypatch.setattr(gpuid, "_run", boom)
    assert gpuid.gpu_uuid_from_nvidia_smi(0) is None
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_gpuid.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gpuqueue.gpuid'`

- [ ] **Step 3: Write the minimal implementation**

```python
# src/gpuqueue/gpuid.py
"""GPU identity.

Keying the lock on the UUID rather than the index is load-bearing: two
processes with different CUDA_VISIBLE_DEVICES mappings both see their card
as index 0, so an index-keyed lock hands them different locks for the same
physical GPU.

Normalization is load-bearing for the same reason one level down. We read
nvidia-smi, which reports "GPU-<hex>"; a torch-based implementation of the
same protocol reports the bare hex. Unnormalized, those are two lock files
for one card.

torch is deliberately never imported here. A JAX project must be able to
use this lock, and importing torch in the runner daemon costs seconds and
initializes a CUDA context in the one process that should never touch the
card.
"""
from __future__ import annotations

import re
import subprocess

_PREFIX = re.compile(r"^(GPU|MIG)-", re.IGNORECASE)
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


class GpuIdError(RuntimeError):
    """No usable GPU identity could be derived."""


def normalize_gpu_uuid(raw: str) -> str:
    s = (raw or "").strip()
    s = _PREFIX.sub("", s)
    s = s.strip().lower()
    if not s:
        raise GpuIdError(f"empty GPU uuid: {raw!r}")
    return s


def _run(argv: list[str]) -> str:
    return subprocess.run(argv, check=True, capture_output=True,
                          text=True, timeout=15).stdout


def gpu_uuid_from_nvidia_smi(index: int = 0) -> str | None:
    try:
        out = _run(["nvidia-smi", "--query-gpu=uuid",
                    "--format=csv,noheader"])
    except Exception:
        return None
    lines = [l.strip() for l in out.splitlines() if l.strip()]
    return lines[index] if index < len(lines) else None


def gpu_name_index_fallback(index: int = 0) -> str | None:
    """Design-specified fallback for drivers that report no UUID.

    Weaker than a UUID — it cannot survive a CUDA_VISIBLE_DEVICES remap —
    but a degraded key shared by everyone on the box still serializes the
    card, and refusing to run is worse.
    """
    try:
        out = _run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    except Exception:
        return None
    lines = [l.strip() for l in out.splitlines() if l.strip()]
    return f"{lines[index]}-{index}" if index < len(lines) else None


def gpu_key(index: int = 0) -> str:
    for source in (gpu_uuid_from_nvidia_smi, gpu_name_index_fallback):
        raw = source(index)
        if raw:
            return normalize_gpu_uuid(raw)
    raise GpuIdError(
        "cannot derive a GPU key: nvidia-smi reports neither a uuid nor a "
        "device name. Is a GPU visible in this container?"
    )


def lock_filename(key: str) -> str:
    return _UNSAFE.sub("-", key) + ".lock"
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_gpuid.py -v`
Expected: 12 passed

- [ ] **Step 5: Commit**

```bash
git add src/gpuqueue/gpuid.py tests/test_gpuid.py
git commit -m "feat: GPU identity derivation with cross-source uuid normalization"
```

---

### Task 5: Claim file and flock

**Files:**
- Create: `src/gpuqueue/claim.py`
- Test: `tests/test_claim.py`

**Interfaces:**
- Consumes: `gpu_key`, `lock_filename` from `gpuqueue.gpuid`; `utcnow_iso` from `gpuqueue.spec`.
- Produces: `claim_dir() -> Path` (`$GPU_CLAIM_DIR`, default `/var/lock/gpu`), `read_claim(path: Path) -> dict | None`, `pid_alive(pid: int) -> bool`, `list_claims(directory: Path | None = None) -> list[tuple[Path, dict]]`, `release_stale(directory=None) -> list[dict]`, and context manager `gpu_claim(key: str | None = None, owner: str | None = None, cmd: list[str] | None = None, wait: bool = False, directory: Path | None = None)` yielding the claim dict. Exception `ClaimBusy(RuntimeError)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_claim.py
import json
import os
import subprocess
import sys
import pytest
from gpuqueue.claim import (gpu_claim, ClaimBusy, read_claim, pid_alive,
                            list_claims, release_stale)

KEY = "4b8f2c1a-0000-0000-0000-000000000001"

def test_claim_writes_claim_file_with_pid_and_cmd(tmp_path):
    with gpu_claim(key=KEY, owner="me", cmd=["python", "t.py"],
                   directory=tmp_path) as c:
        assert c["pid"] == os.getpid()
        (path, body), = list_claims(tmp_path)
        assert body["owner"] == "me"
        assert body["cmd"] == ["python", "t.py"]
        assert body["started_at"].endswith("Z")

def test_claim_file_removed_on_exit(tmp_path):
    with gpu_claim(key=KEY, directory=tmp_path):
        pass
    assert list_claims(tmp_path) == []

def test_claim_file_removed_on_exception(tmp_path):
    with pytest.raises(ValueError):
        with gpu_claim(key=KEY, directory=tmp_path):
            raise ValueError("boom")
    assert list_claims(tmp_path) == []

def test_second_claim_in_another_process_is_busy(tmp_path):
    """flock is per-open-file-description; a real second process is the
    only honest test of exclusion."""
    holder = subprocess.Popen(
        [sys.executable, "-c",
         "import time,sys;from gpuqueue.claim import gpu_claim;"
         f"ctx=gpu_claim(key={KEY!r},directory={str(tmp_path)!r});ctx.__enter__();"
         "print('held',flush=True);time.sleep(30)"],
        stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "held"
        with pytest.raises(ClaimBusy):
            with gpu_claim(key=KEY, directory=tmp_path):
                pass
    finally:
        holder.kill()
        holder.wait()

def test_different_keys_do_not_collide(tmp_path):
    with gpu_claim(key=KEY, directory=tmp_path):
        with gpu_claim(key="other-uuid", directory=tmp_path):
            assert len(list_claims(tmp_path)) == 2

def test_pid_alive_true_for_self():
    assert pid_alive(os.getpid()) is True

def test_pid_alive_false_for_impossible_pid():
    assert pid_alive(4000000) is False

def test_release_stale_removes_dead_pid_claims(tmp_path):
    stale = tmp_path / f"{KEY}.lock.json"
    stale.write_text(json.dumps(
        {"pid": 4000000, "owner": "ghost", "cmd": ["x"],
         "started_at": "2026-08-05T00:00:00Z", "key": KEY}))
    released = release_stale(tmp_path)
    assert [r["owner"] for r in released] == ["ghost"]
    assert not stale.exists()

def test_release_stale_keeps_live_claims(tmp_path):
    with gpu_claim(key=KEY, directory=tmp_path):
        assert release_stale(tmp_path) == []
        assert len(list_claims(tmp_path)) == 1

def test_read_claim_returns_none_on_garbage(tmp_path):
    p = tmp_path / "bad.lock.json"
    p.write_text("{not json")
    assert read_claim(p) is None
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_claim.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gpuqueue.claim'`

- [ ] **Step 3: Write the minimal implementation**

```python
# src/gpuqueue/claim.py
"""The advisory lock and its claim file.

Enforcement is advisory because flock cannot be otherwise between
unprivileged processes. The claim file exists so that a human or agent
looking at a busy card learns *who* holds it without a running service.
"""
from __future__ import annotations

import errno
import fcntl
import json
import os
import getpass
from contextlib import contextmanager
from pathlib import Path

from .gpuid import gpu_key, lock_filename
from .spec import utcnow_iso

DEFAULT_CLAIM_DIR = "/var/lock/gpu"


class ClaimBusy(RuntimeError):
    """Another process holds the card."""


def claim_dir() -> Path:
    return Path(os.environ.get("GPU_CLAIM_DIR", DEFAULT_CLAIM_DIR))


def _paths(key: str, directory: Path) -> tuple[Path, Path]:
    lock = directory / lock_filename(key)
    return lock, lock.with_suffix(lock.suffix + ".json")


def read_claim(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError as e:
        return e.errno == errno.EPERM  # exists but not ours
    return True


def list_claims(directory: Path | None = None) -> list[tuple[Path, dict]]:
    d = Path(directory) if directory else claim_dir()
    if not d.is_dir():
        return []
    out = []
    for p in sorted(d.glob("*.lock.json")):
        body = read_claim(p)
        if body:
            out.append((p, body))
    return out


def release_stale(directory: Path | None = None) -> list[dict]:
    """Remove claim files whose owning pid is gone. Returns what it freed."""
    released = []
    for path, body in list_claims(directory):
        if not pid_alive(int(body.get("pid", -1))):
            released.append(body)
            path.unlink(missing_ok=True)
    return released


@contextmanager
def gpu_claim(key: str | None = None, owner: str | None = None,
              cmd: list[str] | None = None, wait: bool = False,
              directory: Path | None = None):
    d = Path(directory) if directory else claim_dir()
    d.mkdir(parents=True, exist_ok=True)
    key = key or gpu_key()
    lock_path, claim_path = _paths(key, d)

    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o666)
    flags = fcntl.LOCK_EX if wait else fcntl.LOCK_EX | fcntl.LOCK_NB
    try:
        fcntl.flock(fd, flags)
    except OSError:
        os.close(fd)
        held = read_claim(claim_path) or {}
        raise ClaimBusy(
            f"GPU {key} is held by pid {held.get('pid', '?')} "
            f"({held.get('owner', '?')}): {' '.join(held.get('cmd') or []) or '?'}"
        )

    body = {
        "pid": os.getpid(),
        "owner": owner or _default_owner(),
        "cmd": cmd or [],
        "started_at": utcnow_iso(),
        "key": key,
    }
    claim_path.write_text(json.dumps(body, indent=2) + "\n")
    try:
        yield body
    finally:
        claim_path.unlink(missing_ok=True)
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _default_owner() -> str:
    try:
        return getpass.getuser()
    except Exception:
        return f"uid{os.getuid()}"
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_claim.py -v`
Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add src/gpuqueue/claim.py tests/test_claim.py
git commit -m "feat: flock-based GPU claim with inspectable claim files"
```

---

### Task 6: Preflight — foreign CUDA process detection

**Files:**
- Create: `src/gpuqueue/preflight.py`
- Test: `tests/test_preflight.py`

**Interfaces:**
- Consumes: `list_claims`, `pid_alive` from `gpuqueue.claim`.
- Produces: `compute_apps() -> list[dict] | None` (each `{"pid": int, "used_mb": int | None, "name": str}`; `None` means *cannot see*, distinct from `[]` meaning *none*), `own_pids() -> set[int]` (this process, its ancestors and children, plus pids named by live claim files), `foreign_processes(allow: set[int] | None = None) -> list[dict]`, `preflight(allow=None) -> None` raising `PreflightFailed` when foreign processes exist. Exception `PreflightFailed(RuntimeError)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_preflight.py
import os
import pytest
from gpuqueue import preflight as pf
from gpuqueue.preflight import PreflightFailed, foreign_processes, compute_apps

SMI_ROWS = "1234, 512 MiB, python\n5678, [N/A], jupyter\n"

def test_compute_apps_parses_rows(monkeypatch):
    monkeypatch.setattr(pf, "_run", lambda argv: SMI_ROWS)
    apps = compute_apps()
    assert apps[0] == {"pid": 1234, "used_mb": 512, "name": "python"}
    assert apps[1]["used_mb"] is None

def test_compute_apps_empty_output_means_none_running(monkeypatch):
    monkeypatch.setattr(pf, "_run", lambda argv: "\n")
    assert compute_apps() == []

def test_compute_apps_not_supported_means_cannot_see(monkeypatch):
    monkeypatch.setattr(pf, "_run", lambda argv: "[Not Supported]\n")
    assert compute_apps() is None

def test_compute_apps_missing_smi_means_cannot_see(monkeypatch):
    def boom(argv):
        raise FileNotFoundError()
    monkeypatch.setattr(pf, "_run", boom)
    assert compute_apps() is None

def test_foreign_excludes_allowed_pids(monkeypatch):
    monkeypatch.setattr(pf, "compute_apps",
                        lambda: [{"pid": 1234, "used_mb": 1, "name": "python"}])
    monkeypatch.setattr(pf, "own_pids", lambda: set())
    assert foreign_processes(allow={1234}) == []

def test_foreign_excludes_own_pids(monkeypatch):
    monkeypatch.setattr(pf, "compute_apps",
                        lambda: [{"pid": os.getpid(), "used_mb": 1, "name": "py"}])
    assert foreign_processes() == []

def test_foreign_reports_stranger(monkeypatch):
    monkeypatch.setattr(pf, "compute_apps",
                        lambda: [{"pid": 4321, "used_mb": 900, "name": "train.py"}])
    monkeypatch.setattr(pf, "own_pids", lambda: set())
    assert [p["pid"] for p in foreign_processes()] == [4321]

def test_preflight_raises_naming_pid_and_command(monkeypatch):
    monkeypatch.setattr(pf, "compute_apps",
                        lambda: [{"pid": 4321, "used_mb": 900, "name": "train.py"}])
    monkeypatch.setattr(pf, "own_pids", lambda: set())
    with pytest.raises(PreflightFailed) as e:
        pf.preflight()
    assert "4321" in str(e.value) and "train.py" in str(e.value)

def test_preflight_queries_the_card_once(monkeypatch):
    """Two queries let the visibility check and the contention check
    disagree about what they saw."""
    calls = []
    def counted():
        calls.append(1)
        return []
    monkeypatch.setattr(pf, "compute_apps", counted)
    pf.preflight()
    assert len(calls) == 1

def test_preflight_passes_when_cannot_see(monkeypatch, capsys):
    """Unprivileged containers often cannot enumerate compute apps. Warn,
    do not block — refusing to run on every box that hides the list makes
    the tool useless exactly where it is needed."""
    monkeypatch.setattr(pf, "compute_apps", lambda: None)
    pf.preflight()
    assert "cannot enumerate" in capsys.readouterr().err

def test_preflight_passes_when_clear(monkeypatch):
    monkeypatch.setattr(pf, "compute_apps", lambda: [])
    pf.preflight()
```

`preflight` must call `compute_apps()` **once** and filter the result itself, via a shared `_foreign(apps, allow)` helper that `foreign_processes` also uses. Delegating to `foreign_processes` means two `nvidia-smi` invocations per preflight, and lets the "can we see the list?" check and the "who is on the card?" check disagree about what they saw.

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_preflight.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gpuqueue.preflight'`

- [ ] **Step 3: Write the minimal implementation**

```python
# src/gpuqueue/preflight.py
"""Refuse to start when someone else already holds the card.

This cannot stop a determined direct run — the lock is advisory. It
converts accidental contention into a fast, readable failure instead of a
CUDA OOM half an hour into a training run.
"""
from __future__ import annotations

import os
import subprocess
import sys

from .claim import list_claims, pid_alive


class PreflightFailed(RuntimeError):
    """Foreign CUDA processes hold the card."""


def _run(argv: list[str]) -> str:
    return subprocess.run(argv, check=True, capture_output=True,
                          text=True, timeout=15).stdout


def compute_apps() -> list[dict] | None:
    """None means we cannot see the process list, which is not the same
    as seeing that it is empty."""
    try:
        out = _run(["nvidia-smi",
                    "--query-compute-apps=pid,used_memory,process_name",
                    "--format=csv,noheader"])
    except Exception:
        return None
    if "not supported" in out.lower():
        return None
    apps = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3 or not parts[0].isdigit():
            continue
        mem = parts[1].replace("MiB", "").strip()
        apps.append({
            "pid": int(parts[0]),
            "used_mb": int(mem) if mem.isdigit() else None,
            "name": parts[2],
        })
    return apps


def own_pids() -> set[int]:
    pids = {os.getpid(), os.getppid()}
    for _, body in list_claims():
        pid = int(body.get("pid", -1))
        if pid > 0 and pid_alive(pid):
            pids.add(pid)
    pids.update(_descendants(os.getpid()))
    return pids


def _descendants(pid: int) -> set[int]:
    try:
        out = _run(["ps", "-o", "pid=", "--ppid", str(pid)])
    except Exception:
        return set()
    kids = {int(l) for l in out.split() if l.strip().isdigit()}
    for k in list(kids):
        kids |= _descendants(k)
    return kids


def _foreign(apps: list[dict], allow: set[int] | None) -> list[dict]:
    exempt = set(allow or set()) | own_pids()
    return [a for a in apps if a["pid"] not in exempt]


def foreign_processes(allow: set[int] | None = None) -> list[dict]:
    apps = compute_apps()
    if apps is None:
        return []
    return _foreign(apps, allow)


def preflight(allow: set[int] | None = None) -> None:
    # One query, not two: asking twice costs a second nvidia-smi and lets
    # the "can we see the list?" check and the "who is on the card?" check
    # disagree about what they saw.
    apps = compute_apps()
    if apps is None:
        print("gpu-claim: warning: cannot enumerate CUDA processes on this "
              "box; proceeding on the advisory lock alone", file=sys.stderr)
        return
    foreign = _foreign(apps, allow)
    if foreign:
        lines = [f"  pid {a['pid']:>7}  {a['used_mb'] or '?'} MiB  {a['name']}"
                 for a in foreign]
        raise PreflightFailed(
            "foreign CUDA processes hold this GPU:\n" + "\n".join(lines))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_preflight.py -v`
Expected: 11 passed

- [ ] **Step 5: Commit**

```bash
git add src/gpuqueue/preflight.py tests/test_preflight.py
git commit -m "feat: preflight detection of foreign CUDA processes"
```

---

### Task 7: `gpu-claim` CLI

**Files:**
- Create: `src/gpuqueue/cli_claim.py`
- Test: `tests/test_cli_claim.py`

**Interfaces:**
- Consumes: `gpu_claim`, `ClaimBusy`, `release_stale`, `list_claims`; `preflight`, `PreflightFailed`; `gpu_key`, `GpuIdError`.
- Produces: `main(argv=None) -> int`. Usage `gpu-claim [--wait] [--no-preflight] [--owner NAME] [--gpu-index N] -- CMD...`; plus `gpu-claim --status` and `gpu-claim --reap`. Exit codes: child's exit code on success, `75` (EX_TEMPFAIL) when the card is busy, `69` (EX_UNAVAILABLE) on preflight failure or no GPU.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli_claim.py
import json
import pytest
from gpuqueue import cli_claim
from gpuqueue.claim import ClaimBusy
from gpuqueue.preflight import PreflightFailed
from gpuqueue.gpuid import GpuIdError

KEY = "4b8f2c1a-0000-0000-0000-000000000001"

@pytest.fixture(autouse=True)
def fake_gpu(tmp_path, monkeypatch):
    monkeypatch.setenv("GPU_CLAIM_DIR", str(tmp_path))
    monkeypatch.setattr(cli_claim, "gpu_key", lambda index=0: KEY)
    monkeypatch.setattr(cli_claim, "preflight", lambda allow=None: None)

def test_runs_command_and_returns_its_exit_code():
    assert cli_claim.main(["--", "sh", "-c", "exit 0"]) == 0
    assert cli_claim.main(["--", "sh", "-c", "exit 3"]) == 3

def test_claim_released_after_command(tmp_path):
    cli_claim.main(["--", "true"])
    assert list(tmp_path.glob("*.lock.json")) == []

def test_busy_exits_75(monkeypatch, capsys):
    def busy(**kw):
        raise ClaimBusy("held by pid 999")
    monkeypatch.setattr(cli_claim, "gpu_claim", busy)
    assert cli_claim.main(["--", "true"]) == 75
    assert "999" in capsys.readouterr().err

def test_preflight_failure_exits_69(monkeypatch, capsys):
    def fail(allow=None):
        raise PreflightFailed("pid 4321 train.py")
    monkeypatch.setattr(cli_claim, "preflight", fail)
    assert cli_claim.main(["--", "true"]) == 69
    assert "4321" in capsys.readouterr().err

def test_no_preflight_flag_skips_it(monkeypatch):
    def fail(allow=None):
        raise PreflightFailed("should not be called")
    monkeypatch.setattr(cli_claim, "preflight", fail)
    assert cli_claim.main(["--no-preflight", "--", "true"]) == 0

def test_no_gpu_exits_69(monkeypatch, capsys):
    def boom(index=0):
        raise GpuIdError("no CUDA device")
    monkeypatch.setattr(cli_claim, "gpu_key", boom)
    assert cli_claim.main(["--", "true"]) == 69
    assert "no CUDA device" in capsys.readouterr().err

def test_status_prints_claims_as_json(tmp_path, capsys):
    (tmp_path / "x.lock.json").write_text(json.dumps(
        {"pid": 1, "owner": "me", "cmd": ["t"], "started_at": "2026-08-05T00:00:00Z"}))
    assert cli_claim.main(["--status"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["owner"] == "me"

def test_reap_removes_dead_claims(tmp_path, capsys):
    (tmp_path / "x.lock.json").write_text(json.dumps(
        {"pid": 4000000, "owner": "ghost", "cmd": ["t"],
         "started_at": "2026-08-05T00:00:00Z"}))
    assert cli_claim.main(["--reap"]) == 0
    assert not (tmp_path / "x.lock.json").exists()

def test_missing_command_exits_2(capsys):
    assert cli_claim.main([]) == 2
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_cli_claim.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gpuqueue.cli_claim'`

- [ ] **Step 3: Write the minimal implementation**

```python
# src/gpuqueue/cli_claim.py
"""gpu-claim: hold the advisory lock for the duration of a command."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys

from .claim import gpu_claim, ClaimBusy, release_stale, list_claims
from .gpuid import gpu_key, GpuIdError
from .preflight import preflight, PreflightFailed

EX_UNAVAILABLE = 69
EX_TEMPFAIL = 75


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gpu-claim",
        description="Run a command holding the advisory GPU lock.")
    p.add_argument("--wait", action="store_true",
                   help="block until the card is free instead of failing")
    p.add_argument("--no-preflight", action="store_true")
    p.add_argument("--owner", default=None)
    p.add_argument("--gpu-index", type=int, default=0)
    p.add_argument("--status", action="store_true", help="print live claims")
    p.add_argument("--reap", action="store_true", help="release dead claims")
    p.add_argument("cmd", nargs=argparse.REMAINDER)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.status:
        print(json.dumps([body for _, body in list_claims()], indent=2))
        return 0
    if args.reap:
        for body in release_stale():
            print(f"released stale claim: pid {body.get('pid')} "
                  f"{body.get('owner')}", file=sys.stderr)
        return 0

    cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd
    if not cmd:
        print("gpu-claim: a command is required after --", file=sys.stderr)
        return 2

    try:
        key = gpu_key(args.gpu_index)
    except GpuIdError as e:
        print(f"gpu-claim: {e}", file=sys.stderr)
        return EX_UNAVAILABLE

    if not args.no_preflight:
        try:
            preflight()
        except PreflightFailed as e:
            print(f"gpu-claim: {e}", file=sys.stderr)
            return EX_UNAVAILABLE

    try:
        with gpu_claim(key=key, owner=args.owner, cmd=cmd, wait=args.wait):
            return subprocess.run(cmd).returncode
    except ClaimBusy as e:
        print(f"gpu-claim: {e}", file=sys.stderr)
        return EX_TEMPFAIL
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_cli_claim.py -v`
Expected: 9 passed

- [ ] **Step 5: Manually verify on a box with a GPU**

Run: `gpu-claim -- sh -c 'nvidia-smi -L; sleep 30'`, and in a second shell while it runs: `gpu-claim --status` and `gpu-claim -- true` (expect exit 75).
Expected: first prints `True`; `--status` shows one claim; second returns 75 naming the holder's pid.

- [ ] **Step 6: Commit**

```bash
git add src/gpuqueue/cli_claim.py tests/test_cli_claim.py
git commit -m "feat: gpu-claim CLI with preflight, status and reap"
```

---

### Task 8: Configuration

**Files:**
- Create: `src/gpuqueue/config.py`, `gpuq.example.toml`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: dataclasses `ProjectConfig(name, remote, checkout, venv, commit_artifacts, push)` and `RunnerConfig(queue_root: Path, cpu_slots: int, poll_interval_s: float, claim_dir: Path | None, kill_orphan_cuda: bool, projects: dict[str, ProjectConfig])`. Functions `load_config(path: Path) -> RunnerConfig`, `default_config_path() -> Path` (`$GPUQ_CONFIG`, else `/workspace/gpuq.toml`). Exception `ConfigError(ValueError)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
import pytest
from gpuqueue.config import load_config, ConfigError

TOML = """
[queue]
root = "/workspace/queue"
cpu_slots = 4

[project.wgan-synthetic]
remote   = "git@github.com:Daniel-T-S-Adams/wgan-synthetic.git"
checkout = "/workspace/checkouts/wgan-synthetic"
venv     = "/workspace/checkouts/wgan-synthetic/.venv"
commit_artifacts = true
"""

def _write(tmp_path, text):
    p = tmp_path / "gpuq.toml"
    p.write_text(text)
    return p

def test_loads_queue_settings(tmp_path):
    cfg = load_config(_write(tmp_path, TOML))
    assert str(cfg.queue_root) == "/workspace/queue"
    assert cfg.cpu_slots == 4

def test_loads_project(tmp_path):
    proj = load_config(_write(tmp_path, TOML)).projects["wgan-synthetic"]
    assert proj.name == "wgan-synthetic"
    assert proj.commit_artifacts is True
    assert str(proj.venv).endswith(".venv")

def test_defaults_applied(tmp_path):
    cfg = load_config(_write(tmp_path, '[queue]\nroot = "/q"\n'))
    assert cfg.cpu_slots == 4
    assert cfg.poll_interval_s == 2.0
    assert cfg.kill_orphan_cuda is True
    assert cfg.projects == {}

def test_missing_root_rejected(tmp_path):
    with pytest.raises(ConfigError, match="root"):
        load_config(_write(tmp_path, "[queue]\n"))

def test_project_without_checkout_rejected(tmp_path):
    bad = '[queue]\nroot="/q"\n[project.p]\nremote="git@x:y.git"\n'
    with pytest.raises(ConfigError, match="checkout"):
        load_config(_write(tmp_path, bad))

def test_zero_cpu_slots_rejected(tmp_path):
    with pytest.raises(ConfigError, match="cpu_slots"):
        load_config(_write(tmp_path, '[queue]\nroot="/q"\ncpu_slots=0\n'))

def test_missing_file_raises_config_error(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.toml")

def test_example_config_in_repo_is_loadable():
    from pathlib import Path
    load_config(Path(__file__).resolve().parents[1] / "gpuq.example.toml")
```

The last test is deliberate: a shipped example that no longer parses is a documented lie, and it is cheap to keep honest.

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gpuqueue.config'`

- [ ] **Step 3: Write the minimal implementation**

```python
# src/gpuqueue/config.py
"""Runner configuration. Every project the runner serves is declared here."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import tomllib  # stdlib from 3.11, which is why 3.11 is the floor


class ConfigError(ValueError):
    """The configuration file is missing or malformed."""


@dataclass
class ProjectConfig:
    name: str
    remote: str
    checkout: Path
    venv: Path | None = None
    commit_artifacts: bool = False
    push: bool = False


@dataclass
class RunnerConfig:
    queue_root: Path
    cpu_slots: int = 4
    poll_interval_s: float = 2.0
    claim_dir: Path | None = None
    kill_orphan_cuda: bool = True
    projects: dict[str, ProjectConfig] = field(default_factory=dict)


def default_config_path() -> Path:
    return Path(os.environ.get("GPUQ_CONFIG", "/workspace/gpuq.toml"))


def load_config(path: Path) -> RunnerConfig:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config not found: {path}")
    data = tomllib.loads(path.read_text())

    queue = data.get("queue") or {}
    root = queue.get("root")
    if not root:
        raise ConfigError("[queue].root is required")
    cpu_slots = int(queue.get("cpu_slots", 4))
    if cpu_slots < 1:
        raise ConfigError("[queue].cpu_slots must be >= 1")

    projects: dict[str, ProjectConfig] = {}
    for name, p in (data.get("project") or {}).items():
        if not p.get("checkout"):
            raise ConfigError(f"[project.{name}].checkout is required")
        if not p.get("remote"):
            raise ConfigError(f"[project.{name}].remote is required")
        projects[name] = ProjectConfig(
            name=name,
            remote=p["remote"],
            checkout=Path(p["checkout"]),
            venv=Path(p["venv"]) if p.get("venv") else None,
            commit_artifacts=bool(p.get("commit_artifacts", False)),
            push=bool(p.get("push", False)),
        )

    claim_dir = queue.get("claim_dir")
    return RunnerConfig(
        queue_root=Path(root),
        cpu_slots=cpu_slots,
        poll_interval_s=float(queue.get("poll_interval_s", 2.0)),
        claim_dir=Path(claim_dir) if claim_dir else None,
        kill_orphan_cuda=bool(queue.get("kill_orphan_cuda", True)),
        projects=projects,
    )
```

```toml
# gpuq.example.toml
[queue]
root = "/workspace/queue"
# 4, not the core count: typical CPU jobs here are BLAS-bound and already
# thread internally, so one per core oversubscribes. Measure before tuning.
cpu_slots = 4
poll_interval_s = 2.0
kill_orphan_cuda = true

[project.wgan-synthetic]
remote   = "git@github.com:Daniel-T-S-Adams/wgan-synthetic.git"
checkout = "/workspace/checkouts/wgan-synthetic"
venv     = "/workspace/checkouts/wgan-synthetic/.venv"
commit_artifacts = true
push = false
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_config.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add src/gpuqueue/config.py gpuq.example.toml tests/test_config.py
git commit -m "feat: TOML runner configuration with shipped example"
```

---

### Task 9: Git operations (runner-loop only)

**Files:**
- Create: `src/gpuqueue/git_ops.py`
- Test: `tests/test_git_ops.py`

**Interfaces:**
- Consumes: `ProjectConfig`.
- Produces: `git(args: list[str], cwd: Path | None = None, check: bool = True) -> str`, `ensure_checkout(project: ProjectConfig) -> Path`, `add_worktree(checkout: Path, dest: Path, commit: str) -> Path`, `remove_worktree(checkout: Path, dest: Path) -> None`, `commit_artifacts(project, branch, files: list[Path], rel_paths: list[str], message: str) -> str | None` (returns the new commit sha, or `None` if nothing changed). Exception `GitError(RuntimeError)`.

**Every function here is called from `Runner.tick()` and nowhere else.** Concurrent jobs committing into one checkout would corrupt the index. Because the runner is single-threaded, there is exactly one caller and serialization is a property of the design rather than a rule someone has to remember.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_git_ops.py
import subprocess
import pytest
from pathlib import Path
from gpuqueue.git_ops import (git, ensure_checkout, add_worktree,
                              remove_worktree, commit_artifacts, GitError)
from gpuqueue.config import ProjectConfig

def _init_origin(tmp_path) -> Path:
    origin = tmp_path / "origin"
    origin.mkdir()
    git(["init", "-q", "-b", "main"], cwd=origin)
    git(["config", "user.email", "t@t"], cwd=origin)
    git(["config", "user.name", "t"], cwd=origin)
    (origin / "a.txt").write_text("one\n")
    git(["add", "a.txt"], cwd=origin)
    git(["commit", "-qm", "first"], cwd=origin)
    return origin

@pytest.fixture
def project(tmp_path):
    origin = _init_origin(tmp_path)
    return ProjectConfig(name="p", remote=str(origin),
                         checkout=tmp_path / "checkout", commit_artifacts=True)

def test_ensure_checkout_clones(project):
    path = ensure_checkout(project)
    assert (path / "a.txt").exists()

def test_ensure_checkout_is_idempotent(project):
    ensure_checkout(project)
    assert (ensure_checkout(project) / "a.txt").exists()

def test_add_worktree_at_pinned_commit(project, tmp_path):
    checkout = ensure_checkout(project)
    sha = git(["rev-parse", "HEAD"], cwd=checkout).strip()
    (checkout / "b.txt").write_text("two\n")
    git(["add", "b.txt"], cwd=checkout)
    git(["-c", "user.email=t@t", "-c", "user.name=t",
         "commit", "-qm", "second"], cwd=checkout)

    wt = add_worktree(checkout, tmp_path / "work" / "j1", sha)
    assert (wt / "a.txt").exists()
    assert not (wt / "b.txt").exists()   # pinned to the older commit

def test_two_worktrees_at_different_commits_coexist(project, tmp_path):
    """The reason worktrees exist here at all: concurrent CPU jobs pinned
    to different commits cannot share one working tree."""
    checkout = ensure_checkout(project)
    old = git(["rev-parse", "HEAD"], cwd=checkout).strip()
    (checkout / "b.txt").write_text("two\n")
    git(["add", "b.txt"], cwd=checkout)
    git(["-c", "user.email=t@t", "-c", "user.name=t",
         "commit", "-qm", "second"], cwd=checkout)
    new = git(["rev-parse", "HEAD"], cwd=checkout).strip()

    w1 = add_worktree(checkout, tmp_path / "w" / "j1", old)
    w2 = add_worktree(checkout, tmp_path / "w" / "j2", new)
    assert not (w1 / "b.txt").exists()
    assert (w2 / "b.txt").exists()

def test_remove_worktree_cleans_up(project, tmp_path):
    checkout = ensure_checkout(project)
    sha = git(["rev-parse", "HEAD"], cwd=checkout).strip()
    wt = add_worktree(checkout, tmp_path / "w" / "j1", sha)
    remove_worktree(checkout, wt)
    assert not wt.exists()
    assert "j1" not in git(["worktree", "list"], cwd=checkout)

def test_unknown_commit_raises_git_error(project, tmp_path):
    checkout = ensure_checkout(project)
    with pytest.raises(GitError):
        add_worktree(checkout, tmp_path / "w" / "j1", "deadbee")

def test_commit_artifacts_creates_commit(project, tmp_path):
    checkout = ensure_checkout(project)
    src = tmp_path / "summary.json"
    src.write_text('{"loss": 1}')
    sha = commit_artifacts(project, "main", [src], ["runs/v0/summary.json"],
                           "artifacts: j1")
    assert sha
    assert (checkout / "runs" / "v0" / "summary.json").exists()
    assert "artifacts: j1" in git(["log", "-1", "--pretty=%s"], cwd=checkout)

def test_commit_artifacts_noop_when_unchanged(project, tmp_path):
    checkout = ensure_checkout(project)
    src = tmp_path / "s.json"
    src.write_text("{}")
    commit_artifacts(project, "main", [src], ["runs/s.json"], "first")
    assert commit_artifacts(project, "main", [src], ["runs/s.json"],
                            "again") is None

def test_git_error_includes_stderr(tmp_path):
    with pytest.raises(GitError, match="not a git repository"):
        git(["status"], cwd=tmp_path)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_git_ops.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gpuqueue.git_ops'`

- [ ] **Step 3: Write the minimal implementation**

```python
# src/gpuqueue/git_ops.py
"""All git. Called only from Runner.tick().

Jobs write artifacts into their own worktree; the loop moves them into the
checkout and commits between polls. The runner has one thread, so there is
one caller here and repository mutation is serialized by construction.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .config import ProjectConfig

_IDENTITY = ["-c", "user.email=gpuq@localhost", "-c", "user.name=gpuq-runner"]


class GitError(RuntimeError):
    pass


def git(args: list[str], cwd: Path | None = None, check: bool = True) -> str:
    proc = subprocess.run(["git", *args], cwd=cwd, text=True,
                          capture_output=True)
    if check and proc.returncode != 0:
        raise GitError(f"git {' '.join(args)} failed ({proc.returncode}): "
                       f"{proc.stderr.strip()}")
    return proc.stdout


def ensure_checkout(project: ProjectConfig) -> Path:
    path = Path(project.checkout)
    if (path / ".git").exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    git(["clone", project.remote, str(path)])
    return path


def add_worktree(checkout: Path, dest: Path, commit: str) -> Path:
    dest = Path(dest)
    if dest.exists():
        remove_worktree(checkout, dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    git(["worktree", "add", "--detach", str(dest), commit], cwd=checkout)
    return dest


def remove_worktree(checkout: Path, dest: Path) -> None:
    git(["worktree", "remove", "--force", str(dest)], cwd=checkout, check=False)
    if Path(dest).exists():
        shutil.rmtree(dest, ignore_errors=True)
    git(["worktree", "prune"], cwd=checkout, check=False)


def commit_artifacts(project: ProjectConfig, branch: str,
                     files: list[Path], rel_paths: list[str],
                     message: str) -> str | None:
    checkout = ensure_checkout(project)
    git(["checkout", "-q", branch], cwd=checkout, check=False)
    for src, rel in zip(files, rel_paths):
        dst = checkout / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if Path(src).is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
        git(["add", "--", rel], cwd=checkout)
    if not git(["status", "--porcelain"], cwd=checkout).strip():
        return None
    git([*_IDENTITY, "commit", "-qm", message], cwd=checkout)
    sha = git(["rev-parse", "HEAD"], cwd=checkout).strip()
    if project.push:
        git(["push", "origin", f"HEAD:{branch}"], cwd=checkout, check=False)
    return sha
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_git_ops.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add src/gpuqueue/git_ops.py tests/test_git_ops.py
git commit -m "feat: git operations with per-job worktrees at pinned commits"
```

---

### Task 10: Executor — start, poll and kill one job

**Files:**
- Create: `src/gpuqueue/executor.py`
- Test: `tests/test_executor.py`

**Interfaces:**
- Consumes: `JobSpec`, `ProjectConfig`.
- Produces:
  - dataclass `RunningJob(spec, proc, out_log, err_log, out_fh, err_fh, deadline: float)` with property `pid`.
  - dataclass `JobResult(exit_code: int, timed_out: bool, oom: bool, stderr_tail: str)`.
  - `start_job(spec, workdir: Path, out_log: Path, err_log: Path, project: ProjectConfig | None = None, extra_env: dict | None = None) -> RunningJob`, raising `StartFailed` when the binary cannot be executed.
  - `poll_job(running: RunningJob) -> JobResult | None` — `None` while it is still running; kills the process group and returns a `timed_out` result once the deadline passes.
  - `kill_job(running: RunningJob) -> JobResult` — for runner shutdown.
  - Constant `STDERR_TAIL_BYTES = 4000`; helper `looks_like_oom(text: str) -> bool`; exception `StartFailed(RuntimeError)`.

The split into start/poll is what lets the runner be single-threaded. A blocking `run_job(...)` forces a thread per job; `poll_job` returning `None` lets one loop supervise every job at once. The wall-clock watchdog lives here, in `poll_job`, because that is the only place that knows the deadline and holds the handle needed to kill the group.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_executor.py
import os
import time
from pathlib import Path
import pytest
from gpuqueue.executor import (start_job, poll_job, kill_job, looks_like_oom,
                               StartFailed, STDERR_TAIL_BYTES)
from gpuqueue.spec import JobSpec

def mkspec(cmd, timeout_s=30, **over):
    d = dict(id="j1", lane="cpu", project="p", commit="abc", branch="main",
             cmd=cmd, artifacts=[], timeout_s=timeout_s)
    d.update(over)
    return JobSpec.from_dict(d)

def _logs(tmp_path):
    return tmp_path / "j1.out", tmp_path / "j1.err"

def finish(running, limit=30.0):
    """Poll to completion the way the runner's tick loop would."""
    deadline = time.monotonic() + limit
    while time.monotonic() < deadline:
        result = poll_job(running)
        if result is not None:
            return result
        time.sleep(0.02)
    raise AssertionError("job did not finish")

def test_success_returns_zero_and_captures_stdout(tmp_path):
    out, err = _logs(tmp_path)
    r = finish(start_job(mkspec(["sh", "-c", "echo hello"]), tmp_path, out, err))
    assert r.exit_code == 0 and r.timed_out is False
    assert out.read_text().strip() == "hello"

def test_poll_returns_none_while_running(tmp_path):
    out, err = _logs(tmp_path)
    running = start_job(mkspec(["sleep", "5"]), tmp_path, out, err)
    assert poll_job(running) is None
    kill_job(running)

def test_start_reports_the_pid_immediately(tmp_path):
    """The runner writes this into running/<id>.json before the next tick,
    so the reaper can tell a live job from an abandoned one."""
    out, err = _logs(tmp_path)
    running = start_job(mkspec(["sleep", "5"]), tmp_path, out, err)
    assert running.pid > 0
    kill_job(running)

def test_failure_captures_stderr_tail(tmp_path):
    out, err = _logs(tmp_path)
    r = finish(start_job(mkspec(["sh", "-c", "echo boom >&2; exit 7"]),
                         tmp_path, out, err))
    assert r.exit_code == 7
    assert "boom" in r.stderr_tail

def test_stderr_tail_is_bounded(tmp_path):
    out, err = _logs(tmp_path)
    r = finish(start_job(
        mkspec(["sh", "-c", "head -c 100000 /dev/zero | tr '\\0' 'x' >&2; exit 1"]),
        tmp_path, out, err))
    assert len(r.stderr_tail) <= STDERR_TAIL_BYTES

def test_timeout_kills_and_flags(tmp_path):
    out, err = _logs(tmp_path)
    r = finish(start_job(mkspec(["sleep", "30"], timeout_s=1), tmp_path, out, err))
    assert r.timed_out is True and r.exit_code != 0

def test_timeout_kills_the_whole_process_group(tmp_path):
    """A trainer that spawns dataloader workers must not leave them behind
    holding the card."""
    out, err = _logs(tmp_path)
    marker = tmp_path / "child.pid"
    cmd = ["sh", "-c", f"sleep 30 & echo $! > {marker}; sleep 30"]
    r = finish(start_job(mkspec(cmd, timeout_s=1), tmp_path, out, err))
    assert r.timed_out is True
    child = int(marker.read_text().strip())
    with pytest.raises(OSError):
        os.kill(child, 0)

def test_kill_job_terminates_a_live_job(tmp_path):
    """Runner shutdown must not leave a job holding the card."""
    out, err = _logs(tmp_path)
    running = start_job(mkspec(["sleep", "30"]), tmp_path, out, err)
    r = kill_job(running)
    assert r.exit_code != 0
    assert running.proc.poll() is not None

def test_runs_in_the_given_workdir(tmp_path):
    out, err = _logs(tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    (work / "here.txt").write_text("x")
    finish(start_job(mkspec(["sh", "-c", "ls"]), work, out, err))
    assert "here.txt" in out.read_text()

def test_extra_env_is_passed(tmp_path):
    out, err = _logs(tmp_path)
    finish(start_job(mkspec(["sh", "-c", "echo $GPUQ_JOB_ID"]), tmp_path, out, err,
                     extra_env={"GPUQ_JOB_ID": "j1"}))
    assert out.read_text().strip() == "j1"

def test_venv_bin_is_prepended_to_path(tmp_path):
    from gpuqueue.config import ProjectConfig
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    out, err = _logs(tmp_path)
    proj = ProjectConfig(name="p", remote="r", checkout=tmp_path, venv=venv)
    finish(start_job(mkspec(["sh", "-c", "echo $PATH"]), tmp_path, out, err,
                     project=proj))
    assert out.read_text().startswith(str(venv / "bin"))

def test_missing_executable_raises_start_failed_naming_the_binary(tmp_path):
    out, err = _logs(tmp_path)
    with pytest.raises(StartFailed, match="definitely-not-a-real-binary"):
        start_job(mkspec(["definitely-not-a-real-binary"]), tmp_path, out, err)

def test_looks_like_oom_detects_cuda_oom():
    assert looks_like_oom("RuntimeError: CUDA out of memory. Tried to allocate")
    assert looks_like_oom("torch.cuda.OutOfMemoryError: CUDA out of memory")
    assert not looks_like_oom("ValueError: bad config")

def test_oom_flag_set_from_stderr(tmp_path):
    out, err = _logs(tmp_path)
    r = finish(start_job(mkspec(["sh", "-c", "echo 'CUDA out of memory' >&2; exit 1"]),
                         tmp_path, out, err))
    assert r.oom is True
```

`looks_like_oom` matching torch's exception text is not an import of torch — it is a string the job's stderr may contain.

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_executor.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gpuqueue.executor'`

- [ ] **Step 3: Write the minimal implementation**

```python
# src/gpuqueue/executor.py
"""Start, poll and kill one job's subprocess.

Knows nothing about queues, lanes or git. Deliberately non-blocking: the
runner is single-threaded, so nothing here may wait for a job to finish.
`start_job` hands back a handle, `poll_job` answers "done yet?" and returns
None while the answer is no.
"""
from __future__ import annotations

import os
import re
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from .config import ProjectConfig
from .spec import JobSpec

STDERR_TAIL_BYTES = 4000
_OOM = re.compile(r"cuda out of memory|outofmemoryerror|cublas_status_alloc_failed",
                  re.IGNORECASE)


class StartFailed(RuntimeError):
    """The job's command could not be executed at all."""


def looks_like_oom(text: str) -> bool:
    return bool(_OOM.search(text or ""))


@dataclass
class JobResult:
    exit_code: int
    timed_out: bool
    oom: bool
    stderr_tail: str


@dataclass
class RunningJob:
    spec: JobSpec
    proc: subprocess.Popen
    out_log: Path
    err_log: Path
    out_fh: IO[bytes]
    err_fh: IO[bytes]
    deadline: float  # time.monotonic() value past which it is killed

    @property
    def pid(self) -> int:
        return self.proc.pid


def _tail(path: Path, n: int = STDERR_TAIL_BYTES) -> str:
    try:
        data = path.read_bytes()
    except OSError:
        return ""
    return data[-n:].decode("utf-8", "replace")


def _env_for(project: ProjectConfig | None, extra: dict | None) -> dict:
    env = dict(os.environ)
    if project and project.venv:
        bin_dir = str(Path(project.venv) / "bin")
        env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")
        env["VIRTUAL_ENV"] = str(project.venv)
    env.update(extra or {})
    return env


def start_job(spec: JobSpec, workdir: Path, out_log: Path, err_log: Path,
              project: ProjectConfig | None = None,
              extra_env: dict | None = None) -> RunningJob:
    out_log.parent.mkdir(parents=True, exist_ok=True)
    fo = open(out_log, "wb")
    fe = open(err_log, "wb")
    try:
        proc = subprocess.Popen(
            spec.cmd, cwd=str(workdir), stdout=fo, stderr=fe,
            env=_env_for(project, extra_env),
            start_new_session=True,  # own process group, so we can kill it all
        )
    except OSError as e:
        # Record it where a consumer looks for failures, then tell the caller.
        fe.write(f"{e}\n".encode())
        fo.close()
        fe.close()
        raise StartFailed(f"cannot execute {spec.cmd[0]!r}: {e}") from e
    return RunningJob(spec=spec, proc=proc, out_log=out_log, err_log=err_log,
                      out_fh=fo, err_fh=fe,
                      deadline=time.monotonic() + spec.timeout_s)


def poll_job(running: RunningJob) -> JobResult | None:
    """None while the job is still running.

    The wall-clock watchdog lives here because this is the one function
    called about a job on every tick.
    """
    if running.proc.poll() is None:
        if time.monotonic() < running.deadline:
            return None
        _kill_group(running.proc)
        return _result(running, timed_out=True)
    return _result(running, timed_out=False)


def kill_job(running: RunningJob) -> JobResult:
    """Stop a live job now. Used when the runner is shutting down."""
    if running.proc.poll() is None:
        _kill_group(running.proc)
    return _result(running, timed_out=False)


def _result(running: RunningJob, timed_out: bool) -> JobResult:
    for fh in (running.out_fh, running.err_fh):
        if not fh.closed:
            fh.close()  # flush before reading the tail back
    tail = _tail(running.err_log)
    code = running.proc.returncode if running.proc.returncode is not None else -1
    return JobResult(exit_code=code, timed_out=timed_out,
                     oom=looks_like_oom(tail), stderr_tail=tail)


def _kill_group(proc: subprocess.Popen) -> None:
    """SIGTERM the group, then SIGKILL what survives. A trainer's dataloader
    workers must not outlive it holding VRAM.

    This does block, for up to the two grace periods. That is deliberate: a
    job being killed is not the moment to admit another one, and the runner
    has nothing useful to do until the card is actually free.
    """
    try:
        pgid = os.getpgid(proc.pid)
    except OSError:
        return
    for sig, grace in ((signal.SIGTERM, 10.0), (signal.SIGKILL, 5.0)):
        try:
            os.killpg(pgid, sig)
        except OSError:
            return
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                return
            time.sleep(0.1)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_executor.py -v`
Expected: 14 passed

- [ ] **Step 5: Commit**

```bash
git add src/gpuqueue/executor.py tests/test_executor.py
git commit -m "feat: non-blocking job executor with process-group timeout and OOM detection"
```

---

### Task 11: Reaper

**Files:**
- Create: `src/gpuqueue/reaper.py`
- Test: `tests/test_reaper.py`

**Interfaces:**
- Consumes: `QueueRoot`, `JobSpec`, `RunnerConfig`, `release_stale`, `pid_alive`, `compute_apps`, `own_pids`.
- Produces: `MAX_ATTEMPTS = 1` and `reap(queue: QueueRoot, cfg: RunnerConfig, active_ids: set[str] | None = None) -> dict` returning `{"stale_claims": [...], "requeued": [...], "failed": [...], "killed_pids": [...], "cleaned_paths": [...]}`; helpers `requeue_orphans(queue, active_ids) -> tuple[list, list]`, `kill_orphan_cuda(protect: set[int]) -> list[int]`, `clean_partials(queue) -> list[str]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_reaper.py
import json
import pytest
from gpuqueue import reaper as rp
from gpuqueue.reaper import reap, MAX_ATTEMPTS
from gpuqueue.queue import QueueRoot
from gpuqueue.spec import JobSpec
from gpuqueue.config import RunnerConfig

def mkspec(job_id="j1", **over):
    d = dict(id=job_id, lane="gpu", project="p", commit="abc", branch="main",
             cmd=["true"], artifacts=[], timeout_s=60)
    d.update(over)
    return JobSpec.from_dict(d)

@pytest.fixture
def q(tmp_path):
    qr = QueueRoot(tmp_path / "queue")
    qr.ensure_dirs()
    return qr

@pytest.fixture
def cfg(q):
    return RunnerConfig(queue_root=q.root, kill_orphan_cuda=False)

@pytest.fixture(autouse=True)
def no_gpu_calls(monkeypatch):
    monkeypatch.setattr(rp, "release_stale", lambda directory=None: [])
    monkeypatch.setattr(rp, "compute_apps", lambda: [])
    monkeypatch.setattr(rp, "own_pids", lambda: set())

def test_requeues_running_job_with_dead_pid(q, cfg):
    q.submit(mkspec())
    spec = q.claim("j1")
    spec.pid = 4000000
    q._write(q.path_for("running", "j1"), spec)
    result = reap(q, cfg)
    assert result["requeued"] == ["j1"]
    assert json.loads((q.root / "pending" / "j1.json").read_text())["attempts"] == 1

def test_second_reap_fails_instead_of_requeueing(q, cfg):
    """Without an attempt ceiling a crash-looping job occupies the only
    card indefinitely."""
    q.submit(mkspec(attempts=MAX_ATTEMPTS))
    spec = q.claim("j1")
    spec.pid = 4000000
    q._write(q.path_for("running", "j1"), spec)
    result = reap(q, cfg)
    assert result["failed"] == ["j1"]
    body = json.loads((q.root / "failed" / "j1.json").read_text())
    assert "attempt" in (body["error"] or "").lower()

def test_leaves_running_job_with_live_pid(q, cfg):
    import os
    q.submit(mkspec())
    spec = q.claim("j1")
    spec.pid = os.getpid()
    q._write(q.path_for("running", "j1"), spec)
    assert reap(q, cfg)["requeued"] == []
    assert (q.root / "running" / "j1.json").exists()

def test_leaves_jobs_this_runner_is_actively_executing(q, cfg):
    """A job dispatched microseconds ago has no pid yet; reaping it would
    run it twice."""
    q.submit(mkspec())
    q.claim("j1")
    assert reap(q, cfg, active_ids={"j1"})["requeued"] == []

def test_requeues_running_job_with_no_pid_when_not_active(q, cfg):
    q.submit(mkspec())
    q.claim("j1")
    assert reap(q, cfg, active_ids=set())["requeued"] == ["j1"]

def test_releases_stale_claims(q, cfg, monkeypatch):
    monkeypatch.setattr(rp, "release_stale",
                        lambda directory=None: [{"pid": 999, "owner": "ghost"}])
    assert reap(q, cfg)["stale_claims"] == [{"pid": 999, "owner": "ghost"}]

def test_removes_part_files(q, cfg):
    (q.root / "work" / "j1").mkdir(parents=True)
    stray = q.root / "work" / "j1" / "out.json.part"
    stray.write_text("half")
    assert str(stray) in reap(q, cfg)["cleaned_paths"]
    assert not stray.exists()

def test_kills_orphan_cuda_when_enabled(q, monkeypatch):
    cfg = RunnerConfig(queue_root=q.root, kill_orphan_cuda=True)
    monkeypatch.setattr(rp, "compute_apps",
                        lambda: [{"pid": 4321, "used_mb": 900, "name": "t.py"}])
    killed = []
    monkeypatch.setattr(rp, "_kill", lambda pid: killed.append(pid) or True)
    assert reap(q, cfg)["killed_pids"] == [4321]
    assert killed == [4321]

def test_does_not_kill_pids_of_running_jobs(q, monkeypatch):
    cfg = RunnerConfig(queue_root=q.root, kill_orphan_cuda=True)
    q.submit(mkspec())
    spec = q.claim("j1")
    spec.pid = 4321
    q._write(q.path_for("running", "j1"), spec)
    monkeypatch.setattr(rp, "compute_apps",
                        lambda: [{"pid": 4321, "used_mb": 900, "name": "t.py"}])
    monkeypatch.setattr(rp, "_kill", lambda pid: pytest.fail("killed a live job"))
    assert reap(q, cfg)["killed_pids"] == []

def test_does_not_kill_when_cuda_list_is_invisible(q, monkeypatch):
    cfg = RunnerConfig(queue_root=q.root, kill_orphan_cuda=True)
    monkeypatch.setattr(rp, "compute_apps", lambda: None)
    monkeypatch.setattr(rp, "_kill", lambda pid: pytest.fail("killed blind"))
    assert reap(q, cfg)["killed_pids"] == []
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_reaper.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gpuqueue.reaper'`

- [ ] **Step 3: Write the minimal implementation**

```python
# src/gpuqueue/reaper.py
"""Reaping lives in the runner because it has to run when nothing else is
alive — which is exactly when a leaked job needs reaping."""
from __future__ import annotations

import os
import shutil
import signal
from pathlib import Path

from .claim import release_stale, pid_alive
from .config import RunnerConfig
from .preflight import compute_apps, own_pids
from .queue import QueueRoot

MAX_ATTEMPTS = 1


def _kill(pid: int) -> bool:
    try:
        os.kill(pid, signal.SIGKILL)
        return True
    except OSError:
        return False


def requeue_orphans(queue: QueueRoot,
                    active_ids: set[str] | None) -> tuple[list[str], list[str]]:
    """Requeue running jobs whose process is gone — once, then fail."""
    active = active_ids or set()
    requeued, failed = [], []
    for spec in queue.list_state("running"):
        if spec.id in active:
            continue
        if spec.pid and pid_alive(spec.pid):
            continue
        if spec.attempts < MAX_ATTEMPTS:
            queue.requeue(spec)
            requeued.append(spec.id)
        else:
            spec.error = (f"abandoned after {spec.attempts + 1} attempts; "
                          "the runner or the job died repeatedly")
            queue.finish(spec, ok=False)
            failed.append(spec.id)
    return requeued, failed


def kill_orphan_cuda(protect: set[int]) -> list[int]:
    apps = compute_apps()
    if apps is None:
        return []  # cannot see the list; killing blind is worse than leaking
    exempt = set(protect) | own_pids()
    killed = []
    for app in apps:
        if app["pid"] not in exempt and _kill(app["pid"]):
            killed.append(app["pid"])
    return killed


def clean_partials(queue: QueueRoot) -> list[str]:
    cleaned = []
    work = queue.root / "work"
    if not work.is_dir():
        return cleaned
    live = {s.id for s in queue.list_state("running")}
    for path in work.rglob("*.part"):
        cleaned.append(str(path))
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)
    for d in work.iterdir():
        if d.is_dir() and d.name not in live and not any(d.iterdir()):
            d.rmdir()
    return cleaned


def reap(queue: QueueRoot, cfg: RunnerConfig,
         active_ids: set[str] | None = None) -> dict:
    stale = release_stale(cfg.claim_dir)
    requeued, failed = requeue_orphans(queue, active_ids)
    protect = {s.pid for s in queue.list_state("running") if s.pid}
    killed = kill_orphan_cuda(protect) if cfg.kill_orphan_cuda else []
    cleaned = clean_partials(queue)
    return {"stale_claims": stale, "requeued": requeued, "failed": failed,
            "killed_pids": killed, "cleaned_paths": cleaned}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_reaper.py -v`
Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add src/gpuqueue/reaper.py tests/test_reaper.py
git commit -m "feat: reaper for dead claims, orphan CUDA procs and requeue-once"
```

---

### Task 12: Runner main loop and lane admission

**Files:**
- Create: `src/gpuqueue/runner.py`, `src/gpuqueue/cli_runner.py`
- Test: `tests/test_runner.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `Runner(cfg: RunnerConfig)` with attributes `queue`, `cfg`, `active: dict[str, Active]`, and methods `tick() -> None` (one full cycle: reap, collect, admit), `admit() -> list[str]`, `collect() -> list[str]`, `run_forever() -> None`, `stop() -> None`, `shutdown() -> None`. Dataclass `Active(running: RunningJob, project: ProjectConfig, workdir: Path, claim_cm: object | None)`. `cli_runner.main(argv=None) -> int` with `--config`, `--once`.

**The runner is single-threaded.** `tick()` is one non-blocking pass and the only way anything happens; `run_forever()` is a sleep around it. Concurrency comes from `Popen`, not from threads, so there is no lock anywhere in this module and no state shared between a worker and the loop. Everything the tests drive, they drive by calling `tick()` — which means the daemon's real control flow is what is under test, not a threaded approximation of it.

Admission rules: at most `cfg.cpu_slots` CPU jobs in flight; at most **one** GPU job in flight, run under `gpu_claim`. Pending jobs are considered in submission order (`submitted_at`, then id). A job whose `project` is not in config is failed immediately with a legible error rather than left pending forever.

Ordering within `admit()` is load-bearing: take the card *before* the pending→running rename. The claim is non-blocking (`wait=False`) — a blocking claim in a single-threaded loop would freeze the CPU lane behind whoever holds the card — so a busy card means "leave it pending, try next tick", and a job that never made it to `running/` needs no unwinding. If the rename then loses to another process, the claim is released again.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_runner.py
import json
import threading
import time
import pytest
from pathlib import Path
from gpuqueue import runner as rn
from gpuqueue.runner import Runner
from gpuqueue.config import RunnerConfig, ProjectConfig
from gpuqueue.spec import JobSpec
from gpuqueue.git_ops import git

def _origin(tmp_path):
    o = tmp_path / "origin"
    o.mkdir()
    git(["init", "-q", "-b", "main"], cwd=o)
    git(["config", "user.email", "t@t"], cwd=o)
    git(["config", "user.name", "t"], cwd=o)
    (o / "a.txt").write_text("one\n")
    git(["add", "a.txt"], cwd=o)
    git(["commit", "-qm", "first"], cwd=o)
    return o, git(["rev-parse", "HEAD"], cwd=o).strip()

@pytest.fixture
def env(tmp_path, monkeypatch):
    origin, sha = _origin(tmp_path)
    monkeypatch.setenv("GPU_CLAIM_DIR", str(tmp_path / "claims"))
    cfg = RunnerConfig(
        queue_root=tmp_path / "queue", cpu_slots=2, poll_interval_s=0.01,
        claim_dir=tmp_path / "claims", kill_orphan_cuda=False,
        projects={"p": ProjectConfig(name="p", remote=str(origin),
                                     checkout=tmp_path / "checkout",
                                     commit_artifacts=True)})
    monkeypatch.setattr(rn, "gpu_key", lambda index=0: "test-uuid")
    monkeypatch.setattr(rn, "preflight", lambda allow=None: None)
    r = Runner(cfg)
    return r, sha

def submit(r, sha, job_id, cmd, lane="cpu", artifacts=(), timeout_s=30):
    r.queue.submit(JobSpec.from_dict(dict(
        id=job_id, lane=lane, project="p", commit=sha, branch="main",
        cmd=list(cmd), artifacts=list(artifacts), timeout_s=timeout_s)))

def drain(r, limit=30.0):
    """Tick until nothing is pending or running — what run_forever does,
    without the sleep."""
    deadline = time.monotonic() + limit
    while time.monotonic() < deadline:
        r.tick()
        if not r.active and not r.queue.list_state("pending"):
            return True
        time.sleep(0.02)
    raise AssertionError("runner did not drain the queue in time")

def test_runs_a_cpu_job_to_done(env):
    r, sha = env
    submit(r, sha, "j1", ["sh", "-c", "echo hi"])
    drain(r)
    assert (r.queue.root / "done" / "j1.json").exists()

def test_job_runs_in_a_worktree_at_the_pinned_commit(env):
    r, sha = env
    submit(r, sha, "j1", ["sh", "-c", "cat a.txt"])
    drain(r)
    out, _ = r.queue.log_paths("j1")
    assert out.read_text().strip() == "one"

def test_failing_job_lands_in_failed_with_stderr_tail(env):
    r, sha = env
    submit(r, sha, "j1", ["sh", "-c", "echo bad >&2; exit 4"])
    drain(r)
    body = json.loads((r.queue.root / "failed" / "j1.json").read_text())
    assert body["exit_code"] == 4 and "bad" in body["error"]

def test_cpu_slots_are_respected(env):
    r, sha = env
    for i in range(4):
        submit(r, sha, f"j{i}", ["sleep", "5"])
    assert len(r.admit()) == 2
    assert r.admit() == []
    r.shutdown()

def test_only_one_gpu_job_runs_at_a_time(env):
    r, sha = env
    submit(r, sha, "g1", ["sleep", "5"], lane="gpu")
    submit(r, sha, "g2", ["sleep", "5"], lane="gpu")
    assert len(r.admit()) == 1
    assert r.admit() == []
    r.shutdown()

def test_gpu_and_cpu_lanes_run_concurrently(env):
    r, sha = env
    submit(r, sha, "g1", ["sleep", "5"], lane="gpu")
    submit(r, sha, "c1", ["sleep", "5"], lane="cpu")
    assert sorted(r.admit()) == ["c1", "g1"]
    r.shutdown()

def test_gpu_job_holds_a_claim_file_while_running(env, tmp_path):
    r, sha = env
    submit(r, sha, "g1", ["sleep", "5"], lane="gpu")
    r.admit()
    assert list((tmp_path / "claims").glob("*.lock.json"))
    r.shutdown()

def test_gpu_claim_is_released_when_the_job_ends(env, tmp_path):
    r, sha = env
    submit(r, sha, "g1", ["true"], lane="gpu")
    drain(r)
    assert list((tmp_path / "claims").glob("*.lock.json")) == []

def test_a_busy_card_leaves_the_job_pending_not_failed(env, tmp_path):
    """An outside gpu-claim holder must not consume the queued job."""
    from gpuqueue.claim import gpu_claim
    r, sha = env
    submit(r, sha, "g1", ["true"], lane="gpu")
    with gpu_claim(key="test-uuid", owner="outsider", directory=tmp_path / "claims"):
        assert r.admit() == []
        assert (r.queue.root / "pending" / "g1.json").exists()
    drain(r)
    assert (r.queue.root / "done" / "g1.json").exists()

def test_running_job_records_its_pid(env):
    r, sha = env
    submit(r, sha, "j1", ["sleep", "5"])
    r.admit()
    body = json.loads((r.queue.root / "running" / "j1.json").read_text())
    assert body["pid"] > 0
    r.shutdown()

def test_timeout_marks_failed_and_does_not_retry(env):
    r, sha = env
    submit(r, sha, "j1", ["sleep", "30"], timeout_s=1)
    drain(r, limit=60)
    body = json.loads((r.queue.root / "failed" / "j1.json").read_text())
    assert "timeout" in body["error"].lower()
    assert body["attempts"] == 0

def test_artifacts_are_committed_by_the_main_loop(env):
    r, sha = env
    submit(r, sha, "j1", ["sh", "-c", "mkdir -p runs && echo '{}' > runs/s.json"],
           artifacts=["runs/s.json"])
    drain(r)
    assert (r.cfg.projects["p"].checkout / "runs" / "s.json").exists()

def test_missing_artifact_fails_the_job(env):
    r, sha = env
    submit(r, sha, "j1", ["true"], artifacts=["runs/never.json"])
    drain(r)
    body = json.loads((r.queue.root / "failed" / "j1.json").read_text())
    assert "never.json" in body["error"]

def test_worktree_removed_after_job(env):
    r, sha = env
    submit(r, sha, "j1", ["true"])
    drain(r)
    assert not r.queue.work_dir("j1").exists()

def test_unknown_project_fails_fast(env):
    r, sha = env
    r.queue.submit(JobSpec.from_dict(dict(
        id="j1", lane="cpu", project="nope", commit=sha, branch="main",
        cmd=["true"], artifacts=[], timeout_s=30)))
    r.tick()
    body = json.loads((r.queue.root / "failed" / "j1.json").read_text())
    assert "nope" in body["error"]

def test_unstartable_command_fails_the_job(env):
    r, sha = env
    submit(r, sha, "j1", ["definitely-not-a-real-binary"])
    drain(r)
    body = json.loads((r.queue.root / "failed" / "j1.json").read_text())
    assert "definitely-not-a-real-binary" in body["error"]

def test_shutdown_leaves_killed_jobs_for_the_reaper(env):
    """Graceful stop does not decide the job's fate — it clears the pid and
    leaves it in running/, which is exactly the state the reaper already
    knows how to requeue once."""
    r, sha = env
    submit(r, sha, "j1", ["sleep", "30"])
    r.admit()
    r.shutdown()
    body = json.loads((r.queue.root / "running" / "j1.json").read_text())
    assert body["pid"] is None
    assert r.active == {}

def test_the_runner_spawns_no_threads(env):
    """Concurrency is Popen, not threads. A thread here would put a worker
    and the loop on the same queue files and the same JobSpec objects."""
    r, sha = env
    before = threading.active_count()
    for i in range(3):
        submit(r, sha, f"j{i}", ["sh", "-c", "sleep 0.2; echo ok"])
    r.tick()
    assert threading.active_count() == before
    drain(r)
    assert threading.active_count() == before
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_runner.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gpuqueue.runner'`

- [ ] **Step 3: Write the runner**

```python
# src/gpuqueue/runner.py
"""The sole launcher of queued work on this box.

One thread, one loop. `tick()` reaps, collects what has finished, and admits
what the lanes allow; `run_forever` is a sleep around it. Concurrency is
`Popen`, not threads — several jobs run at once because they are separate
processes, and the runner supervises them by polling.

That is why there is no lock in this file. A thread per job would put a
worker and this loop on the same queue files, the same lane counters and the
same JobSpec objects; single-threaded, none of that state is shared, and
every git call is trivially serialized because there is only one caller.
"""
from __future__ import annotations

import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from . import git_ops
from .claim import gpu_claim, ClaimBusy
from .config import RunnerConfig, ProjectConfig
from .executor import (start_job, poll_job, kill_job, JobResult, RunningJob,
                       StartFailed)
from .gpuid import gpu_key, GpuIdError
from .preflight import preflight, PreflightFailed
from .queue import QueueRoot
from .reaper import reap
from .spec import JobSpec

log = logging.getLogger("gpuqueue.runner")


@dataclass
class Active:
    running: RunningJob
    project: ProjectConfig
    workdir: Path
    claim_cm: object | None = None  # entered gpu_claim, released on settle


class Runner:
    def __init__(self, cfg: RunnerConfig):
        self.cfg = cfg
        self.queue = QueueRoot(cfg.queue_root)
        self.queue.ensure_dirs()
        self.active: dict[str, Active] = {}
        self._stopping = False
        self._reap_due = True  # on start, then after every completion

    # --- lifecycle ----------------------------------------------------
    def run_forever(self) -> None:
        while not self._stopping:
            self.tick()
            if self._stopping:
                break
            time.sleep(self.cfg.poll_interval_s)
        self.shutdown()

    def stop(self) -> None:
        """Signal-handler safe: sets a flag, does no work."""
        self._stopping = True

    def tick(self) -> None:
        if self._reap_due:
            reap(self.queue, self.cfg, active_ids=set(self.active))
            self._reap_due = False
        self.collect()
        self.admit()

    def shutdown(self) -> None:
        """Kill what is running and leave it in running/ with no pid.

        Deciding the fate of an interrupted job is the reaper's business, and
        it already has the rule: requeue once, then fail. Duplicating that
        here would give a job two different retry policies depending on how
        the runner died.
        """
        for job_id, active in list(self.active.items()):
            kill_job(active.running)
            self._release_card(active)
            self._remove_worktree(active)
            spec = active.running.spec
            spec.pid = None
            self.queue.update(spec)
            self.active.pop(job_id, None)
            log.info("stopped %s for shutdown; left for the reaper", job_id)

    # --- admission ----------------------------------------------------
    def _capacity(self, lane: str) -> int:
        limit = self.cfg.cpu_slots if lane == "cpu" else 1
        in_lane = sum(1 for a in self.active.values()
                      if a.running.spec.lane == lane)
        return limit - in_lane

    def admit(self) -> list[str]:
        if self._stopping:
            return []
        pending = sorted(self.queue.list_state("pending"),
                         key=lambda s: (s.submitted_at, s.id))
        started = []
        for spec in pending:
            project = self.cfg.projects.get(spec.project)
            if project is None:
                self._fail_pending(spec, f"unknown project {spec.project!r}; "
                                         "declare it in the runner config")
                continue
            if self._capacity(spec.lane) <= 0:
                continue

            # Take the card before the rename: a job that never reached
            # running/ needs no unwinding if the card is busy.
            claim_cm = None
            if spec.lane == "gpu":
                claim_cm = self._take_card(spec)
                if claim_cm is None:
                    continue

            claimed = self.queue.claim(spec.id)
            if claimed is None:  # cancelled, or another process won the rename
                self._exit_claim(claim_cm)
                continue

            if self._launch(claimed, project, claim_cm):
                started.append(claimed.id)
        return started

    def _take_card(self, spec: JobSpec):
        """Enter a gpu_claim by hand so it can be held across ticks.

        Non-blocking on purpose: `wait=True` in a single-threaded loop would
        stall the CPU lane behind whoever holds the card.
        """
        try:
            preflight()
        except PreflightFailed as e:
            log.warning("%s waiting: %s", spec.id, e)
            return None
        try:
            key = gpu_key()
        except GpuIdError as e:
            # No card will appear on a box that has none; do not queue forever.
            self._fail_pending(spec, f"no usable GPU: {e}")
            return None
        cm = gpu_claim(key=key, owner=f"gpuq:{spec.id}", cmd=spec.cmd,
                       wait=False, directory=self.cfg.claim_dir)
        try:
            cm.__enter__()
        except ClaimBusy as e:
            log.info("%s waiting: %s", spec.id, e)
            return None
        return cm

    def _launch(self, spec: JobSpec, project: ProjectConfig, claim_cm) -> bool:
        try:
            workdir = self._prepare_workdir(spec, project)  # git, on the loop
        except Exception as e:
            self._exit_claim(claim_cm)
            self._fail_running(spec, f"checkout failed: {e}")
            return False
        out_log, err_log = self.queue.log_paths(spec.id)
        try:
            running = start_job(
                spec, workdir, out_log, err_log, project=project,
                extra_env={"GPUQ_JOB_ID": spec.id,
                           "GPUQ_QUEUE_ROOT": str(self.queue.root)})
        except StartFailed as e:
            self._exit_claim(claim_cm)
            self._remove_worktree_at(project, workdir)
            self._fail_running(spec, str(e))
            return False

        spec.pid = running.pid
        self.queue.update(spec)  # the reaper reads this to tell live from dead
        self.active[spec.id] = Active(running=running, project=project,
                                      workdir=workdir, claim_cm=claim_cm)
        log.info("started %s (%s lane, pid %d)", spec.id, spec.lane, running.pid)
        return True

    def _prepare_workdir(self, spec: JobSpec, project: ProjectConfig) -> Path:
        checkout = git_ops.ensure_checkout(project)
        git_ops.git(["fetch", "--quiet", "origin"], cwd=checkout, check=False)
        return git_ops.add_worktree(checkout, self.queue.work_dir(spec.id),
                                    spec.commit)

    # --- collection ---------------------------------------------------
    def collect(self) -> list[str]:
        finished = []
        for job_id, active in list(self.active.items()):
            result = poll_job(active.running)
            if result is None:
                continue
            self.active.pop(job_id, None)
            self._settle(active, result)
            finished.append(job_id)
        if finished:
            self._reap_due = True  # design: reap between jobs
        return finished

    def _settle(self, active: Active, result: JobResult) -> None:
        spec = active.running.spec
        # Free the card before the git work: artifact commits can take a
        # while and nothing about them needs the GPU.
        self._release_card(active)

        spec.pid = None
        spec.exit_code = result.exit_code
        ok = result.exit_code == 0 and not result.timed_out
        if ok:
            try:
                self._collect_artifacts(spec, active.project, active.workdir)
            except Exception as e:
                ok = False
                spec.error = str(e)
        if not ok and spec.error is None:
            spec.error = self._describe_failure(spec, result)

        self._remove_worktree(active)
        self.queue.finish(spec, ok=ok)
        log.info("%s %s", spec.id, "done" if ok else f"failed: {spec.error}")

    def _collect_artifacts(self, spec: JobSpec, project: ProjectConfig,
                           workdir: Path) -> None:
        if not spec.artifacts:
            return
        srcs, rels = [], []
        for rel in spec.artifacts:
            src = workdir / rel
            if not src.exists():
                raise RuntimeError(f"declared artifact not produced: {rel}")
            srcs.append(src)
            rels.append(rel)
        if project.commit_artifacts:
            git_ops.commit_artifacts(project, spec.branch, srcs, rels,
                                     f"artifacts: {spec.id}")

    def _describe_failure(self, spec: JobSpec, result: JobResult) -> str:
        if result.timed_out:
            return (f"timeout after {spec.timeout_s}s; killed. A hung job "
                    "is a bug, not a transient — not retried.")
        if result.oom:
            return ("CUDA out of memory — a configuration error, not a "
                    f"transient; not retried.\n{result.stderr_tail}")
        return f"exit {result.exit_code}\n{result.stderr_tail}"

    # --- teardown helpers ---------------------------------------------
    def _release_card(self, active: Active) -> None:
        self._exit_claim(active.claim_cm)
        active.claim_cm = None

    @staticmethod
    def _exit_claim(claim_cm) -> None:
        if claim_cm is not None:
            claim_cm.__exit__(None, None, None)

    def _remove_worktree(self, active: Active) -> None:
        self._remove_worktree_at(active.project, active.workdir)

    @staticmethod
    def _remove_worktree_at(project: ProjectConfig, workdir: Path) -> None:
        try:
            git_ops.remove_worktree(Path(project.checkout), workdir)
        except Exception:
            shutil.rmtree(workdir, ignore_errors=True)

    # --- failure helpers ----------------------------------------------
    def _fail_pending(self, spec: JobSpec, message: str) -> None:
        claimed = self.queue.claim(spec.id)
        if claimed:
            self._fail_running(claimed, message)

    def _fail_running(self, spec: JobSpec, message: str) -> None:
        spec.error = message
        spec.exit_code = -1
        spec.pid = None
        self.queue.finish(spec, ok=False)
        log.warning("%s failed: %s", spec.id, message)
```

```python
# src/gpuqueue/cli_runner.py
"""gpuq-runner: the supervisor-managed daemon."""
from __future__ import annotations

import argparse
import logging
import signal
import sys

from .config import load_config, default_config_path, ConfigError
from .runner import Runner


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="gpuq-runner")
    p.add_argument("--config", default=None)
    p.add_argument("--once", action="store_true",
                   help="run a single tick and exit (for debugging)")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(message)s")
    try:
        cfg = load_config(args.config or default_config_path())
    except ConfigError as e:
        print(f"gpuq-runner: {e}", file=sys.stderr)
        return 2

    runner = Runner(cfg)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: runner.stop())

    logging.info("runner started: queue=%s cpu_slots=%d projects=%s",
                 cfg.queue_root, cfg.cpu_slots, ", ".join(cfg.projects) or "none")
    if args.once:
        runner.tick()
        return 0
    runner.run_forever()
    logging.info("runner stopped")
    return 0
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_runner.py -v`
Expected: 18 passed

- [ ] **Step 5: Run the whole suite**

Run: `pytest -v`
Expected: all tests pass. Fix any cross-module regressions before committing.

- [ ] **Step 6: Commit**

```bash
git add src/gpuqueue/runner.py src/gpuqueue/cli_runner.py tests/test_runner.py
git commit -m "feat: single-threaded runner loop with lane admission and serialized git"
```

---

### Task 13: Bootstrap and supervisor program file

**Files:**
- Create: `bootstrap.sh`, `supervisor/gpuq-runner.conf`
- Modify: `README.md` (replace the Status section)
- Test: `tests/test_bootstrap.sh`

**Interfaces:**
- Consumes: the installed console scripts.
- Produces: `bootstrap.sh` honouring `GPUQ_PREFIX` (default `/workspace`), `GPUQ_CONFIG` (default `$GPUQ_PREFIX/gpuq.toml`), `QUEUE_ROOT` (default `$GPUQ_PREFIX/queue`), `GPU_CLAIM_DIR` (default `$GPUQ_PREFIX/lock/gpu`), `SUPERVISOR_CONF_DIR` (default `/etc/supervisor/conf.d`). Flags: `--no-supervisor`, `--dry-run`.

The default claim dir here is `$GPUQ_PREFIX/lock/gpu`, not `/var/lock/gpu`: the target is an unprivileged container that may not be able to write under `/var`. `bootstrap.sh` exports `GPU_CLAIM_DIR` into the supervisor environment so every participant on the box derives the same path — a lock is only correct if everyone uses one.

Three things this task got wrong on first contact with a real box, all fixed above:

- **`$PYTHON`, not `python3`.** A box with several interpreters must not have this guessed for it — the runner has to be installed into the one that is 3.11+, which is not always the one called `python3`. The version check names the interpreter it rejected and says how to override it.
- **The checkout step shelled out to bare `python`.** That binary does not exist on most modern distributions; only `python3` does. Under `set -e` it aborted the run before the supervisor program file was installed.
- **A failed clone must not be fatal.** The first run writes a config full of example placeholders and tells you to edit it, so a clone failure there is the *expected* case. Aborting on it meant a first run never installed the supervisor program file at all. It now reports per project and continues.

The bootstrap test picks a 3.11+ interpreter that has `pip`, preferring the repo venv, and **skips** the checks that actually run `bootstrap.sh` when the box has none — printing `SKIP`, rather than reporting a pass it did not earn.

- [ ] **Step 1: Write the failing test**

```bash
#!/usr/bin/env bash
# tests/test_bootstrap.sh — run with: bash tests/test_bootstrap.sh
set -uo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
fails=0
check() { if eval "$2"; then echo "ok   - $1"; else echo "FAIL - $1"; fails=$((fails+1)); fi; }

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

# The checks that actually run bootstrap need a 3.11+ interpreter with pip.
# Prefer the repo venv, then any newer python on PATH. If there is none,
# skip those checks loudly rather than reporting a pass we did not earn.
PYTHON=""
for cand in "$repo/.venv/bin/python" python3.13 python3.12 python3.11 python3; do
  if command -v "$cand" >/dev/null 2>&1 &&
     "$cand" -c 'import sys;sys.exit(0 if sys.version_info>=(3,11) else 1)' 2>/dev/null &&
     "$cand" -c 'import pip' 2>/dev/null; then
    PYTHON="$cand"; break
  fi
done
export PYTHON

check "bootstrap.sh is executable" "[ -x '$repo/bootstrap.sh' ]"
check "supervisor conf is shipped" "[ -f '$repo/supervisor/gpuq-runner.conf' ]"
check "shellcheck-clean (skipped if absent)" \
  "! command -v shellcheck >/dev/null || shellcheck '$repo/bootstrap.sh'"
check "sets -euo pipefail" "grep -q 'set -euo pipefail' '$repo/bootstrap.sh'"
check "supervisor conf runs gpuq-runner" \
  "grep -q 'command=.*gpuq-runner' '$repo/supervisor/gpuq-runner.conf'"
check "supervisor conf autorestarts" \
  "grep -q 'autorestart=true' '$repo/supervisor/gpuq-runner.conf'"
check "supervisor conf passes GPU_CLAIM_DIR" \
  "grep -q 'GPU_CLAIM_DIR' '$repo/supervisor/gpuq-runner.conf'"

if [ -z "$PYTHON" ]; then
  echo "SKIP - bootstrap install checks: no Python 3.11+ with pip on this box"
  echo "---"; [ "$fails" -eq 0 ] && echo "all passed" || { echo "$fails failed"; exit 1; }
  exit 0
fi

out="$(GPUQ_PREFIX="$tmp/ws" SUPERVISOR_CONF_DIR="$tmp/conf" \
       bash "$repo/bootstrap.sh" --dry-run --no-supervisor 2>&1)"
check "dry run touches nothing" "[ ! -d '$tmp/ws' ]"
check "dry run reports the queue root it would create" \
  "grep -q '$tmp/ws/queue' <<<'$out'"

GPUQ_PREFIX="$tmp/ws" SUPERVISOR_CONF_DIR="$tmp/conf" \
  bash "$repo/bootstrap.sh" --no-supervisor >/dev/null 2>&1
check "creates the queue tree" "[ -d '$tmp/ws/queue/pending' ]"
check "creates the claim dir" "[ -d '$tmp/ws/lock/gpu' ]"
check "writes a config when absent" "[ -f '$tmp/ws/gpuq.toml' ]"

before="$(cat "$tmp/ws/gpuq.toml")"
echo "# edited by hand" >> "$tmp/ws/gpuq.toml"
GPUQ_PREFIX="$tmp/ws" SUPERVISOR_CONF_DIR="$tmp/conf" \
  bash "$repo/bootstrap.sh" --no-supervisor >/dev/null 2>&1
check "second run is idempotent and preserves an edited config" \
  "grep -q 'edited by hand' '$tmp/ws/gpuq.toml'"

GPUQ_PREFIX="$tmp/ws" SUPERVISOR_CONF_DIR="$tmp/conf" \
  bash "$repo/bootstrap.sh" >/dev/null 2>&1
check "installs the supervisor program file" \
  "[ -f '$tmp/conf/gpuq-runner.conf' ]"

echo "---"; [ "$fails" -eq 0 ] && echo "all passed" || { echo "$fails failed"; exit 1; }
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `bash tests/test_bootstrap.sh`
Expected: FAIL — bootstrap.sh does not exist

- [ ] **Step 3: Write bootstrap.sh and the supervisor program file**

```bash
#!/usr/bin/env bash
# bootstrap.sh — take a bare box to a working runner, idempotently.
#
# Host identity lives in one variable: GPUQ_PREFIX. Rebuilding a destroyed
# box is an ssh-target edit plus a run of this script.
set -euo pipefail

GPUQ_PREFIX="${GPUQ_PREFIX:-/workspace}"
QUEUE_ROOT="${QUEUE_ROOT:-$GPUQ_PREFIX/queue}"
GPU_CLAIM_DIR="${GPU_CLAIM_DIR:-$GPUQ_PREFIX/lock/gpu}"
GPUQ_CONFIG="${GPUQ_CONFIG:-$GPUQ_PREFIX/gpuq.toml}"
SUPERVISOR_CONF_DIR="${SUPERVISOR_CONF_DIR:-/etc/supervisor/conf.d}"
# Which interpreter the runner is installed into. A box with several
# Pythons must not have this guessed for it: the runner has to live in
# the one that is 3.11+, which is not always the one called python3.
PYTHON="${PYTHON:-python3}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DRY_RUN=0
USE_SUPERVISOR=1
for arg in "$@"; do
  case "$arg" in
    --dry-run)       DRY_RUN=1 ;;
    --no-supervisor) USE_SUPERVISOR=0 ;;
    -h|--help)
      sed -n '2,8p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "bootstrap: unknown argument: $arg" >&2; exit 2 ;;
  esac
done

say() { printf '%s\n' "$*" >&2; }
run() { if [ "$DRY_RUN" -eq 1 ]; then say "would: $*"; else "$@"; fi; }

say "prefix:      $GPUQ_PREFIX"
say "queue root:  $QUEUE_ROOT"
say "claim dir:   $GPU_CLAIM_DIR"
say "config:      $GPUQ_CONFIG"
say "python:      $PYTHON"

# 1. install the package
# Check the interpreter first: pip's requires-python failure names a version
# but not what to do about it, and this runs on boxes nobody built by hand.
"$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' || {
  say "bootstrap: need Python 3.11+ (tomllib); $PYTHON is $("$PYTHON" -V 2>&1)"
  say "           set PYTHON=/path/to/python3.11 if the box has another one"
  exit 1
}

if [ "$DRY_RUN" -eq 1 ]; then
  say "would: $PYTHON -m pip install -e $REPO_DIR"
else
  "$PYTHON" -m pip install --quiet -e "$REPO_DIR"
fi

# 2. state directories
for d in pending running done failed logs work; do
  run mkdir -p "$QUEUE_ROOT/$d"
done
run mkdir -p "$GPU_CLAIM_DIR"

# 3. config, written once and never overwritten
if [ "$DRY_RUN" -eq 1 ]; then
  say "would: write $GPUQ_CONFIG if absent"
elif [ -f "$GPUQ_CONFIG" ]; then
  say "config exists, leaving it alone: $GPUQ_CONFIG"
else
  sed -e "s|^root = .*|root = \"$QUEUE_ROOT\"|" \
      "$REPO_DIR/gpuq.example.toml" > "$GPUQ_CONFIG"
  say "wrote $GPUQ_CONFIG — declare your projects in it, then rerun"
fi

# 4. clone declared checkouts
#
# Never fatal. The first run writes a config full of example placeholders and
# tells you to edit it, so a failed clone here is the expected case, not an
# error -- and aborting would stop the supervisor program file from being
# installed at all. Report per project and carry on; the runner fails jobs
# for an unclonable project with a legible message of its own.
if [ "$DRY_RUN" -eq 0 ] && [ -f "$GPUQ_CONFIG" ]; then
  GPUQ_CONFIG="$GPUQ_CONFIG" "$PYTHON" - <<'PYEOF' || say "checkout step reported problems; continuing"
import os
import sys
from pathlib import Path
from gpuqueue.config import load_config
from gpuqueue.git_ops import ensure_checkout

cfg = load_config(Path(os.environ["GPUQ_CONFIG"]))
for name, project in cfg.projects.items():
    try:
        print(f"checkout {name}: {ensure_checkout(project)}", file=sys.stderr)
    except Exception as e:
        print(f"checkout {name}: SKIPPED -- {e}", file=sys.stderr)
PYEOF
fi

# 5. supervisor program file, shipped rather than hand-written
if [ "$USE_SUPERVISOR" -eq 1 ]; then
  run mkdir -p "$SUPERVISOR_CONF_DIR"
  if [ "$DRY_RUN" -eq 1 ]; then
    say "would: install $SUPERVISOR_CONF_DIR/gpuq-runner.conf"
  else
    sed -e "s|@QUEUE_ROOT@|$QUEUE_ROOT|g" \
        -e "s|@GPU_CLAIM_DIR@|$GPU_CLAIM_DIR|g" \
        -e "s|@GPUQ_CONFIG@|$GPUQ_CONFIG|g" \
        -e "s|@GPUQ_PREFIX@|$GPUQ_PREFIX|g" \
        "$REPO_DIR/supervisor/gpuq-runner.conf" \
        > "$SUPERVISOR_CONF_DIR/gpuq-runner.conf"
    if command -v supervisorctl >/dev/null 2>&1; then
      supervisorctl reread  || say "supervisorctl reread failed; is supervisord running?"
      supervisorctl update  || true
      supervisorctl restart gpuq-runner || supervisorctl start gpuq-runner || true
    else
      say "supervisorctl not found; program file installed but not started"
    fi
  fi
fi

say "bootstrap complete"
```

```ini
; supervisor/gpuq-runner.conf — installed by bootstrap.sh, not by hand.
; Placeholders are substituted at install time; that is what makes a
; rebuilt box identical rather than similar.
[program:gpuq-runner]
command=gpuq-runner --config @GPUQ_CONFIG@
directory=@GPUQ_PREFIX@
autostart=true
autorestart=true
startsecs=5
stopasgroup=true
killasgroup=true
stopwaitsecs=30
redirect_stderr=true
stdout_logfile=@QUEUE_ROOT@/logs/runner.log
stdout_logfile_maxbytes=50MB
stdout_logfile_backups=3
environment=QUEUE_ROOT="@QUEUE_ROOT@",GPU_CLAIM_DIR="@GPU_CLAIM_DIR@",GPUQ_CONFIG="@GPUQ_CONFIG@",PYTHONUNBUFFERED="1"
```

`stopasgroup` and `killasgroup` are not incidental: without them a supervisor restart leaves the runner's job subprocesses alive, still holding the card, with nothing tracking them.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `chmod +x bootstrap.sh && bash tests/test_bootstrap.sh`
Expected: `all passed`

- [ ] **Step 5: Update the README Status section**

Replace the `## Status` section with:

```markdown
## Status

Implemented. Install with `bootstrap.sh`; see `docs/design.md` for the
architecture and `docs/plans/` for the implementation plan.

Per-job VRAM limits are not implemented: a job that holds the card holds
all of it. See "Not in scope" in `docs/design.md`.

Originating context: `Daniel-T-S-Adams/wgan-synthetic`, which needs six agents
to research six datasets in parallel against a single RTX 4060.
```

- [ ] **Step 6: End-to-end verification on a real box**

Run, on the target box:
```bash
GPUQ_PREFIX=/workspace ./bootstrap.sh
# declare your project in /workspace/gpuq.toml, then:
./bootstrap.sh
gpuq submit --project wgan-synthetic --commit "$(git rev-parse HEAD)" \
  --branch main --lane gpu --artifact runs/smoke.json \
  -- python -c "import json,os; os.makedirs('runs',exist_ok=True); json.dump({'ok':1},open('runs/smoke.json','w'))"
gpuq list
supervisorctl status gpuq-runner
```
Expected: the job moves pending → running → done within a few seconds, `gpuq show <id>` reports `exit_code: 0`, and `runs/smoke.json` appears in the project checkout.

- [ ] **Step 7: Commit**

```bash
git add bootstrap.sh supervisor/gpuq-runner.conf tests/test_bootstrap.sh README.md
git commit -m "feat: idempotent bootstrap and shipped supervisor program file"
```

---

### Task 14: `gpuq wait` and the agent-facing skill

The consumers here are agents, and an agent has two needs that look like two features but are one: sometimes it wants the result now, and sometimes it wants to submit six jobs and come back. Both are `wait`.

**Files:**
- Create: `skills/gpu-jobs/SKILL.md`
- Modify: `src/gpuqueue/cli_gpuq.py` (add `wait`, add `--wait` to `submit`), `bootstrap.sh` (install the skill), `tests/test_cli_gpuq.py`

**Interfaces:**
- Produces: subcommand `gpuq wait <id> [--timeout S] [--poll S]` and flag `gpuq submit --wait`. Helper `wait_for(q: QueueRoot, job_id: str, timeout: float | None, poll: float) -> int`.
- Exit codes: `0` the job is done, `1` it failed, `2` no such job, `124` the wait timed out (as `timeout(1)`).

Three decisions, all of which an implementer would otherwise have to guess at:

**Waiting on an already-finished job returns immediately.** This is what makes submit-and-wait and submit-and-go-away the same primitive rather than two features. An agent never has to decide up front which mode it is in; calling `wait` late costs nothing, so "submit, do other work, wait whenever" needs no new machinery.

**`--timeout` is the waiter's patience, not the job's.** It does not cancel, kill or otherwise touch the job — the job's own wall-clock limit is `timeout_s` in its spec, enforced by the runner. A timed-out wait is a thing the caller may simply do again. Say this in the help text; it is easy to get backwards.

**A job that briefly cannot be found is not a missing job.** `find()` scans the state directories in order, and the reaper's requeue moves a job *backwards* from `running/` to `pending/`. A scan that passes `pending` before the move and reads `running` after it sees the job in neither. So: a `None` on the *first* lookup means no such job and exits 2; a `None` after that is a transient the loop rides out. Without this, a requeue makes a waiting agent believe its job vanished.

`wait` polls the queue tree directly. It needs no daemon and no IPC, which is the same property that lets `gpuq list` work when everything else is broken.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cli_gpuq.py`:

```python
import threading
import time
from gpuqueue.queue import QueueRoot
from gpuqueue.spec import JobSpec

def _submit(root, job_id="j1"):
    main(["submit", "--project", "p", "--commit", "abc", "--branch", "main",
          "--id", job_id, "--", "true"])

def test_wait_returns_zero_for_an_already_finished_job(root, capsys):
    _submit(root)
    q = QueueRoot(root)
    q.finish(q.claim("j1"), ok=True)
    capsys.readouterr()
    start = time.monotonic()
    assert main(["wait", "j1"]) == 0
    assert time.monotonic() - start < 1.0  # immediate, not one poll interval

def test_wait_returns_one_for_a_failed_job(root):
    _submit(root)
    q = QueueRoot(root)
    spec = q.claim("j1")
    spec.error = "boom"
    q.finish(spec, ok=False)
    assert main(["wait", "j1"]) == 1

def test_wait_blocks_until_the_job_finishes(root):
    _submit(root)
    q = QueueRoot(root)
    q.claim("j1")

    def finish_soon():
        time.sleep(0.3)
        state, spec = q.find("j1")
        q.finish(spec, ok=True)

    t = threading.Thread(target=finish_soon)
    t.start()
    assert main(["wait", "j1", "--poll", "0.05"]) == 0
    t.join()

def test_wait_timeout_exits_124_and_leaves_the_job_alone(root):
    _submit(root)
    QueueRoot(root).claim("j1")
    assert main(["wait", "j1", "--timeout", "0.2", "--poll", "0.05"]) == 124
    assert (root / "running" / "j1.json").exists()  # not cancelled

def test_wait_unknown_job_exits_2(root):
    assert main(["wait", "nope", "--timeout", "0.2"]) == 2

def test_wait_rides_out_a_requeue(root):
    """The reaper moves a job running -> pending. A scan that passes pending
    before the move and reads running after it sees neither; that is a
    transient, not a missing job."""
    _submit(root)
    q = QueueRoot(root)
    spec = q.claim("j1")

    def requeue_then_finish():
        time.sleep(0.2)
        q.requeue(spec)
        time.sleep(0.2)
        again = q.claim("j1")
        q.finish(again, ok=True)

    t = threading.Thread(target=requeue_then_finish)
    t.start()
    assert main(["wait", "j1", "--poll", "0.05"]) == 0
    t.join()

def test_submit_wait_does_both(root, capsys):
    q = QueueRoot(root)
    q.ensure_dirs()

    def finish_soon():
        time.sleep(0.3)
        found = q.find("j1")
        while found is None or found[0] == "pending":
            if found and found[0] == "pending":
                q.claim("j1")
            time.sleep(0.05)
            found = q.find("j1")
        q.finish(found[1], ok=True)

    t = threading.Thread(target=finish_soon)
    t.start()
    rc = main(["submit", "--project", "p", "--commit", "abc", "--branch", "main",
               "--id", "j1", "--wait", "--poll", "0.05", "--", "true"])
    t.join()
    assert rc == 0
```

These tests use threads to drive the queue from the outside — that is a test harness standing in for the runner, and does not contradict the runner itself being single-threaded.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_cli_gpuq.py -v`
Expected: FAIL — `argument command: invalid choice: 'wait'`

- [ ] **Step 3: Implement**

In `src/gpuqueue/cli_gpuq.py`, add `import time`, then:

```python
WAIT_TIMED_OUT = 124  # as timeout(1), so a shell can tell the cases apart


def wait_for(q: QueueRoot, job_id: str, timeout: float | None,
             poll: float) -> int:
    """Block until the job is done or failed. Returns the exit code.

    Returns immediately for a job that has already finished — which is what
    lets a caller submit, go and do something else, and wait whenever it
    suits. Waiting late costs nothing.
    """
    deadline = None if timeout is None else time.monotonic() + timeout
    seen = False
    while True:
        found = q.find(job_id)
        if found is None:
            if not seen:
                print(f"gpuq: no such job: {job_id}", file=sys.stderr)
                return 2
            # Seen before, missing now: the reaper is moving it between
            # directories. Not a missing job.
        else:
            seen = True
            state, spec = found
            if state in ("done", "failed"):
                if state == "done":
                    print(f"done {job_id}")
                    return 0
                print(f"failed {job_id}: {spec.error or 'no error recorded'}",
                      file=sys.stderr)
                return 1
        if deadline is not None and time.monotonic() >= deadline:
            state = found[0] if found else "unknown"
            print(f"gpuq: {job_id} still {state} after {timeout}s; "
                  "the job is untouched, wait again when you like",
                  file=sys.stderr)
            return WAIT_TIMED_OUT
        time.sleep(poll)


def _cmd_wait(args) -> int:
    return wait_for(_queue(args), args.id, args.timeout, args.poll)
```

Extend `_cmd_submit` so that, after it prints the id:

```python
    print(job_id)
    if getattr(args, "wait", False):
        return wait_for(q, job_id, args.timeout, args.poll)
    return 0
```

And in `build_parser`, add the flags to `submit` and the new subcommand:

```python
    s.add_argument("--wait", action="store_true",
                   help="block until the job finishes, then exit with its result")
    s.add_argument("--timeout", type=float, default=None,
                   help="how long to wait, not how long the job may run")
    s.add_argument("--poll", type=float, default=2.0)

    w = sub.add_parser("wait", help="block until a job finishes")
    w.add_argument("id")
    w.add_argument("--timeout", type=float, default=None,
                   help="how long YOU wait. The job is never cancelled by "
                        "this; its own limit is timeout_s in the spec.")
    w.add_argument("--poll", type=float, default=2.0)
    w.set_defaults(func=_cmd_wait)
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_cli_gpuq.py -v`
Expected: 20 passed

- [ ] **Step 5: Write the skill**

Create `skills/gpu-jobs/SKILL.md`:

```markdown
---
name: gpu-jobs
description: Use when running training, evaluation or any GPU work on this box, and when a long CPU job would otherwise block you - submit it to the queue instead of running it directly, then either wait for it or go do something else.
---

# Running work on a shared GPU box

This box has one GPU and several people or agents who want it. Running a
training script directly will either fail on an OOM half an hour in, or make
someone else's run mysteriously slow. Submit it instead.

## Submit

    gpuq submit --project <name> --commit "$(git rev-parse HEAD)" \
      --branch "$(git rev-parse --abbrev-ref HEAD)" \
      --lane gpu --artifact runs/glove/v0/summary.json \
      -- python -m src.train.train_wgan_gp --config configs/glove/v0.yaml

It prints a job id and returns at once. Then choose:

**You need the result now:**

    id=$(gpuq submit … )
    gpuq wait "$id"        # 0 done, 1 failed, 124 you gave up waiting

**You have other work to do:** do it, then come back and `gpuq wait "$id"`
whenever you like. If the job already finished, `wait` returns immediately —
there is no penalty for waiting late, and no need to decide up front.

Six datasets to process? Submit all six, then wait on them one at a time.
The runner will already have been working through them.

## Which lane

- `--lane gpu` — training, anything that calls CUDA. One at a time, box-wide.
- `--lane cpu` — data fetching, profiling, analysis. Several at once.

Putting CPU work in the GPU lane blocks everyone else's training for no
reason. Putting GPU work in the CPU lane means several jobs hit the card at
once, which is the failure this queue exists to prevent.

## Always pin the commit

`--commit "$(git rev-parse HEAD)"` is not optional. The runner checks out
that exact tree, so a number it reports can be traced to a configuration
someone can read back. A branch name alone lets the tree move under a
queued job and produce a result nobody can reproduce.

Declare what you want kept with `--artifact` (repeatable, paths relative to
the repo root). The runner collects those and can commit them; anything
else your job writes dies with the box.

## When something goes wrong

    gpuq list                  # everything, by state
    gpuq show <id>             # the spec, plus paths to its stdout/stderr logs
    gpuq cancel <id>           # only while still pending

A failed job carries its stderr tail in `error`, so `gpuq show` usually
tells you what happened without opening the logs.

A job that fails on CUDA out-of-memory is reported as such and is never
retried: it is a configuration problem, not a transient. Make the model or
the batch smaller.

## Do not

- Run GPU work directly. `gpu-claim -- <cmd>` if you truly must run
  something interactively; it takes the same lock the queue does.
- Run git in a checkout the runner owns. It manages those, and a concurrent
  checkout corrupts the tree under a running job.
```

- [ ] **Step 6: Install it from bootstrap**

In `bootstrap.sh`, after the supervisor section, add:

```bash
# 6. agent skill, so anything working on this box knows to queue its work
GPUQ_SKILLS_DIR="${GPUQ_SKILLS_DIR:-$HOME/.claude/skills}"
run mkdir -p "$GPUQ_SKILLS_DIR/gpu-jobs"
if [ "$DRY_RUN" -eq 1 ]; then
  say "would: install skill to $GPUQ_SKILLS_DIR/gpu-jobs/SKILL.md"
else
  cp "$REPO_DIR/skills/gpu-jobs/SKILL.md" "$GPUQ_SKILLS_DIR/gpu-jobs/SKILL.md"
  say "skill installed: $GPUQ_SKILLS_DIR/gpu-jobs/SKILL.md"
fi
```

Copied, not symlinked: the skill must survive the repo checkout moving, and
an agent reading a dangling symlink gets nothing with no explanation.

- [ ] **Step 7: Verify end to end**

```bash
gpuq submit --project wgan-synthetic --commit "$(git rev-parse HEAD)" \
  --branch main --lane cpu --wait --poll 1 -- sh -c 'sleep 3; echo ok'
echo "exit: $?"
```
Expected: prints the id, blocks about three seconds, prints `done <id>`, exits 0.

- [ ] **Step 8: Commit**

```bash
git add src/gpuqueue/cli_gpuq.py tests/test_cli_gpuq.py skills/gpu-jobs/SKILL.md bootstrap.sh
git commit -m "feat: gpuq wait, and a skill telling agents to queue their work"
```

---

## Self-review against the spec

**Spec coverage:**

| `docs/design.md` section | Task |
|---|---|
| Two lanes, cpu_slots default 4 | 8, 12 |
| Queue directory tree, atomic rename | 2 |
| Job spec fields, pinned `commit` | 1 |
| `dedupe_key` idempotent resubmission | 2, 3 |
| Runner loop, workers never touch git | 9, 12 (single-threaded by construction; asserted by `test_the_runner_spawns_no_threads`) |
| Per-project config | 8 |
| Reaper: dead claims, orphan CUDA, `.part`, requeue-once | 11 |
| Wall-clock watchdog | 10 (`poll_job`), driven by 12 |
| Lock path / key derivation / claim file | 4, 5 — derivation is `nvidia-smi` here, normalized so a torch-based implementation of the protocol takes the same lock (design gap 2) |
| Preflight refuses on foreign processes | 6, 7 (`gpu-claim`), 12 (queued GPU jobs, in `_take_card`) |
| Failure table (non-zero, runner death, timeout, OOM, duplicate) | 2, 10, 11, 12 |
| Bootstrap, shipped supervisor config, one host variable | 13 |
| Consumers may be agents, not people (design "Constraints") | 14 — `gpuq wait` plus a skill; both submit-and-wait and submit-and-go-away fall out of one primitive |
| Not in scope (multi-GPU, multi-host, durable storage, auth) | no tasks, correctly |

**Deliberately deferred:** per-job VRAM/memory limits, per the 2026-08-05 decision recorded in Global Constraints.

**Known adjustments an implementer will hit:**
- Task 3 Step 3 notes the `--lane` `choices=` conflict with `test_submit_invalid_lane_exits_nonzero`; the resolution (drop `choices=`) is stated there.
- `test_dedupe_prints_existing_id_for_pending` in Task 3 relies on generated ids; if it proves brittle, pass explicit `--id j1` / `--id j2` and assert the printed value equals `j1` both times.
- Task 12's `test_a_busy_card_leaves_the_job_pending_not_failed` takes the lock from the test process itself. `flock` is per-open-file-description, not per-process, so this genuinely excludes — but if it ever passes vacuously, promote it to the subprocess form used in `tests/test_claim.py`.
- A held GPU claim is entered and exited by hand (`cm.__enter__()` / `cm.__exit__()`), because the claim outlives the function that takes it. Every path that abandons a job — busy card, failed checkout, unstartable command, shutdown — must go through `_exit_claim`, or the card stays locked by a runner that is no longer using it.

**Rejected alternatives, so they are not re-proposed:**
- *A thread per job.* Obvious, and wrong here: it puts a worker and the loop on the same queue files, lane counters and `JobSpec` objects, then needs a lock around each and a handshake so the pid lands before the reaper looks. `Popen` already provides the concurrency; polling it costs one `tick()`.
- *Importing torch for GPU identity, even guarded.* Costs seconds of import and a CUDA context in the daemon that should never touch the card, and excludes JAX and TensorFlow users from a lock that is supposed to be host-wide. `normalize_gpu_uuid` gets the interoperability without the import.
- *A shared checkout with `git checkout` per job.* Two concurrent jobs at different pinned commits rewrite the tree under each other mid-run — see design gap 1. Per-job worktrees are not an optimization; they are what makes `commit` mean anything.
