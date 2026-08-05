# gpu-queue-management Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the host-level GPU arbitration and job queueing layer described in `docs/design.md` — an advisory GPU lock, a directory-backed job queue, a CLI, and a supervisor-managed runner that executes queued jobs one-GPU-at-a-time while running CPU jobs concurrently.

**Architecture:** A single Python package `gpuq` with no runtime dependencies beyond the standard library. The queue is a directory tree whose state transitions are atomic `rename(2)` calls. The lock is `flock` on a file named by GPU UUID. The runner is one process that owns all git operations and all subprocess launching. Two console entry points: `gpu-claim` (wrap any command in the GPU lock) and `gpuq` (submit and inspect jobs).

**Tech Stack:** Python 3.11+ (stdlib only — `fcntl`, `subprocess`, `tomllib`, `json`, `pathlib`), pytest for tests, `nvidia-smi` shelled out for GPU identity, supervisor for process management, bash for bootstrap.

## Global Constraints

- **No runtime dependencies outside the standard library.** This installs on boxes that have arbitrary ML stacks; adding a dependency risks conflicting with them. `torch` in particular must NOT be imported — a JAX project must be able to use this lock. Test-only dependency on `pytest` is fine.
- **Python 3.11 minimum**, for `tomllib` in the standard library.
- **Every test must pass without a GPU.** CI and development machines have none. GPU identity is obtained by shelling out to `nvidia-smi`, which tests monkeypatch. A test that requires real CUDA is a broken test.
- **Queue state transitions use `os.rename` only.** Never copy-then-delete, never write-in-place to move a job between states. Atomicity is what prevents two runner threads claiming one job.
- **The lock file format is a pinned protocol**, not an implementation detail. Lock path `$GPU_CLAIM_DIR/gpu-<uuid>.lock` (default dir `/var/lock/gpu`), claim JSON with exactly the keys `pid`, `owner`, `cmd`, `started_at`. Changing any of these breaks interoperation with other installations and requires a version bump documented in `docs/design.md`.
- **GPU key derivation is the UUID**, never the index. Two processes with different `CUDA_VISIBLE_DEVICES` both see their card as index 0; an index-keyed lock would hand them different locks for the same physical card.
- **Workers never run git.** Only the runner's main loop performs git operations. Any code path that shells out to git from a worker thread is a bug.
- Commit after every task. Use conventional-commit prefixes (`feat:`, `test:`, `fix:`, `docs:`, `chore:`).

## File Structure

| Path | Responsibility |
|---|---|
| `pyproject.toml` | Package metadata, console scripts, pytest config |
| `src/gpuq/config.py` | Load and validate `queue.toml`; resolve paths |
| `src/gpuq/gpuinfo.py` | GPU identity and foreign-process detection via `nvidia-smi` |
| `src/gpuq/lock.py` | `flock`-based claim, the pinned lock protocol |
| `src/gpuq/cli_claim.py` | `gpu-claim` entry point — wrap a command in the lock |
| `src/gpuq/spec.py` | Job spec dataclass, serialization, validation |
| `src/gpuq/store.py` | Queue directory tree, atomic state transitions, dedupe |
| `src/gpuq/cli.py` | `gpuq` entry point — submit, list, show, logs, cancel |
| `src/gpuq/gitops.py` | Checkout at a commit, commit artifacts. Runner-only. |
| `src/gpuq/reaper.py` | Dead-claim release, orphan kill, partial-output cleanup |
| `src/gpuq/runner.py` | The daemon loop: lane admission, execution, watchdog |
| `bootstrap.sh` | Bare box → working runner, idempotent |
| `supervisor/gpuq-runner.conf` | Supervisor program definition |

---

### Task 1: Package skeleton and configuration

Nothing else can be tested until the package imports and knows where the queue lives.

**Files:**
- Create: `pyproject.toml`, `src/gpuq/__init__.py`, `src/gpuq/config.py`, `tests/test_config.py`, `.gitignore`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `Project` — frozen dataclass: `name: str`, `remote: str`, `checkout: Path`, `venv: Optional[Path]`, `commit_artifacts: bool`.
  - `Config` — frozen dataclass: `queue_root: Path`, `cpu_slots: int`, `lock_dir: Path`, `projects: Dict[str, Project]`.
  - `load_config(path: Path) -> Config`
  - `DEFAULT_CONFIG_PATH: Path` — `/etc/gpuq/queue.toml`.

- [ ] **Step 1: Create the package skeleton**

Create `.gitignore`:

```
__pycache__/
*.egg-info/
.venv/
.pytest_cache/
dist/
```

Create `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "gpuq"
version = "0.1.0"
description = "Host-level GPU arbitration and job queueing for shared single-GPU boxes"
requires-python = ">=3.11"
dependencies = []

[project.optional-dependencies]
dev = ["pytest>=8"]

[project.scripts]
gpuq = "gpuq.cli:main"
gpu-claim = "gpuq.cli_claim:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

Create `src/gpuq/__init__.py` containing only:

```python
__version__ = "0.1.0"
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_config.py`:

```python
from pathlib import Path

import pytest

from gpuq.config import Config, Project, load_config


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "queue.toml"
    path.write_text(text)
    return path


def test_loads_queue_settings(tmp_path: Path):
    cfg = load_config(_write(tmp_path, """
[queue]
root = "/workspace/queue"
cpu_slots = 4
"""))
    assert cfg.queue_root == Path("/workspace/queue")
    assert cfg.cpu_slots == 4
    assert cfg.projects == {}


def test_cpu_slots_defaults_to_four(tmp_path: Path):
    cfg = load_config(_write(tmp_path, """
[queue]
root = "/workspace/queue"
"""))
    assert cfg.cpu_slots == 4


def test_lock_dir_defaults(tmp_path: Path):
    cfg = load_config(_write(tmp_path, """
[queue]
root = "/workspace/queue"
"""))
    assert cfg.lock_dir == Path("/var/lock/gpu")


def test_loads_projects(tmp_path: Path):
    cfg = load_config(_write(tmp_path, """
[queue]
root = "/workspace/queue"

[project.wgan-synthetic]
remote = "git@github.com:example/wgan-synthetic.git"
checkout = "/workspace/checkouts/wgan-synthetic"
venv = "/workspace/checkouts/wgan-synthetic/.venv"
commit_artifacts = true
"""))
    project = cfg.projects["wgan-synthetic"]
    assert isinstance(project, Project)
    assert project.name == "wgan-synthetic"
    assert project.checkout == Path("/workspace/checkouts/wgan-synthetic")
    assert project.venv == Path("/workspace/checkouts/wgan-synthetic/.venv")
    assert project.commit_artifacts is True


def test_venv_is_optional(tmp_path: Path):
    cfg = load_config(_write(tmp_path, """
[queue]
root = "/workspace/queue"

[project.plain]
remote = "git@github.com:example/plain.git"
checkout = "/workspace/checkouts/plain"
"""))
    assert cfg.projects["plain"].venv is None
    assert cfg.projects["plain"].commit_artifacts is False


def test_missing_queue_root_is_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match="queue.root"):
        load_config(_write(tmp_path, "[queue]\ncpu_slots = 2\n"))


def test_non_positive_cpu_slots_is_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match="cpu_slots"):
        load_config(_write(tmp_path, """
[queue]
root = "/workspace/queue"
cpu_slots = 0
"""))


def test_missing_file_names_the_path(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="nope.toml"):
        load_config(tmp_path / "nope.toml")
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `python -m pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gpuq.config'`

- [ ] **Step 4: Implement the config loader**

Create `src/gpuq/config.py`:

```python
"""Load the box's queue configuration.

One file describes the whole box: where the queue lives, how many CPU jobs
may run at once, and which project checkouts the runner owns. Keeping it in
one place is what lets bootstrap.sh rebuild a box without hand edits.
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

DEFAULT_CONFIG_PATH = Path("/etc/gpuq/queue.toml")
DEFAULT_CPU_SLOTS = 4
DEFAULT_LOCK_DIR = Path("/var/lock/gpu")


@dataclass(frozen=True)
class Project:
    """A repository the runner checks out and runs jobs against."""

    name: str
    remote: str
    checkout: Path
    venv: Optional[Path] = None
    commit_artifacts: bool = False


@dataclass(frozen=True)
class Config:
    queue_root: Path
    cpu_slots: int = DEFAULT_CPU_SLOTS
    lock_dir: Path = DEFAULT_LOCK_DIR
    projects: Dict[str, Project] = field(default_factory=dict)


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> Config:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No queue config at {path}")
    payload = tomllib.loads(path.read_text())

    queue = payload.get("queue", {})
    root = queue.get("root")
    if not root:
        raise ValueError(f"{path}: queue.root is required")

    cpu_slots = int(queue.get("cpu_slots", DEFAULT_CPU_SLOTS))
    if cpu_slots < 1:
        raise ValueError(f"{path}: queue.cpu_slots must be >= 1, got {cpu_slots}")

    lock_dir = Path(queue.get("lock_dir", DEFAULT_LOCK_DIR))

    projects: Dict[str, Project] = {}
    for name, block in payload.get("project", {}).items():
        venv = block.get("venv")
        projects[name] = Project(
            name=name,
            remote=block["remote"],
            checkout=Path(block["checkout"]),
            venv=Path(venv) if venv else None,
            commit_artifacts=bool(block.get("commit_artifacts", False)),
        )

    return Config(
        queue_root=Path(root),
        cpu_slots=cpu_slots,
        lock_dir=lock_dir,
        projects=projects,
    )
```

- [ ] **Step 5: Run the tests**

Run: `python -m pytest tests/test_config.py -v`
Expected: PASS, all eight.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat(config): queue configuration loader

One TOML file describes the box: queue root, CPU concurrency, lock
directory and the project checkouts the runner owns."
```

---

### Task 2: GPU identity

The lock key must be the card's UUID. Obtained by shelling out to `nvidia-smi` rather than importing torch, so a JAX or bare-CUDA project can use the same lock.

**Files:**
- Create: `src/gpuq/gpuinfo.py`, `tests/test_gpuinfo.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `NoGpuError(RuntimeError)`
  - `gpu_uuid(index: int = 0) -> str` — UUID of the card at the given *visible* index, honouring `CUDA_VISIBLE_DEVICES`.
  - `foreign_processes(uuid: str) -> List[Tuple[int, str]]` — `(pid, process_name)` for compute processes on that card, excluding this process and its children.
  - `_run_nvidia_smi(args: List[str]) -> str` — seam that tests monkeypatch.

- [ ] **Step 1: Write the failing test**

Create `tests/test_gpuinfo.py`:

```python
import os

import pytest

from gpuq import gpuinfo
from gpuq.gpuinfo import NoGpuError, foreign_processes, gpu_uuid

UUIDS = "GPU-aaaa1111-2222-3333-4444-555566667777\nGPU-bbbb1111-2222-3333-4444-555566667777\n"


def test_gpu_uuid_returns_the_first_card_by_default(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(gpuinfo, "_run_nvidia_smi", lambda args: UUIDS)
    assert gpu_uuid() == "GPU-aaaa1111-2222-3333-4444-555566667777"


def test_gpu_uuid_honours_cuda_visible_devices(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setattr(gpuinfo, "_run_nvidia_smi", lambda args: UUIDS)
    assert gpu_uuid() == "GPU-bbbb1111-2222-3333-4444-555566667777"


def test_gpu_uuid_rejects_an_out_of_range_index(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(gpuinfo, "_run_nvidia_smi", lambda args: UUIDS)
    with pytest.raises(NoGpuError, match="index 5"):
        gpu_uuid(5)


def test_gpu_uuid_raises_when_no_gpu_is_present(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(gpuinfo, "_run_nvidia_smi", lambda args: "")
    with pytest.raises(NoGpuError):
        gpu_uuid()


def test_foreign_processes_excludes_this_process(monkeypatch):
    mine = os.getpid()
    out = f"{mine}, python\n4242, train.py\n"
    monkeypatch.setattr(gpuinfo, "_run_nvidia_smi", lambda args: out)
    assert foreign_processes("GPU-aaaa") == [(4242, "train.py")]


def test_foreign_processes_is_empty_on_an_idle_card(monkeypatch):
    monkeypatch.setattr(gpuinfo, "_run_nvidia_smi", lambda args: "")
    assert foreign_processes("GPU-aaaa") == []


def test_foreign_processes_tolerates_the_no_processes_banner(monkeypatch):
    # nvidia-smi prints this instead of empty output on some driver versions.
    monkeypatch.setattr(
        gpuinfo, "_run_nvidia_smi", lambda args: "No running processes found\n"
    )
    assert foreign_processes("GPU-aaaa") == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_gpuinfo.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gpuq.gpuinfo'`

- [ ] **Step 3: Implement**

Create `src/gpuq/gpuinfo.py`:

```python
"""GPU identity and occupancy, via nvidia-smi.

Deliberately does not import torch. This lock is meant to coordinate every
process that touches the card, including ones built on JAX or raw CUDA, so it
must not require any particular ML stack to be installed.
"""
from __future__ import annotations

import os
import subprocess
from typing import List, Tuple


class NoGpuError(RuntimeError):
    """Raised when no usable GPU can be identified."""


def _run_nvidia_smi(args: List[str]) -> str:
    """Seam for tests. Returns stdout, or "" if nvidia-smi is unusable."""
    try:
        proc = subprocess.run(
            ["nvidia-smi", *args],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""
    return proc.stdout if proc.returncode == 0 else ""


def _visible_uuids() -> List[str]:
    raw = _run_nvidia_smi(["--query-gpu=uuid", "--format=csv,noheader"])
    all_uuids = [line.strip() for line in raw.splitlines() if line.strip()]

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible:
        return all_uuids

    selected: List[str] = []
    for token in visible.split(","):
        token = token.strip()
        if not token:
            continue
        if token.startswith("GPU-"):
            selected.append(token)
        elif token.isdigit() and int(token) < len(all_uuids):
            selected.append(all_uuids[int(token)])
    return selected


def gpu_uuid(index: int = 0) -> str:
    """UUID of the card at `index` among the visible devices.

    Keyed on the UUID rather than the index because two processes with
    different CUDA_VISIBLE_DEVICES both see their card as index 0. An
    index-keyed lock would hand them different locks for the same physical
    card -- isolation that looks correct and is not.
    """
    uuids = _visible_uuids()
    if not uuids:
        raise NoGpuError(
            "No GPU found. nvidia-smi returned no devices, is missing, or failed."
        )
    if index >= len(uuids):
        raise NoGpuError(f"No GPU at index {index}; {len(uuids)} visible.")
    return uuids[index]


def foreign_processes(uuid: str) -> List[Tuple[int, str]]:
    """Compute processes on the card that are not this process.

    Used by preflight to turn accidental contention into a readable failure
    rather than an out-of-memory error some way into a run.
    """
    raw = _run_nvidia_smi([
        f"--id={uuid}",
        "--query-compute-apps=pid,process_name",
        "--format=csv,noheader",
    ])
    mine = os.getpid()
    found: List[Tuple[int, str]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or "," not in line:
            continue
        pid_text, _, name = line.partition(",")
        pid_text = pid_text.strip()
        if not pid_text.isdigit():
            continue
        pid = int(pid_text)
        if pid != mine:
            found.append((pid, name.strip()))
    return found
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_gpuinfo.py -v`
Expected: PASS, all seven.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(gpuinfo): GPU identity and occupancy via nvidia-smi

Keys on the card UUID rather than the index, and avoids importing torch so
non-torch projects can share the same lock."
```

---

### Task 3: The lock

Ported from `wgan-synthetic`'s `src/train/gpu_lock.py`, which is proven. Changes: the claim payload becomes the pinned protocol (`pid`, `owner`, `cmd`, `started_at`), and the key comes from `gpuinfo` rather than torch.

**Files:**
- Create: `src/gpuq/lock.py`, `tests/test_lock.py`

**Interfaces:**
- Consumes: `gpuq.gpuinfo.gpu_uuid`, `gpuq.config.DEFAULT_LOCK_DIR`.
- Produces:
  - `GpuBusyError(RuntimeError)`
  - `lock_path(uuid: str, lock_dir: Path) -> Path` — `<lock_dir>/gpu-<uuid>.lock`
  - `read_claim(path: Path) -> Optional[dict]`
  - `claim(uuid, *, owner: str, cmd: str, lock_dir: Path, timeout_s: float = 0.0, poll_s: float = 5.0)` — context manager yielding the lock `Path`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_lock.py`:

```python
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import pytest

from gpuq.lock import GpuBusyError, claim, lock_path, read_claim

UUID = "GPU-aaaa1111-2222-3333-4444-555566667777"


def test_lock_path_is_named_by_uuid(tmp_path: Path):
    assert lock_path(UUID, tmp_path) == tmp_path / f"gpu-{UUID}.lock"


def test_claim_creates_the_lock_and_writes_the_protocol_fields(tmp_path: Path):
    with claim(UUID, owner="agent-glove", cmd="train.py", lock_dir=tmp_path) as path:
        payload = json.loads(path.read_text())
    assert set(payload) == {"pid", "owner", "cmd", "started_at"}
    assert payload["pid"] == os.getpid()
    assert payload["owner"] == "agent-glove"
    assert payload["cmd"] == "train.py"


def test_lock_is_released_on_exit(tmp_path: Path):
    with claim(UUID, owner="first", cmd="a", lock_dir=tmp_path):
        pass
    with claim(UUID, owner="second", cmd="b", lock_dir=tmp_path) as path:
        assert json.loads(path.read_text())["owner"] == "second"


def test_lock_is_released_when_the_body_raises(tmp_path: Path):
    with pytest.raises(ZeroDivisionError):
        with claim(UUID, owner="first", cmd="a", lock_dir=tmp_path):
            1 / 0
    with claim(UUID, owner="second", cmd="b", lock_dir=tmp_path):
        pass


def test_read_claim_returns_none_when_absent(tmp_path: Path):
    assert read_claim(tmp_path / "nope.lock") is None


def _hold(lock_dir: str, ready, release):
    from gpuq.lock import claim as inner

    with inner(UUID, owner="holder", cmd="held", lock_dir=Path(lock_dir)):
        ready.set()
        release.wait(timeout=30)


def test_a_second_process_cannot_take_a_held_lock(tmp_path: Path):
    ctx = mp.get_context("spawn")
    ready, release = ctx.Event(), ctx.Event()
    proc = ctx.Process(target=_hold, args=(str(tmp_path), ready, release))
    proc.start()
    try:
        assert ready.wait(timeout=30), "holder never acquired"
        t0 = time.monotonic()
        with pytest.raises(GpuBusyError, match="holder"):
            with claim(UUID, owner="me", cmd="x", lock_dir=tmp_path,
                       timeout_s=1.0, poll_s=0.05):
                pass
        assert time.monotonic() - t0 >= 1.0, "did not wait out the timeout"
    finally:
        release.set()
        proc.join(timeout=30)


def test_waiting_process_acquires_once_the_holder_exits(tmp_path: Path):
    ctx = mp.get_context("spawn")
    ready, release = ctx.Event(), ctx.Event()
    proc = ctx.Process(target=_hold, args=(str(tmp_path), ready, release))
    proc.start()
    try:
        assert ready.wait(timeout=30)
        release.set()
        proc.join(timeout=30)
        with claim(UUID, owner="me", cmd="x", lock_dir=tmp_path,
                   timeout_s=10.0, poll_s=0.05) as path:
            assert json.loads(path.read_text())["owner"] == "me"
    finally:
        release.set()
        if proc.is_alive():
            proc.terminate()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_lock.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gpuq.lock'`

- [ ] **Step 3: Implement**

Create `src/gpuq/lock.py`:

```python
"""Advisory exclusive claim on a physical GPU.

flock is advisory and host-local: it coordinates cooperating processes on one
machine, and does nothing across hosts or against a process that declines to
take the lock. That matches the threat -- other agents and other projects on
the same box, all of which can be made to cooperate.

The on-disk format here is a pinned protocol, not an implementation detail.
Independent implementations interoperate only if they agree on the lock path,
the key derivation and the claim payload. Changing any of the three is a
breaking change.
"""
from __future__ import annotations

import fcntl
import json
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

from .config import DEFAULT_LOCK_DIR


class GpuBusyError(RuntimeError):
    """Raised when another process already holds the requested GPU."""


def lock_path(uuid: str, lock_dir: Path = DEFAULT_LOCK_DIR) -> Path:
    return Path(lock_dir) / f"gpu-{uuid}.lock"


def read_claim(path: Path) -> Optional[dict]:
    """The current holder's metadata, or None if unheld or unreadable."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        text = path.read_text().strip()
    except OSError:
        return None
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


@contextmanager
def claim(
    uuid: str,
    *,
    owner: str,
    cmd: str,
    lock_dir: Path = DEFAULT_LOCK_DIR,
    timeout_s: float = 0.0,
    poll_s: float = 5.0,
) -> Iterator[Path]:
    """Hold the card named by `uuid` for the duration of the block."""
    path = lock_path(uuid, lock_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    t0 = time.monotonic()
    deadline = t0 + max(0.0, timeout_s)
    while True:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except OSError:
            if time.monotonic() >= deadline:
                elapsed = time.monotonic() - t0
                handle.seek(0)
                holder = handle.read().strip() or "(holder wrote no metadata)"
                handle.close()
                raise GpuBusyError(
                    f"GPU lock {path} is held by: {holder}. "
                    f"Waited {elapsed:.0f}s."
                ) from None
            time.sleep(poll_s)
    try:
        handle.seek(0)
        handle.truncate()
        handle.write(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "owner": owner,
                    "cmd": cmd,
                    "started_at": datetime.now(timezone.utc).isoformat(),
                }
            )
        )
        handle.flush()
        yield path
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_lock.py -v`
Expected: PASS, all seven. The two multiprocessing tests are the ones that
matter — they are the only proof the lock actually excludes.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(lock): flock-based GPU claim with a pinned protocol

Ported from wgan-synthetic's src/train/gpu_lock.py. The claim payload and
lock path are now a documented protocol so independent installations
interoperate."
```

---

### Task 4: Preflight and the `gpu-claim` CLI

Makes the lock usable by anything, and turns accidental contention into a readable failure.

**Files:**
- Create: `src/gpuq/cli_claim.py`, `tests/test_cli_claim.py`

**Interfaces:**
- Consumes: `gpuq.gpuinfo.gpu_uuid`, `gpuq.gpuinfo.foreign_processes`, `gpuq.lock.claim`.
- Produces:
  - `ForeignProcessError(RuntimeError)`
  - `preflight(uuid: str) -> None` — raises `ForeignProcessError` naming pids and commands when the card is occupied.
  - `main(argv: Optional[List[str]] = None) -> int` — the `gpu-claim` entry point.

- [ ] **Step 1: Write the failing test**

Create `tests/test_cli_claim.py`:

```python
import sys
from pathlib import Path

import pytest

from gpuq import cli_claim
from gpuq.cli_claim import ForeignProcessError, main, preflight

UUID = "GPU-aaaa1111-2222-3333-4444-555566667777"


def test_preflight_passes_on_an_idle_card(monkeypatch):
    monkeypatch.setattr(cli_claim, "foreign_processes", lambda uuid: [])
    preflight(UUID)  # does not raise


def test_preflight_names_the_occupying_process(monkeypatch):
    monkeypatch.setattr(
        cli_claim, "foreign_processes", lambda uuid: [(4242, "train.py")]
    )
    with pytest.raises(ForeignProcessError, match="4242"):
        preflight(UUID)
    with pytest.raises(ForeignProcessError, match="train.py"):
        preflight(UUID)


def test_main_runs_the_command_under_the_lock(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(cli_claim, "gpu_uuid", lambda index=0: UUID)
    monkeypatch.setattr(cli_claim, "foreign_processes", lambda uuid: [])
    marker = tmp_path / "ran"
    code = main([
        "--lock-dir", str(tmp_path), "--owner", "test", "--",
        sys.executable, "-c", f"open({str(marker)!r}, 'w').write('ok')",
    ])
    assert code == 0
    assert marker.read_text() == "ok"


def test_main_returns_the_child_exit_code(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(cli_claim, "gpu_uuid", lambda index=0: UUID)
    monkeypatch.setattr(cli_claim, "foreign_processes", lambda uuid: [])
    code = main([
        "--lock-dir", str(tmp_path), "--owner", "test", "--",
        sys.executable, "-c", "raise SystemExit(3)",
    ])
    assert code == 3


def test_main_refuses_when_the_card_is_occupied(monkeypatch, tmp_path: Path, capsys):
    monkeypatch.setattr(cli_claim, "gpu_uuid", lambda index=0: UUID)
    monkeypatch.setattr(
        cli_claim, "foreign_processes", lambda uuid: [(4242, "train.py")]
    )
    code = main([
        "--lock-dir", str(tmp_path), "--owner", "test", "--",
        sys.executable, "-c", "pass",
    ])
    assert code == 2
    assert "4242" in capsys.readouterr().err


def test_skip_preflight_allows_an_occupied_card(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(cli_claim, "gpu_uuid", lambda index=0: UUID)
    monkeypatch.setattr(
        cli_claim, "foreign_processes", lambda uuid: [(4242, "train.py")]
    )
    code = main([
        "--lock-dir", str(tmp_path), "--owner", "test", "--skip-preflight", "--",
        sys.executable, "-c", "pass",
    ])
    assert code == 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_cli_claim.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gpuq.cli_claim'`

- [ ] **Step 3: Implement**

Create `src/gpuq/cli_claim.py`:

```python
"""`gpu-claim` -- run a command holding the GPU lock.

    gpu-claim -- python -m my_project.train --config foo.yaml

Preflight refuses to start when the card already carries foreign compute
processes. It cannot stop a process that declines to take the lock; what it
buys is that accidental contention fails immediately, with the offending pid
named, instead of surfacing as an out-of-memory error half an hour in.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from .config import DEFAULT_LOCK_DIR
from .gpuinfo import NoGpuError, foreign_processes, gpu_uuid
from .lock import GpuBusyError, claim


class ForeignProcessError(RuntimeError):
    """Raised when the card is already occupied by an unrelated process."""


def preflight(uuid: str) -> None:
    occupants = foreign_processes(uuid)
    if not occupants:
        return
    listed = ", ".join(f"pid {pid} ({name})" for pid, name in occupants)
    raise ForeignProcessError(
        f"GPU {uuid} already has compute processes: {listed}. "
        "They did not take the lock. Stop them, or pass --skip-preflight "
        "if sharing the card is intended."
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="gpu-claim", description=__doc__)
    parser.add_argument("--lock-dir", type=Path, default=DEFAULT_LOCK_DIR)
    parser.add_argument("--owner", default="gpu-claim")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=0.0,
                        help="Seconds to wait for a held lock. 0 fails fast.")
    parser.add_argument("--poll", type=float, default=5.0)
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("cmd", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd
    if not cmd:
        parser.error("no command given; use: gpu-claim [options] -- CMD ...")

    try:
        uuid = gpu_uuid(args.index)
    except NoGpuError as exc:
        print(f"gpu-claim: {exc}", file=sys.stderr)
        return 2

    if not args.skip_preflight:
        try:
            preflight(uuid)
        except ForeignProcessError as exc:
            print(f"gpu-claim: {exc}", file=sys.stderr)
            return 2

    try:
        with claim(
            uuid,
            owner=args.owner,
            cmd=" ".join(cmd),
            lock_dir=args.lock_dir,
            timeout_s=args.timeout,
            poll_s=args.poll,
        ):
            return subprocess.run(cmd).returncode
    except GpuBusyError as exc:
        print(f"gpu-claim: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_cli_claim.py -v`
Expected: PASS, all six.

- [ ] **Step 5: Verify the entry point end to end**

```bash
pip install -e .
gpu-claim --help
```

Expected: usage text. On a machine with no GPU, `gpu-claim -- echo hi` exits 2
with "No GPU found", which is correct behaviour rather than a failure.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat(cli): gpu-claim wraps any command in the GPU lock

Preflight refuses to start on an occupied card, naming the pid, so
accidental contention fails fast instead of OOMing mid-run."
```

---

### Task 5: Job specs and the queue store

The queue's whole state model. Everything after this is a consumer of it.

**Files:**
- Create: `src/gpuq/spec.py`, `src/gpuq/store.py`, `tests/test_store.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `JobSpec` — dataclass with `id, lane, project, commit, branch, cmd, artifacts, timeout_s, attempts, dedupe_key, pid, error`. `to_dict()` / `from_dict(payload)`.
  - `STATES = ("pending", "running", "done", "failed")`
  - `Store(root: Path)` with: `init()`, `submit(spec) -> str`, `read(job_id) -> Tuple[str, JobSpec]`, `list(state=None) -> List[JobSpec]`, `move(spec, to_state)`, `update(spec)`, `log_paths(job_id) -> Tuple[Path, Path]`, `find_by_dedupe(key) -> Optional[JobSpec]`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_store.py`:

```python
from pathlib import Path

import pytest

from gpuq.spec import JobSpec
from gpuq.store import STATES, Store


def make_spec(job_id="j1", lane="gpu", dedupe_key="k1") -> JobSpec:
    return JobSpec(
        id=job_id,
        lane=lane,
        project="demo",
        commit="a1b2c3d",
        branch="main",
        cmd=["echo", "hi"],
        artifacts=["out/summary.json"],
        timeout_s=60,
        dedupe_key=dedupe_key,
    )


@pytest.fixture
def store(tmp_path: Path) -> Store:
    s = Store(tmp_path / "queue")
    s.init()
    return s


def test_init_creates_every_state_directory(store: Store):
    for state in STATES:
        assert (store.root / state).is_dir()
    assert (store.root / "logs").is_dir()


def test_submitted_job_lands_in_pending(store: Store):
    job_id = store.submit(make_spec())
    assert job_id == "j1"
    assert [s.id for s in store.list("pending")] == ["j1"]


def test_spec_round_trips_through_disk(store: Store):
    store.submit(make_spec())
    state, spec = store.read("j1")
    assert state == "pending"
    assert spec.cmd == ["echo", "hi"]
    assert spec.artifacts == ["out/summary.json"]
    assert spec.timeout_s == 60
    assert spec.attempts == 0


def test_move_is_a_state_transition(store: Store):
    store.submit(make_spec())
    _, spec = store.read("j1")
    store.move(spec, "running")
    assert store.list("pending") == []
    assert [s.id for s in store.list("running")] == ["j1"]
    assert store.read("j1")[0] == "running"


def test_move_rejects_an_unknown_state(store: Store):
    store.submit(make_spec())
    _, spec = store.read("j1")
    with pytest.raises(ValueError, match="nonsense"):
        store.move(spec, "nonsense")


def test_update_rewrites_in_place_without_moving(store: Store):
    store.submit(make_spec())
    _, spec = store.read("j1")
    spec.attempts = 2
    store.update(spec)
    state, reread = store.read("j1")
    assert state == "pending"
    assert reread.attempts == 2


def test_list_without_a_state_returns_everything(store: Store):
    store.submit(make_spec("a"))
    store.submit(make_spec("b", dedupe_key="k2"))
    _, spec = store.read("b")
    store.move(spec, "done")
    assert {s.id for s in store.list()} == {"a", "b"}


def test_duplicate_dedupe_key_returns_the_existing_id(store: Store):
    store.submit(make_spec("first", dedupe_key="same"))
    assert store.submit(make_spec("second", dedupe_key="same")) == "first"
    assert [s.id for s in store.list("pending")] == ["first"]


def test_dedupe_ignores_finished_jobs(store: Store):
    store.submit(make_spec("first", dedupe_key="same"))
    _, spec = store.read("first")
    store.move(spec, "done")
    assert store.submit(make_spec("second", dedupe_key="same")) == "second"


def test_reading_an_unknown_job_raises(store: Store):
    with pytest.raises(KeyError, match="ghost"):
        store.read("ghost")


def test_log_paths_are_under_logs(store: Store):
    out, err = store.log_paths("j1")
    assert out == store.root / "logs" / "j1.out"
    assert err == store.root / "logs" / "j1.err"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gpuq.spec'`

- [ ] **Step 3: Implement the job spec**

Create `src/gpuq/spec.py`:

```python
"""The unit of work.

`commit` is pinned rather than only `branch` because a queued job may wait
hours. If the runner resolved the branch at execution time, the tree could
move underneath it and produce a result nobody can reproduce.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import List, Optional

LANES = ("cpu", "gpu")


@dataclass
class JobSpec:
    id: str
    lane: str
    project: str
    commit: str
    branch: str
    cmd: List[str]
    artifacts: List[str] = field(default_factory=list)
    timeout_s: int = 3600
    attempts: int = 0
    dedupe_key: str = ""
    pid: Optional[int] = None
    error: Optional[str] = None

    def __post_init__(self) -> None:
        if self.lane not in LANES:
            raise ValueError(f"Unknown lane {self.lane!r}; expected one of {LANES}")
        if not self.cmd:
            raise ValueError(f"Job {self.id} has an empty cmd")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "JobSpec":
        return cls(**payload)
```

- [ ] **Step 4: Implement the store**

Create `src/gpuq/store.py`:

```python
"""The queue: a directory tree whose state transitions are renames.

State lives in the filesystem rather than a database so that the whole system
is legible to `ls` and repairable with `mv`. A queue that needs a running
service to inspect becomes opaque exactly when something has gone wrong --
which is when you most need to see it.

`os.rename` within a filesystem is atomic, so a job is in exactly one state at
any instant and two runner threads cannot both claim it.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

from .spec import JobSpec

STATES = ("pending", "running", "done", "failed")
ACTIVE_STATES = ("pending", "running")


class Store:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def init(self) -> None:
        for state in STATES:
            (self.root / state).mkdir(parents=True, exist_ok=True)
        (self.root / "logs").mkdir(parents=True, exist_ok=True)

    def _path(self, state: str, job_id: str) -> Path:
        return self.root / state / f"{job_id}.json"

    def _write(self, path: Path, spec: JobSpec) -> None:
        """Write via a temp file and rename, so a reader never sees half a spec."""
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(spec.to_dict(), indent=2))
        os.rename(tmp, path)

    def submit(self, spec: JobSpec) -> str:
        existing = self.find_by_dedupe(spec.dedupe_key)
        if existing is not None:
            return existing.id
        self.init()
        self._write(self._path("pending", spec.id), spec)
        return spec.id

    def read(self, job_id: str) -> Tuple[str, JobSpec]:
        for state in STATES:
            path = self._path(state, job_id)
            if path.exists():
                return state, JobSpec.from_dict(json.loads(path.read_text()))
        raise KeyError(f"No job {job_id!r} in {self.root}")

    def list(self, state: Optional[str] = None) -> List[JobSpec]:
        states = (state,) if state else STATES
        found: List[JobSpec] = []
        for st in states:
            directory = self.root / st
            if not directory.is_dir():
                continue
            for path in sorted(directory.glob("*.json")):
                found.append(JobSpec.from_dict(json.loads(path.read_text())))
        return found

    def move(self, spec: JobSpec, to_state: str) -> None:
        if to_state not in STATES:
            raise ValueError(f"Unknown state {to_state!r}; expected one of {STATES}")
        current, _ = self.read(spec.id)
        target = self._path(to_state, spec.id)
        self._write(self._path(current, spec.id), spec)
        os.rename(self._path(current, spec.id), target)

    def update(self, spec: JobSpec) -> None:
        state, _ = self.read(spec.id)
        self._write(self._path(state, spec.id), spec)

    def find_by_dedupe(self, key: str) -> Optional[JobSpec]:
        """A pending or running job with this key. Finished jobs do not block."""
        if not key:
            return None
        for state in ACTIVE_STATES:
            for spec in self.list(state):
                if spec.dedupe_key == key:
                    return spec
        return None

    def log_paths(self, job_id: str) -> Tuple[Path, Path]:
        logs = self.root / "logs"
        return logs / f"{job_id}.out", logs / f"{job_id}.err"
```

- [ ] **Step 5: Run the tests**

Run: `python -m pytest tests/test_store.py -v`
Expected: PASS, all eleven.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat(store): directory-backed job queue with atomic transitions

States are subdirectories and transitions are renames, so the queue is
inspectable with ls and repairable with mv. Dedupe covers pending and
running jobs only."
```

---

### Task 6: The `gpuq` CLI

What producers actually touch. Keep it thin — all logic lives in `Store`.

**Files:**
- Create: `src/gpuq/cli.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: `gpuq.store.Store`, `gpuq.spec.JobSpec`, `gpuq.config.load_config`.
- Produces: `main(argv: Optional[List[str]] = None) -> int` with subcommands
  `submit`, `list`, `show`, `logs`, `cancel`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_cli.py`:

```python
import json
from pathlib import Path

import pytest

from gpuq.cli import main
from gpuq.spec import JobSpec
from gpuq.store import Store


@pytest.fixture
def root(tmp_path: Path) -> Path:
    Store(tmp_path / "queue").init()
    return tmp_path / "queue"


def submit(root: Path, *extra: str) -> int:
    return main([
        "--queue-root", str(root), "submit",
        "--id", "j1", "--lane", "gpu", "--project", "demo",
        "--commit", "a1b2c3d", "--branch", "main",
        *extra, "--", "echo", "hi",
    ])


def test_submit_creates_a_pending_job(root: Path, capsys):
    assert submit(root) == 0
    assert capsys.readouterr().out.strip() == "j1"
    assert [s.id for s in Store(root).list("pending")] == ["j1"]


def test_submit_records_artifacts_and_timeout(root: Path):
    submit(root, "--artifact", "out/a.json", "--artifact", "out/b.yaml",
           "--timeout", "120")
    _, spec = Store(root).read("j1")
    assert spec.artifacts == ["out/a.json", "out/b.yaml"]
    assert spec.timeout_s == 120


def test_submit_is_idempotent_on_dedupe_key(root: Path, capsys):
    submit(root, "--dedupe-key", "same")
    capsys.readouterr()
    main([
        "--queue-root", str(root), "submit",
        "--id", "j2", "--lane", "gpu", "--project", "demo",
        "--commit", "a1b2c3d", "--branch", "main",
        "--dedupe-key", "same", "--", "echo", "hi",
    ])
    assert capsys.readouterr().out.strip() == "j1"
    assert len(Store(root).list("pending")) == 1


def test_list_shows_state_and_id(root: Path, capsys):
    submit(root)
    assert main(["--queue-root", str(root), "list"]) == 0
    out = capsys.readouterr().out
    assert "j1" in out and "pending" in out


def test_show_emits_the_spec_as_json(root: Path, capsys):
    submit(root)
    assert main(["--queue-root", str(root), "show", "j1"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["id"] == "j1"
    assert payload["cmd"] == ["echo", "hi"]


def test_show_on_an_unknown_job_fails_cleanly(root: Path, capsys):
    assert main(["--queue-root", str(root), "show", "ghost"]) == 1
    assert "ghost" in capsys.readouterr().err


def test_cancel_moves_a_pending_job_to_failed(root: Path):
    submit(root)
    assert main(["--queue-root", str(root), "cancel", "j1"]) == 0
    state, spec = Store(root).read("j1")
    assert state == "failed"
    assert "cancelled" in spec.error


def test_logs_reports_absence_without_crashing(root: Path, capsys):
    submit(root)
    assert main(["--queue-root", str(root), "logs", "j1"]) == 0
    assert "no output" in capsys.readouterr().out.lower()


def test_logs_prints_captured_output(root: Path, capsys):
    submit(root)
    out, err = Store(root).log_paths("j1")
    out.write_text("hello from stdout\n")
    err.write_text("and stderr\n")
    main(["--queue-root", str(root), "logs", "j1"])
    printed = capsys.readouterr().out
    assert "hello from stdout" in printed
    assert "and stderr" in printed
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gpuq.cli'`

- [ ] **Step 3: Implement**

Create `src/gpuq/cli.py`:

```python
"""`gpuq` -- submit and inspect queued jobs.

Deliberately thin. Every decision lives in Store; this module parses
arguments and prints. Agents are the main consumer, so output is stable and
line-oriented, and `show` emits plain JSON.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from .config import DEFAULT_CONFIG_PATH, load_config
from .spec import JobSpec
from .store import Store


def _resolve_root(args: argparse.Namespace) -> Path:
    if args.queue_root:
        return Path(args.queue_root)
    return load_config(Path(args.config)).queue_root


def _cmd_submit(args: argparse.Namespace, store: Store) -> int:
    cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd
    if not cmd:
        print("gpuq: no command given; use -- CMD ...", file=sys.stderr)
        return 1
    spec = JobSpec(
        id=args.id,
        lane=args.lane,
        project=args.project,
        commit=args.commit,
        branch=args.branch,
        cmd=cmd,
        artifacts=args.artifact or [],
        timeout_s=args.timeout,
        dedupe_key=args.dedupe_key or f"{args.project}:{args.id}:{args.commit}",
    )
    print(store.submit(spec))
    return 0


def _cmd_list(args: argparse.Namespace, store: Store) -> int:
    for state in (args.state,) if args.state else ("running", "pending", "failed", "done"):
        for spec in store.list(state):
            print(f"{state:8} {spec.lane:3} {spec.id:30} {spec.project}")
    return 0


def _cmd_show(args: argparse.Namespace, store: Store) -> int:
    try:
        state, spec = store.read(args.id)
    except KeyError as exc:
        print(f"gpuq: {exc}", file=sys.stderr)
        return 1
    payload = spec.to_dict()
    payload["state"] = state
    print(json.dumps(payload, indent=2))
    return 0


def _cmd_logs(args: argparse.Namespace, store: Store) -> int:
    out, err = store.log_paths(args.id)
    printed = False
    for label, path in (("stdout", out), ("stderr", err)):
        if path.exists() and path.stat().st_size:
            print(f"--- {label} ---")
            print(path.read_text(), end="")
            printed = True
    if not printed:
        print(f"gpuq: no output recorded for {args.id}")
    return 0


def _cmd_cancel(args: argparse.Namespace, store: Store) -> int:
    try:
        state, spec = store.read(args.id)
    except KeyError as exc:
        print(f"gpuq: {exc}", file=sys.stderr)
        return 1
    if state in ("done", "failed"):
        print(f"gpuq: {args.id} already {state}", file=sys.stderr)
        return 1
    spec.error = f"cancelled from {state}"
    store.move(spec, "failed")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="gpuq", description=__doc__)
    parser.add_argument("--queue-root", default=None,
                        help="Queue directory. Overrides the config file.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("submit")
    p.add_argument("--id", required=True)
    p.add_argument("--lane", choices=("cpu", "gpu"), required=True)
    p.add_argument("--project", required=True)
    p.add_argument("--commit", required=True)
    p.add_argument("--branch", required=True)
    p.add_argument("--artifact", action="append", default=[])
    p.add_argument("--timeout", type=int, default=3600)
    p.add_argument("--dedupe-key", default=None)
    p.add_argument("cmd", nargs=argparse.REMAINDER)
    p.set_defaults(func=_cmd_submit)

    p = sub.add_parser("list")
    p.add_argument("--state", choices=("pending", "running", "done", "failed"))
    p.set_defaults(func=_cmd_list)

    p = sub.add_parser("show")
    p.add_argument("id")
    p.set_defaults(func=_cmd_show)

    p = sub.add_parser("logs")
    p.add_argument("id")
    p.set_defaults(func=_cmd_logs)

    p = sub.add_parser("cancel")
    p.add_argument("id")
    p.set_defaults(func=_cmd_cancel)

    args = parser.parse_args(argv)
    store = Store(_resolve_root(args))
    store.init()
    return args.func(args, store)


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_cli.py -v`
Expected: PASS, all nine.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(cli): gpuq submit, list, show, logs and cancel

Thin over Store. Output is line-oriented and show emits JSON, since
agents are the primary consumer."
```

---

### Task 7: Git operations

Runner-only. Isolated in its own module so the "workers never run git" rule is visible in the import graph rather than merely written down.

**Files:**
- Create: `src/gpuq/gitops.py`, `tests/test_gitops.py`

**Interfaces:**
- Consumes: `gpuq.config.Project`.
- Produces:
  - `GitError(RuntimeError)`
  - `git(args: List[str], cwd: Path) -> str`
  - `ensure_checkout(project: Project) -> Path`
  - `checkout_commit(project: Project, commit: str) -> None`
  - `commit_artifacts(project, branch, paths, message) -> Optional[str]` — returns the new commit sha, or None when nothing changed.

- [ ] **Step 1: Write the failing test**

Create `tests/test_gitops.py`:

```python
import subprocess
from pathlib import Path

import pytest

from gpuq.config import Project
from gpuq.gitops import GitError, checkout_commit, commit_artifacts, ensure_checkout, git


def _init_origin(path: Path) -> str:
    path.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)
    (path / "README.md").write_text("origin\n")
    subprocess.run(["git", "add", "-A"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=path, check=True)
    # Pushing to a non-bare repo's checked-out branch is refused by default.
    subprocess.run(["git", "config", "receive.denyCurrentBranch", "ignore"],
                   cwd=path, check=True)
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=path,
                          capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture
def origin(tmp_path: Path):
    sha = _init_origin(tmp_path / "origin")
    return tmp_path / "origin", sha


def _project(tmp_path: Path, origin_path: Path) -> Project:
    return Project(
        name="demo",
        remote=str(origin_path),
        checkout=tmp_path / "checkout",
        commit_artifacts=True,
    )


def test_ensure_checkout_clones_when_absent(tmp_path: Path, origin):
    origin_path, _ = origin
    project = _project(tmp_path, origin_path)
    path = ensure_checkout(project)
    assert (path / "README.md").read_text() == "origin\n"


def test_ensure_checkout_is_idempotent(tmp_path: Path, origin):
    origin_path, _ = origin
    project = _project(tmp_path, origin_path)
    ensure_checkout(project)
    assert ensure_checkout(project) == project.checkout


def test_checkout_commit_pins_the_tree(tmp_path: Path, origin):
    origin_path, first = origin
    project = _project(tmp_path, origin_path)
    ensure_checkout(project)
    (origin_path / "README.md").write_text("moved on\n")
    subprocess.run(["git", "commit", "-qam", "second"], cwd=origin_path, check=True)

    checkout_commit(project, first)
    assert (project.checkout / "README.md").read_text() == "origin\n"


def test_checkout_commit_rejects_an_unknown_sha(tmp_path: Path, origin):
    origin_path, _ = origin
    project = _project(tmp_path, origin_path)
    ensure_checkout(project)
    with pytest.raises(GitError):
        checkout_commit(project, "0" * 40)


def test_commit_artifacts_commits_only_named_paths(tmp_path: Path, origin):
    origin_path, _ = origin
    project = _project(tmp_path, origin_path)
    ensure_checkout(project)
    (project.checkout / "out").mkdir()
    (project.checkout / "out" / "summary.json").write_text("{}\n")
    (project.checkout / "untracked.txt").write_text("should not be committed\n")

    sha = commit_artifacts(project, "main", ["out/summary.json"], "add summary")
    assert sha is not None
    listed = git(["show", "--name-only", "--format=", sha], project.checkout)
    assert "out/summary.json" in listed
    assert "untracked.txt" not in listed


def test_commit_artifacts_returns_none_when_nothing_changed(tmp_path: Path, origin):
    origin_path, _ = origin
    project = _project(tmp_path, origin_path)
    ensure_checkout(project)
    assert commit_artifacts(project, "main", ["README.md"], "no change") is None


def test_commit_artifacts_skips_missing_paths(tmp_path: Path, origin):
    origin_path, _ = origin
    project = _project(tmp_path, origin_path)
    ensure_checkout(project)
    assert commit_artifacts(project, "main", ["out/never_written.json"], "m") is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_gitops.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gpuq.gitops'`

- [ ] **Step 3: Implement**

Create `src/gpuq/gitops.py`:

```python
"""Git operations. Runner main-loop only.

Concurrent CPU jobs committing into one checkout would corrupt the index.
Keeping every git call in this module, imported only by the runner's main
loop, makes that rule visible in the import graph instead of relying on
everyone remembering it.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List, Optional

from .config import Project


class GitError(RuntimeError):
    """A git invocation failed."""


def git(args: List[str], cwd: Path) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, text=True
    )
    if proc.returncode != 0:
        raise GitError(
            f"git {' '.join(args)} failed in {cwd}: {proc.stderr.strip()}"
        )
    return proc.stdout.strip()


def ensure_checkout(project: Project) -> Path:
    """Clone the project if its checkout is missing. Idempotent."""
    checkout = Path(project.checkout)
    if (checkout / ".git").is_dir():
        return checkout
    checkout.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["git", "clone", project.remote, str(checkout)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise GitError(f"clone {project.remote} failed: {proc.stderr.strip()}")
    return checkout


def checkout_commit(project: Project, commit: str) -> None:
    """Put the checkout on exactly `commit`, fetching if it is not present yet."""
    checkout = ensure_checkout(project)
    try:
        git(["checkout", "-q", "--detach", commit], checkout)
    except GitError:
        git(["fetch", "--all", "-q"], checkout)
        git(["checkout", "-q", "--detach", commit], checkout)


def commit_artifacts(
    project: Project,
    branch: str,
    paths: List[str],
    message: str,
) -> Optional[str]:
    """Commit the named paths onto `branch` and push. None if nothing changed.

    Only the listed paths are staged. A run leaves all sorts of debris in the
    working tree -- checkpoints, caches, partial outputs -- and `git add -A`
    would sweep it into the history.
    """
    checkout = Path(project.checkout)
    existing = [p for p in paths if (checkout / p).exists()]
    if not existing:
        return None

    git(["fetch", "origin", "-q"], checkout)
    try:
        git(["checkout", "-q", "-B", branch, f"origin/{branch}"], checkout)
    except GitError:
        git(["checkout", "-q", "-B", branch], checkout)

    git(["add", "--", *existing], checkout)
    status = git(["status", "--porcelain", "--", *existing], checkout)
    if not status:
        return None

    git(["commit", "-q", "-m", message], checkout)
    sha = git(["rev-parse", "HEAD"], checkout)
    git(["push", "-q", "origin", branch], checkout)
    return sha
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_gitops.py -v`
Expected: PASS, all seven. `commit_artifacts` pushes to a local-path "origin",
which the fixture allows by setting `receive.denyCurrentBranch=ignore` — git
otherwise refuses a push to the branch a non-bare repo has checked out.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(gitops): checkout at a pinned commit, commit named artifacts

Stages only the paths a job declares, so run debris never enters history.
Isolated in one module because only the runner main loop may call git."
```

---

### Task 8: The reaper

**Files:**
- Create: `src/gpuq/reaper.py`, `tests/test_reaper.py`

**Interfaces:**
- Consumes: `gpuq.store.Store`, `gpuq.spec.JobSpec`, `gpuq.lock.read_claim`, `gpuq.lock.lock_path`.
- Produces:
  - `MAX_ATTEMPTS = 2`
  - `pid_alive(pid: int) -> bool`
  - `reap_jobs(store: Store) -> List[str]` — requeue or fail abandoned `running/` jobs; returns the ids it touched.
  - `stale_claim(uuid: str, lock_dir: Path) -> Optional[dict]` — the claim payload if its pid is dead, else None.

- [ ] **Step 1: Write the failing test**

Create `tests/test_reaper.py`:

```python
import json
import os
from pathlib import Path

import pytest

from gpuq.reaper import MAX_ATTEMPTS, pid_alive, reap_jobs, stale_claim
from gpuq.spec import JobSpec
from gpuq.store import Store


@pytest.fixture
def store(tmp_path: Path) -> Store:
    s = Store(tmp_path / "queue")
    s.init()
    return s


def running(store: Store, job_id: str, pid: int, attempts: int = 0) -> JobSpec:
    spec = JobSpec(id=job_id, lane="gpu", project="demo", commit="a1",
                   branch="main", cmd=["echo", "hi"], attempts=attempts, pid=pid)
    store.submit(spec)
    store.move(spec, "running")
    return spec


def test_pid_alive_is_true_for_this_process():
    assert pid_alive(os.getpid()) is True


def test_pid_alive_is_false_for_an_impossible_pid():
    assert pid_alive(999_999_999) is False


def test_a_live_job_is_left_alone(store: Store):
    running(store, "alive", os.getpid())
    assert reap_jobs(store) == []
    assert [s.id for s in store.list("running")] == ["alive"]


def test_an_abandoned_job_is_requeued_once(store: Store):
    running(store, "dead", 999_999_999)
    assert reap_jobs(store) == ["dead"]
    assert store.list("running") == []
    state, spec = store.read("dead")
    assert state == "pending"
    assert spec.attempts == 1


def test_a_repeatedly_abandoned_job_fails(store: Store):
    running(store, "cursed", 999_999_999, attempts=MAX_ATTEMPTS)
    assert reap_jobs(store) == ["cursed"]
    state, spec = store.read("cursed")
    assert state == "failed"
    assert "abandoned" in spec.error


def test_a_running_job_without_a_pid_is_requeued(store: Store):
    spec = JobSpec(id="nopid", lane="gpu", project="demo", commit="a1",
                   branch="main", cmd=["echo", "hi"])
    store.submit(spec)
    store.move(spec, "running")
    assert reap_jobs(store) == ["nopid"]
    assert store.read("nopid")[0] == "pending"


def test_stale_claim_returns_none_when_the_holder_lives(tmp_path: Path):
    uuid = "GPU-aaaa"
    path = tmp_path / f"gpu-{uuid}.lock"
    path.write_text(json.dumps({"pid": os.getpid(), "owner": "me",
                                "cmd": "x", "started_at": "now"}))
    assert stale_claim(uuid, tmp_path) is None


def test_stale_claim_reports_a_dead_holder(tmp_path: Path):
    uuid = "GPU-aaaa"
    path = tmp_path / f"gpu-{uuid}.lock"
    path.write_text(json.dumps({"pid": 999_999_999, "owner": "ghost",
                                "cmd": "x", "started_at": "now"}))
    assert stale_claim(uuid, tmp_path)["owner"] == "ghost"


def test_stale_claim_returns_none_when_unheld(tmp_path: Path):
    assert stale_claim("GPU-aaaa", tmp_path) is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_reaper.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gpuq.reaper'`

- [ ] **Step 3: Implement**

Create `src/gpuq/reaper.py`:

```python
"""Recover from processes that died without cleaning up.

This lives in the runner rather than in a supervising agent because it has to
run when nothing else is alive -- which is exactly when a leaked job needs
reaping.

The attempt cap is load-bearing. Without it, a job that crashes the runner on
every start occupies the only GPU forever.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from .config import DEFAULT_LOCK_DIR
from .lock import lock_path, read_claim
from .store import Store

MAX_ATTEMPTS = 2


def pid_alive(pid: int) -> bool:
    """True if a process with this pid exists.

    Signal 0 performs the permission and existence checks without delivering
    anything. EPERM means it exists but belongs to someone else, which still
    counts as alive.
    """
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def reap_jobs(store: Store) -> List[str]:
    """Requeue or fail running jobs whose process is gone."""
    touched: List[str] = []
    for spec in store.list("running"):
        if spec.pid is not None and pid_alive(spec.pid):
            continue
        touched.append(spec.id)
        spec.pid = None
        if spec.attempts >= MAX_ATTEMPTS:
            spec.error = (
                f"abandoned {spec.attempts} times without completing; "
                "not retried again"
            )
            store.move(spec, "failed")
        else:
            spec.attempts += 1
            store.move(spec, "pending")
    return touched


def stale_claim(uuid: str, lock_dir: Path = DEFAULT_LOCK_DIR) -> Optional[dict]:
    """The claim payload if it names a dead pid, else None.

    Note that flock itself is released by the kernel when the holder dies, so
    a stale payload is a reporting problem rather than a stuck lock. It is
    still worth surfacing: it is the trace of a crash.
    """
    claim = read_claim(lock_path(uuid, lock_dir))
    if claim is None:
        return None
    pid = claim.get("pid")
    if isinstance(pid, int) and not pid_alive(pid):
        return claim
    return None
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_reaper.py -v`
Expected: PASS, all nine.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(reaper): recover abandoned jobs and report stale claims

Requeues a job whose process vanished, once, then fails it. The attempt
cap is what stops a crash-looping job owning the card forever."
```

---

### Task 9: The runner

The daemon. Everything above is a component of it.

**Files:**
- Create: `src/gpuq/runner.py`, `tests/test_runner.py`
- Modify: `pyproject.toml` — add the `gpuq-runner` console script

**Interfaces:**
- Consumes: `Config`, `Store`, `JobSpec`, `reap_jobs`, `gitops`, `lock.claim`, `gpuinfo.gpu_uuid`.
- Produces:
  - `Runner(config: Config)` with `tick() -> None` (one admission-and-collection pass) and `run_forever(poll_s: float = 5.0) -> None`.
  - `Runner.tick` is the unit under test; `run_forever` is a loop around it.
  - `main(argv: Optional[List[str]] = None) -> int`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_runner.py`:

```python
import subprocess
import sys
import time
from pathlib import Path

import pytest

from gpuq.config import Config, Project
from gpuq.runner import Runner
from gpuq.spec import JobSpec
from gpuq.store import Store


@pytest.fixture
def env(tmp_path: Path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=checkout, check=True)
    subprocess.run(["git", "config", "user.email", "t@e.com"], cwd=checkout, check=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=checkout, check=True)
    (checkout / "README.md").write_text("x\n")
    subprocess.run(["git", "add", "-A"], cwd=checkout, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=checkout, check=True)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=checkout,
                         capture_output=True, text=True, check=True).stdout.strip()

    config = Config(
        queue_root=tmp_path / "queue",
        cpu_slots=2,
        lock_dir=tmp_path / "locks",
        projects={"demo": Project(name="demo", remote=str(checkout),
                                  checkout=checkout, commit_artifacts=False)},
    )
    store = Store(config.queue_root)
    store.init()
    return config, store, sha


def cpu_job(job_id: str, sha: str, script: str, timeout_s: int = 60) -> JobSpec:
    return JobSpec(id=job_id, lane="cpu", project="demo", commit=sha,
                   branch="main", cmd=[sys.executable, "-c", script],
                   timeout_s=timeout_s, dedupe_key=job_id)


def drain(runner: Runner, store: Store, limit: float = 30.0) -> None:
    deadline = time.monotonic() + limit
    while time.monotonic() < deadline:
        runner.tick()
        if not store.list("pending") and not store.list("running"):
            return
        time.sleep(0.05)
    raise AssertionError("runner did not drain the queue in time")


def test_a_successful_job_reaches_done(env):
    config, store, sha = env
    store.submit(cpu_job("ok", sha, "print('hi')"))
    drain(Runner(config), store)
    assert [s.id for s in store.list("done")] == ["ok"]


def test_stdout_is_captured(env):
    config, store, sha = env
    store.submit(cpu_job("noisy", sha, "print('hello world')"))
    drain(Runner(config), store)
    out, _ = store.log_paths("noisy")
    assert "hello world" in out.read_text()


def test_a_failing_job_reaches_failed_with_its_stderr(env):
    config, store, sha = env
    store.submit(cpu_job("bad", sha, "import sys; sys.stderr.write('boom'); sys.exit(1)"))
    drain(Runner(config), store)
    state, spec = store.read("bad")
    assert state == "failed"
    assert "boom" in spec.error


def test_a_job_past_its_timeout_is_killed_and_failed(env):
    config, store, sha = env
    store.submit(cpu_job("hang", sha, "import time; time.sleep(300)", timeout_s=1))
    drain(Runner(config), store, limit=60.0)
    state, spec = store.read("hang")
    assert state == "failed"
    assert "timeout" in spec.error.lower()


def test_cpu_slots_cap_concurrency(env):
    config, store, sha = env
    for i in range(4):
        store.submit(cpu_job(f"j{i}", sha, "import time; time.sleep(0.5)"))
    runner = Runner(config)
    runner.tick()
    assert len(store.list("running")) <= config.cpu_slots
    drain(runner, store)
    assert len(store.list("done")) == 4


def test_a_running_job_records_its_pid(env):
    config, store, sha = env
    store.submit(cpu_job("pidded", sha, "import time; time.sleep(0.5)"))
    runner = Runner(config)
    runner.tick()
    _, spec = store.read("pidded")
    assert spec.pid is not None and spec.pid > 0
    drain(runner, store)


def test_only_one_gpu_job_runs_at_a_time(env, monkeypatch):
    config, store, sha = env
    import gpuq.runner as runner_mod
    monkeypatch.setattr(runner_mod, "gpu_uuid", lambda index=0: "GPU-test")
    for i in range(3):
        spec = cpu_job(f"g{i}", sha, "import time; time.sleep(0.3)")
        spec.lane = "gpu"
        store.submit(spec)
    runner = Runner(config)
    runner.tick()
    assert len(store.list("running")) == 1
    drain(runner, store)
    assert len(store.list("done")) == 3
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_runner.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gpuq.runner'`

- [ ] **Step 3: Implement**

Create `src/gpuq/runner.py`:

```python
"""The daemon. One per box, managed by supervisor.

It is the sole launcher of queued work and the sole caller of git. CPU jobs
run concurrently up to a cap; GPU jobs run strictly one at a time behind the
lock. Artifacts are committed by this loop and never by a worker, because
concurrent commits into one checkout would corrupt the index.

`tick()` is one non-blocking pass -- admit what the lanes allow, collect what
has finished -- and is what the tests drive. `run_forever` is a sleep around
it.
"""
from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .config import DEFAULT_CONFIG_PATH, Config, load_config
from .gitops import GitError, checkout_commit, commit_artifacts
from .gpuinfo import NoGpuError, gpu_uuid
from .lock import GpuBusyError, claim
from .reaper import reap_jobs
from .spec import JobSpec
from .store import Store


@dataclass
class Running:
    spec: JobSpec
    process: subprocess.Popen
    started_at: float
    out_handle: object
    err_handle: object
    claim_cm: Optional[object] = None


class Runner:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.store = Store(config.queue_root)
        self.store.init()
        self.active: Dict[str, Running] = {}
        self._reaped_once = False

    # -- admission ------------------------------------------------------

    def _lane_load(self, lane: str) -> int:
        return sum(1 for r in self.active.values() if r.spec.lane == lane)

    def _can_admit(self, spec: JobSpec) -> bool:
        if spec.lane == "gpu":
            return self._lane_load("gpu") == 0
        return self._lane_load("cpu") < self.config.cpu_slots

    def _start(self, spec: JobSpec) -> None:
        project = self.config.projects.get(spec.project)
        if project is None:
            spec.error = f"unknown project {spec.project!r}"
            self.store.move(spec, "failed")
            return

        try:
            checkout_commit(project, spec.commit)
        except GitError as exc:
            spec.error = f"checkout failed: {exc}"
            self.store.move(spec, "failed")
            return

        claim_cm = None
        if spec.lane == "gpu":
            try:
                claim_cm = claim(
                    gpu_uuid(),
                    owner=f"gpuq:{spec.id}",
                    cmd=" ".join(spec.cmd),
                    lock_dir=self.config.lock_dir,
                )
                claim_cm.__enter__()
            except (GpuBusyError, NoGpuError) as exc:
                # Leave it pending; the card may free up before the next tick.
                print(f"gpuq-runner: {spec.id} waiting: {exc}", file=sys.stderr)
                return

        out_path, err_path = self.store.log_paths(spec.id)
        out_handle = out_path.open("w")
        err_handle = err_path.open("w")

        env_python = str(project.venv / "bin" / "python") if project.venv else None
        cmd = list(spec.cmd)
        if env_python and cmd[0] == "python":
            cmd[0] = env_python

        process = subprocess.Popen(
            cmd,
            cwd=str(project.checkout),
            stdout=out_handle,
            stderr=err_handle,
            start_new_session=True,
        )
        spec.pid = process.pid
        # move() writes the spec before renaming, so the pid lands with it.
        self.store.move(spec, "running")
        self.active[spec.id] = Running(
            spec=spec,
            process=process,
            started_at=time.monotonic(),
            out_handle=out_handle,
            err_handle=err_handle,
            claim_cm=claim_cm,
        )

    # -- collection -----------------------------------------------------

    def _finish(self, running: Running, *, timed_out: bool) -> None:
        spec = running.spec
        running.out_handle.close()
        running.err_handle.close()
        if running.claim_cm is not None:
            running.claim_cm.__exit__(None, None, None)
        self.active.pop(spec.id, None)

        _, err_path = self.store.log_paths(spec.id)
        tail = ""
        if err_path.exists():
            tail = err_path.read_text()[-2000:].strip()

        spec.pid = None
        if timed_out:
            spec.error = f"timeout after {spec.timeout_s}s; killed. {tail}".strip()
            self.store.move(spec, "failed")
            return
        if running.process.returncode != 0:
            spec.error = f"exit {running.process.returncode}. {tail}".strip()
            self.store.move(spec, "failed")
            return

        self._commit(spec)
        self.store.move(spec, "done")

    def _commit(self, spec: JobSpec) -> None:
        project = self.config.projects.get(spec.project)
        if project is None or not project.commit_artifacts or not spec.artifacts:
            return
        try:
            commit_artifacts(
                project,
                spec.branch,
                spec.artifacts,
                f"chore(runs): artifacts for {spec.id}\n\nJob {spec.id} at {spec.commit}.",
            )
        except GitError as exc:
            # The run succeeded; only publishing failed. Say so, do not fail it.
            print(f"gpuq-runner: {spec.id} artifact commit failed: {exc}",
                  file=sys.stderr)

    def _kill(self, running: Running) -> None:
        try:
            running.process.terminate()
            running.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            running.process.kill()
            running.process.wait(timeout=10)

    # -- the loop -------------------------------------------------------

    def tick(self) -> None:
        if not self._reaped_once:
            reap_jobs(self.store)
            self._reaped_once = True

        for running in list(self.active.values()):
            if running.process.poll() is not None:
                self._finish(running, timed_out=False)
            elif time.monotonic() - running.started_at > running.spec.timeout_s:
                self._kill(running)
                self._finish(running, timed_out=True)

        for spec in self.store.list("pending"):
            if spec.id in self.active:
                continue
            if self._can_admit(spec):
                self._start(spec)

    def run_forever(self, poll_s: float = 5.0) -> None:
        stop = False

        def _handle(signum, frame):
            nonlocal stop
            stop = True

        signal.signal(signal.SIGTERM, _handle)
        signal.signal(signal.SIGINT, _handle)

        while not stop:
            try:
                self.tick()
            except Exception as exc:  # noqa: BLE001 - the daemon must not die
                print(f"gpuq-runner: tick failed: {exc}", file=sys.stderr)
            time.sleep(poll_s)

        for running in list(self.active.values()):
            self._kill(running)
            self._finish(running, timed_out=False)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="gpuq-runner")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--poll", type=float, default=5.0)
    parser.add_argument("--once", action="store_true",
                        help="Run a single tick and exit. For debugging.")
    args = parser.parse_args(argv)

    runner = Runner(load_config(Path(args.config)))
    if args.once:
        runner.tick()
        return 0
    runner.run_forever(poll_s=args.poll)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Add the console script**

In `pyproject.toml`, under `[project.scripts]`, add:

```toml
gpuq-runner = "gpuq.runner:main"
```

- [ ] **Step 5: Run the tests**

Run: `python -m pytest tests/test_runner.py -v`
Expected: PASS, all seven.

- [ ] **Step 6: Run the whole suite**

Run: `python -m pytest -v`
Expected: PASS. Note the total count; Task 10 must not change it.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat(runner): lane admission, watchdog and artifact commit

One process admits CPU jobs up to a cap and GPU jobs one at a time behind
the lock. tick() is a single non-blocking pass so the loop is testable
without a daemon."
```

---

### Task 10: Bootstrap and supervisor

Makes a rebuilt box identical rather than similar.

**Files:**
- Create: `bootstrap.sh`, `supervisor/gpuq-runner.conf`, `examples/queue.toml`
- Modify: `README.md` — replace the Status section with real usage

**Interfaces:**
- Consumes: everything above.
- Produces: no Python API.

- [ ] **Step 1: Write the example config**

Create `examples/queue.toml`:

```toml
[queue]
root = "/workspace/queue"
cpu_slots = 4
lock_dir = "/var/lock/gpu"

[project.wgan-synthetic]
remote = "git@github.com:Daniel-T-S-Adams/wgan-synthetic.git"
checkout = "/workspace/checkouts/wgan-synthetic"
venv = "/workspace/checkouts/wgan-synthetic/.venv"
commit_artifacts = true
```

- [ ] **Step 2: Write the supervisor program file**

Create `supervisor/gpuq-runner.conf`:

```ini
[program:gpuq-runner]
command=/usr/local/bin/gpuq-runner --config /etc/gpuq/queue.toml
autostart=true
autorestart=true
startsecs=5
stopsignal=TERM
stopwaitsecs=30
stdout_logfile=/var/log/gpuq-runner.out.log
stderr_logfile=/var/log/gpuq-runner.err.log
user=root
```

`autorestart=true` with `stopsignal=TERM` is what makes the reaper's
requeue-once path reachable: the runner is expected to die and come back.

- [ ] **Step 3: Write the bootstrap script**

Create `bootstrap.sh`:

```bash
#!/usr/bin/env bash
# Take a bare box to a working gpuq runner. Idempotent: safe to re-run.
#
# Usage: ./bootstrap.sh [path/to/queue.toml]
set -euo pipefail

CONFIG_SRC="${1:-examples/queue.toml}"
CONFIG_DST=/etc/gpuq/queue.toml
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> installing gpuq from ${REPO_DIR}"
python3 -m pip install --quiet --upgrade "${REPO_DIR}"

echo "==> installing config to ${CONFIG_DST}"
mkdir -p "$(dirname "${CONFIG_DST}")"
if [[ ! -f "${CONFIG_DST}" ]]; then
    cp "${CONFIG_SRC}" "${CONFIG_DST}"
else
    echo "    exists, leaving alone"
fi

QUEUE_ROOT="$(python3 -c "
from gpuq.config import load_config
print(load_config('${CONFIG_DST}').queue_root)
")"
LOCK_DIR="$(python3 -c "
from gpuq.config import load_config
print(load_config('${CONFIG_DST}').lock_dir)
")"

echo "==> creating ${QUEUE_ROOT} and ${LOCK_DIR}"
mkdir -p "${QUEUE_ROOT}"/{pending,running,done,failed,logs}
mkdir -p "${LOCK_DIR}"

echo "==> cloning project checkouts"
python3 -c "
from gpuq.config import load_config
from gpuq.gitops import ensure_checkout
for project in load_config('${CONFIG_DST}').projects.values():
    print(f'    {project.name} -> {ensure_checkout(project)}')
"

echo "==> installing the supervisor program"
SUPERVISOR_DIR=/etc/supervisor/conf.d
if [[ -d "${SUPERVISOR_DIR}" ]]; then
    cp "${REPO_DIR}/supervisor/gpuq-runner.conf" "${SUPERVISOR_DIR}/"
    supervisorctl reread
    supervisorctl update
    supervisorctl restart gpuq-runner || supervisorctl start gpuq-runner
    supervisorctl status gpuq-runner
else
    echo "    no ${SUPERVISOR_DIR}; start manually with: gpuq-runner --config ${CONFIG_DST}"
fi

echo "==> done"
```

- [ ] **Step 4: Make it executable and shellcheck it**

```bash
chmod +x bootstrap.sh
bash -n bootstrap.sh
```

Expected: no output from `bash -n` (syntax is valid).

- [ ] **Step 5: Verify bootstrap works against a scratch config**

```bash
mkdir -p /tmp/gpuq-smoke
cat > /tmp/gpuq-smoke/queue.toml <<'EOF'
[queue]
root = "/tmp/gpuq-smoke/queue"
cpu_slots = 2
lock_dir = "/tmp/gpuq-smoke/locks"
EOF
python3 -m pip install --quiet -e .
python3 -c "
from gpuq.config import load_config
cfg = load_config('/tmp/gpuq-smoke/queue.toml')
print(cfg.queue_root, cfg.cpu_slots, cfg.projects)
"
```

Expected: `/tmp/gpuq-smoke/queue 2 {}`. This exercises the config path
bootstrap depends on without needing root or supervisor.

- [ ] **Step 6: End-to-end smoke test**

```bash
gpuq --queue-root /tmp/gpuq-smoke/queue submit \
    --id smoke --lane cpu --project demo \
    --commit HEAD --branch main -- echo hello
gpuq --queue-root /tmp/gpuq-smoke/queue list
gpuq --queue-root /tmp/gpuq-smoke/queue show smoke
```

Expected: `smoke` printed by submit, one `pending cpu smoke demo` line from
list, and a JSON spec from show. The job stays pending because no runner is
running and `demo` is not a configured project — that is correct.

- [ ] **Step 7: Rewrite the README Status section**

Replace the `## Status` section with:

````markdown
## Install

```bash
git clone git@github.com:FibonAdithya/gpu-queue-management.git
cd gpu-queue-management
./bootstrap.sh path/to/queue.toml
```

Idempotent — re-run it after any config change.

## Use

Wrap any command in the GPU lock:

```bash
gpu-claim -- python -m my_project.train --config foo.yaml
```

Queue work and walk away:

```bash
gpuq submit --id glove-v0 --lane gpu --project wgan-synthetic \
    --commit a1b2c3d --branch ds/glove \
    --artifact runs/glove/v0/summary.json \
    --timeout 21600 \
    -- python -m src.train.train_wgan_gp --config configs/glove/v0.yaml

gpuq list
gpuq show glove-v0
gpuq logs glove-v0
gpuq cancel glove-v0
```

## Configure

See `examples/queue.toml`. Installed to `/etc/gpuq/queue.toml` by bootstrap.
````

- [ ] **Step 8: Run the full suite one final time**

Run: `python -m pytest -v`
Expected: PASS, with the same count recorded in Task 9 Step 6. Task 10 adds no
Python.

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "feat(bootstrap): idempotent box setup and supervisor program

Ships the supervisor config in the repo rather than adding it by hand,
which is what makes a rebuilt box identical rather than similar."
```

---

## Not in this plan

From `docs/design.md`, deliberately deferred:

- **Multi-GPU scheduling.** The lane model would extend to N GPU slots keyed by
  UUID; nothing here anticipates it.
- **Multi-host scheduling.** `flock` is host-local by nature.
- **Durable artifact storage.** Consumers commit what they want to keep.
- **Authentication.** Anyone who can write to the queue root can queue work;
  ssh access to the box is the security boundary.
- **A `gpuq wait` subcommand.** Producers poll `gpuq show` today. Worth adding
  once something actually needs to block.
