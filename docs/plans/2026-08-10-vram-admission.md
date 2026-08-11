# Capacity-based GPU admission Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a GPU job declare its VRAM footprint so independent submitters share one card, replacing the hardcoded one-job GPU lane with admission against declared capacity.

**Architecture:** `flock` stops being held for the duration of a run and becomes a short-lived mutex over a *ledger* — one JSON record per holder under `<key>.lock.d/`. Two new modules split by responsibility: `procs.py` owns process liveness and process trees (extracted so both `preflight` and the ledger can use them without an import cycle), and `ledger.py` owns records, capacity arithmetic, acquire/release, and attribution of a CUDA pid to its record. `claim.gpu_claim` keeps its public signature and delegates accounting to the ledger. Enforcement is a watchdog riding the reaper's existing `nvidia-smi` sweep: a holder over its declaration on two consecutive sweeps is killed, so an under-declaration convicts its author instead of leaving two jobs to share an unattributable OOM.

**Tech Stack:** Python 3.11+ (stdlib only — `fcntl`, `json`, `os`, `secrets`, `subprocess`, `tomllib`), `nvidia-smi` on the box, pytest.

**Design doc:** `docs/specs/2026-08-10-vram-admission-design.md`. Section references below (§N) point at it.

## Global Constraints

- **Python 3.11+**, standard library only. No new dependencies.
- **Never import torch** anywhere in this package. `gpuid.py`'s module docstring gives the reason: a JAX project must be able to use this lock, and importing torch in the runner daemon costs seconds and initializes a CUDA context in the process that must never touch the card.
- **The runner is single-threaded.** Nothing added to the tick may block it. The ledger mutex is held for the duration of a directory read and one atomic write, never across a subprocess call.
- **Ledger state must stay legible to `ls` and repairable with `rm`.** One file per holder, never one shared mutated document.
- **Record writes are atomic**: write `<name>.json.part`, then `os.rename`. `preflight` and the reaper read records *without* the mutex, so a half-written file must never be observable.
- **`vram_mb: None` means exclusive** — the whole card. It is the default everywhere, and is what preserves today's behaviour for undeclared jobs.
- **Comments explain why, not what**, wrapped at 76 columns, matching the surrounding files.
- Run the full suite with `.venv/bin/python -m pytest tests/ -q` before every commit.

## File Structure

**Created:**

| File | Responsibility |
|---|---|
| `src/gpuqueue/procs.py` | Process liveness (`pid_alive`) and process trees (`descendants`). Extracted from `claim.py`/`preflight.py` so `ledger.py` can use both without importing `claim`, which will import `ledger`. |
| `src/gpuqueue/ledger.py` | The claim ledger: `Record`, reading/writing records, capacity arithmetic, `acquire`/`release` under the mutex, and `attribute()` mapping a CUDA pid to its owning record. |
| `tests/test_procs.py` | Liveness and tree walking. |
| `tests/test_ledger.py` | Records, arithmetic, mutual exclusion, attribution. |

**Modified:**

| File | Change |
|---|---|
| `src/gpuqueue/spec.py` | `JobSpec.vram_mb` + validation. |
| `src/gpuqueue/config.py` | `gpu_vram_mb`, `gpu_vram_reserve_mb`, `gpu_max_jobs`, `enforce_vram`. |
| `src/gpuqueue/gpuid.py` | `total_vram_mb()` — same `--query-gpu` family as the existing uuid query. |
| `src/gpuqueue/claim.py` | `gpu_claim` gains `vram_mb` and delegates to the ledger; `list_claims`/`release_stale` read ledger records plus legacy files. |
| `src/gpuqueue/preflight.py` | "foreign" becomes "unledgered". |
| `src/gpuqueue/reaper.py` | Orphan kill uses attribution; adds the VRAM watchdog. |
| `src/gpuqueue/runner.py` | `gpu_max_jobs` capacity, forwards `vram_mb`, records `usage_pid`, fails permanently-oversized declarations, and (§7) retries a convicted co-tenant's victim. |
| `src/gpuqueue/cli_gpuq.py`, `cli_claim.py` | `--vram-mb`. |
| `docs/design.md`, `README.md`, `gpuq.example.toml`, `docs/deploying.md` | Lock protocol, the scope line the README currently states, config keys, upgrade note. |

**Why `procs.py` exists:** `ledger.attribute()` needs `descendants()`, which lives in `preflight.py`; `preflight` imports `pid_alive` from `claim`; `claim` will import `ledger`. Without extraction that is a cycle. `procs.py` has no gpuqueue imports at all, so everything else can depend on it.

**Accepted trade-off:** `descendants()` is one `ps` subprocess per node in the tree, and the watchdog now walks one tree per holder rather than one per box. At `orphan_cuda_interval_s` (default 60) with `gpu_max_jobs` of 2 that is a handful of subprocesses a minute on a loop that already shells out to `nvidia-smi`. Do not add caching to avoid it; a stale process tree kills the wrong job.

---

### Task 1: Extract process helpers into `procs.py`

Pure refactor. The suite must be green before and after with no test changes beyond the new file's own.

**Files:**
- Create: `src/gpuqueue/procs.py`
- Modify: `src/gpuqueue/claim.py:43-50` (remove `pid_alive` body, re-export)
- Modify: `src/gpuqueue/preflight.py:63-71` (remove `_descendants`, import)
- Test: `tests/test_procs.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `procs.pid_alive(pid: int) -> bool`, `procs.descendants(pid: int) -> set[int]`. `claim.pid_alive` and `preflight._descendants` remain importable names bound to these, because `tests/test_claim.py:6` and `tests/test_reaper.py` import them from their current homes.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_procs.py
import os
import subprocess
import sys
import time

from gpuqueue.procs import pid_alive, descendants


def test_pid_alive_true_for_self():
    assert pid_alive(os.getpid()) is True


def test_pid_alive_false_for_impossible_pid():
    assert pid_alive(4000000) is False


def test_pid_alive_false_for_zero_and_negative():
    assert pid_alive(0) is False
    assert pid_alive(-1) is False


def test_descendants_finds_a_grandchild():
    """One ps call per node, so a grandchild only shows up if the walk
    actually recurses."""
    child = subprocess.Popen(
        [sys.executable, "-c",
         "import subprocess,sys,time;"
         "subprocess.Popen(['sleep','30']);print('up',flush=True);"
         "time.sleep(30)"],
        stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "up"
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            kids = descendants(os.getpid())
            if len(kids) >= 2:
                break
            time.sleep(0.05)
        assert child.pid in kids
        assert len(kids) >= 2, "did not recurse past the direct child"
    finally:
        child.kill()
        child.wait()


def test_descendants_of_a_leaf_is_empty():
    assert descendants(4000000) == set()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_procs.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'gpuqueue.procs'`

- [ ] **Step 3: Write the module**

```python
# src/gpuqueue/procs.py
"""Process liveness and process trees.

Split out of `claim` and `preflight` because `ledger` needs both and
`claim` imports `ledger`. This module imports nothing from gpuqueue, which
is what keeps that from being a cycle.
"""
from __future__ import annotations

import errno
import os
import subprocess


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError as e:
        return e.errno == errno.EPERM  # exists but not ours
    return True


def descendants(pid: int) -> set[int]:
    """Every process under `pid`, transitively.

    Used to decide who is exempt from the orphan sweep and who owns a CUDA
    process. Both callers need the whole tree: a claim names the process
    that took the lock, but the process on the card is normally its child,
    and a trainer's dataloader workers are children of that.
    """
    try:
        out = subprocess.run(["ps", "-o", "pid=", "--ppid", str(pid)],
                             check=True, capture_output=True, text=True,
                             timeout=15).stdout
    except Exception:
        return set()
    kids = {int(l) for l in out.split() if l.strip().isdigit()}
    for k in list(kids):
        kids |= descendants(k)
    return kids
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_procs.py -q`
Expected: PASS (4 tests)

- [ ] **Step 5: Rewire `claim.py` and `preflight.py`**

In `src/gpuqueue/claim.py`, delete the `pid_alive` function body and its now-unused `errno` import, and add to the imports:

```python
from .procs import pid_alive          # re-exported: callers import it from here
```

In `src/gpuqueue/preflight.py`, delete `_descendants` and change the imports:

```python
from .claim import list_claims
from .procs import descendants as _descendants, pid_alive
```

Leave every call site spelled `_descendants(...)` so `tests/test_reaper.py`'s reproduction, which calls `_pf._descendants`, keeps working.

- [ ] **Step 6: Run the full suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: PASS, 304 tests (300 existing + 4 new), no failures.

- [ ] **Step 7: Commit**

```bash
git add src/gpuqueue/procs.py src/gpuqueue/claim.py src/gpuqueue/preflight.py tests/test_procs.py
git commit -m "refactor: extract process helpers into procs.py

The ledger needs pid_alive and descendants, and claim will import the
ledger -- so leaving them in claim and preflight would be a cycle. No
behaviour change; both names stay importable from their old homes."
```

---

### Task 2: `JobSpec.vram_mb`

**Files:**
- Modify: `src/gpuqueue/spec.py:27-44` (field), `src/gpuqueue/spec.py:56-77` (validation)
- Test: `tests/test_spec.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `JobSpec.vram_mb: int | None = None`. `None` means exclusive — the whole card. Task 13 reads it, Task 15 sets it from the CLI.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_spec.py
def test_vram_mb_defaults_to_none_meaning_the_whole_card():
    assert mkspec().vram_mb is None


def test_vram_mb_accepts_a_positive_int():
    spec = mkspec(vram_mb=512)
    spec.validate()
    assert spec.vram_mb == 512
    assert spec.to_dict()["vram_mb"] == 512


@pytest.mark.parametrize("bad", [0, -1, "512", 512.0])
def test_vram_mb_rejects_anything_but_a_positive_int(bad):
    with pytest.raises(SpecError, match="vram_mb"):
        mkspec(vram_mb=bad).validate()


def test_vram_mb_round_trips_through_from_dict():
    spec = JobSpec.from_dict(mkspec(vram_mb=512).to_dict())
    assert spec.vram_mb == 512
```

Check the top of `tests/test_spec.py` for the existing `mkspec` helper and `SpecError`/`JobSpec`/`pytest` imports; add only what is missing.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_spec.py -q -k vram`
Expected: FAIL — `SpecError: unknown fields: ['vram_mb']` from `from_dict`, and `TypeError` on the others.

- [ ] **Step 3: Add the field**

In `src/gpuqueue/spec.py`, after `timeout_s`:

```python
    timeout_s: int = 3600
    # None means "the whole card": admit alone, exclude everything else.
    # That is the default because a job that has not said what it needs
    # cannot be admitted alongside anything safely -- and it is what keeps
    # every spec written before this field behaving exactly as it did.
    vram_mb: int | None = None
```

In `validate()`, after the `timeout_s` check:

```python
        if self.vram_mb is not None and (not isinstance(self.vram_mb, int)
                                         or isinstance(self.vram_mb, bool)
                                         or self.vram_mb <= 0):
            raise SpecError(
                f"vram_mb must be a positive int or None (meaning the whole "
                f"card), got {self.vram_mb!r}")
```

`bool` is excluded explicitly because `isinstance(True, int)` is `True` in Python, and `--vram-mb` reaching validation as `True` would then declare a 1 MiB footprint.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_spec.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/gpuqueue/spec.py tests/test_spec.py
git commit -m "feat: JobSpec.vram_mb, defaulting to the whole card

None means exclusive, which is what keeps every existing spec behaving
as it does today once admission starts reading this field."
```

---

### Task 3: `[queue]` capacity configuration

> **Execution order:** run this task *after* Task 8, not in numeric order. It imports `ledger.DEFAULT_RESERVE_MB` so the reserve has one source of truth, and `ledger.py` does not exist until Task 5. Nothing else about the task changes. `ledger` never imports `config`, so this is not a cycle.

**Files:**
- Modify: `src/gpuqueue/config.py:60-69` (`RunnerConfig`), `src/gpuqueue/config.py:82-88` (parsing)
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `RunnerConfig.gpu_vram_mb: int | None`, `.gpu_vram_reserve_mb: int`, `.gpu_max_jobs: int`, `.enforce_vram: bool`. Tasks 12 and 13 read all four.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_config.py
def test_capacity_defaults(tmp_path):
    cfg = write_and_load(tmp_path, """
        [queue]
        root = "/q"
    """)
    assert cfg.gpu_vram_mb is None      # discovered from nvidia-smi
    assert cfg.gpu_vram_reserve_mb == 512
    assert cfg.gpu_max_jobs == 2
    assert cfg.enforce_vram is True


def test_capacity_keys_are_read(tmp_path):
    cfg = write_and_load(tmp_path, """
        [queue]
        root = "/q"
        gpu_vram_mb = 8188
        gpu_vram_reserve_mb = 1024
        gpu_max_jobs = 4
        enforce_vram = false
    """)
    assert (cfg.gpu_vram_mb, cfg.gpu_vram_reserve_mb) == (8188, 1024)
    assert cfg.gpu_max_jobs == 4
    assert cfg.enforce_vram is False


def test_gpu_max_jobs_must_be_at_least_one(tmp_path):
    with pytest.raises(ConfigError, match="gpu_max_jobs"):
        write_and_load(tmp_path, '[queue]\nroot = "/q"\ngpu_max_jobs = 0\n')


def test_reserve_may_not_swallow_the_whole_card(tmp_path):
    """A reserve at or above capacity admits nothing and would leave every
    GPU job pending forever, which is worse than refusing to start."""
    with pytest.raises(ConfigError, match="gpu_vram_reserve_mb"):
        write_and_load(tmp_path, """
            [queue]
            root = "/q"
            gpu_vram_mb = 1024
            gpu_vram_reserve_mb = 1024
        """)
```

`tests/test_config.py` already has a helper that writes a TOML file and calls `load_config`; reuse it under whatever name it has rather than adding `write_and_load`. If none exists, add:

```python
def write_and_load(tmp_path, body):
    p = tmp_path / "gpuq.toml"
    p.write_text(textwrap.dedent(body))
    return load_config(p)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_config.py -q -k "capacity or gpu_max_jobs or reserve"`
Expected: FAIL — `AttributeError: 'RunnerConfig' object has no attribute 'gpu_vram_mb'`

- [ ] **Step 3: Add the fields and parsing**

In `RunnerConfig`, after `cpu_slots`:

```python
    cpu_slots: int = 4
    # None means "ask the card". An explicit value is for boxes where the
    # query is unavailable or reports something the driver will not
    # actually hand out.
    gpu_vram_mb: int | None = None
    # One source of truth with the standalone gpu-claim path, which needs
    # the same number and has no config to read it from.
    gpu_vram_reserve_mb: int = DEFAULT_RESERVE_MB
    # A latency budget, not a safety one. VRAM accounting alone would admit
    # sixteen 500 MiB jobs onto an 8 GB card, all time-slicing, each slower
    # than it would have been queued -- and with independent submitters
    # that cost lands on a stranger. Two is what has been measured (15% to
    # 62% utilization); raise it on a box that has measured more.
    gpu_max_jobs: int = 2
    enforce_vram: bool = True
```

Add `from .ledger import DEFAULT_RESERVE_MB` to the imports at the top of `config.py`.

In `load_config`, after the `cpu_slots` check:

```python
    gpu_vram_mb = queue.get("gpu_vram_mb")
    gpu_vram_mb = int(gpu_vram_mb) if gpu_vram_mb is not None else None
    gpu_vram_reserve_mb = int(queue.get("gpu_vram_reserve_mb",
                                        DEFAULT_RESERVE_MB))
    gpu_max_jobs = int(queue.get("gpu_max_jobs", 2))
    if gpu_max_jobs < 1:
        raise ConfigError("[queue].gpu_max_jobs must be >= 1")
    if gpu_vram_reserve_mb < 0:
        raise ConfigError("[queue].gpu_vram_reserve_mb must be >= 0")
    if gpu_vram_mb is not None and gpu_vram_reserve_mb >= gpu_vram_mb:
        raise ConfigError(
            f"[queue].gpu_vram_reserve_mb ({gpu_vram_reserve_mb}) must be "
            f"less than gpu_vram_mb ({gpu_vram_mb}); a reserve that "
            "swallows the card admits nothing and queues GPU jobs forever")
```

and pass all four into the `RunnerConfig(...)` call at the bottom.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_config.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/gpuqueue/config.py tests/test_config.py
git commit -m "feat: [queue] capacity keys for GPU admission

gpu_max_jobs is a latency budget rather than a safety one: VRAM alone
would admit many small jobs that then merely time-slice."
```

---

### Task 4: `gpuid.total_vram_mb()`

**Files:**
- Modify: `src/gpuqueue/gpuid.py` (append)
- Test: `tests/test_gpuid.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `gpuid.total_vram_mb(index: int = 0) -> int | None`. `None` means the card could not be queried. Task 13 turns that into a capacity.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_gpuid.py
from gpuqueue.gpuid import total_vram_mb


def test_total_vram_mb_parses_mib(monkeypatch):
    monkeypatch.setattr(gi, "_run", lambda argv: "8188 MiB\n")
    assert total_vram_mb() == 8188


def test_total_vram_mb_reads_the_requested_index(monkeypatch):
    monkeypatch.setattr(gi, "_run", lambda argv: "8188 MiB\n16376 MiB\n")
    assert total_vram_mb(1) == 16376


def test_total_vram_mb_none_when_smi_is_missing(monkeypatch):
    def boom(argv):
        raise FileNotFoundError()
    monkeypatch.setattr(gi, "_run", boom)
    assert total_vram_mb() is None


def test_total_vram_mb_none_on_unparseable_output(monkeypatch):
    monkeypatch.setattr(gi, "_run", lambda argv: "[N/A]\n")
    assert total_vram_mb() is None


def test_total_vram_mb_none_when_index_is_past_the_end(monkeypatch):
    monkeypatch.setattr(gi, "_run", lambda argv: "8188 MiB\n")
    assert total_vram_mb(3) is None
```

`tests/test_gpuid.py` already imports the module; check whether it is bound as `gi` and match the existing alias.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_gpuid.py -q -k total_vram`
Expected: FAIL — `ImportError: cannot import name 'total_vram_mb'`

- [ ] **Step 3: Implement**

```python
# append to src/gpuqueue/gpuid.py
def total_vram_mb(index: int = 0) -> int | None:
    """What the card reports as its total, in MiB. None when unknown.

    Same query family as the uuid lookup above, and deliberately the same
    yardstick the ledger and the watchdog use: nvidia-smi's own MiB, not
    torch's idea of it.
    """
    try:
        out = _run(["nvidia-smi", "--query-gpu=memory.total",
                    "--format=csv,noheader"])
    except Exception:
        return None
    lines = [l.strip() for l in out.splitlines() if l.strip()]
    if index >= len(lines):
        return None
    digits = lines[index].replace("MiB", "").strip()
    return int(digits) if digits.isdigit() else None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_gpuid.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/gpuqueue/gpuid.py tests/test_gpuid.py
git commit -m "feat: gpuid.total_vram_mb for capacity discovery"
```

---

### Task 5: Ledger records

**Files:**
- Create: `src/gpuqueue/ledger.py`
- Test: `tests/test_ledger.py`

**Interfaces:**
- Consumes: `procs.pid_alive`, `procs.descendants` (Task 1); `gpuid.lock_filename`; `spec.utcnow_iso`.
- Produces: `ledger.Record` (fields `path, pid, usage_pid, vram_mb, owner, cmd, started_at, key`, property `name`, method `to_dict()`), `ledger.DEFAULT_RESERVE_MB = 512`, `ledger.ClaimBusy`, `ledger.mutex_path(key, directory)`, `ledger.ledger_dir(key, directory)`, `ledger.records_for(key, directory)`, `ledger.all_records(directory)`, `ledger.write_record(rec)`, `ledger.set_usage_pid(rec, pid)`, `ledger.remove(rec)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ledger.py
import json
from pathlib import Path

import pytest

from gpuqueue import ledger as lg

KEY = "4b8f2c1a-0000-0000-0000-000000000001"


def mkrec(tmp_path, name="100.aaa.json", **over):
    d = dict(pid=100, usage_pid=101, vram_mb=512, owner="me",
             cmd=["python", "t.py"], started_at="2026-08-10T00:00:00Z",
             key=KEY)
    d.update(over)
    rec = lg.Record(path=lg.ledger_dir(KEY, tmp_path) / name, **d)
    lg.write_record(rec)
    return rec


def test_ledger_dir_sits_beside_the_mutex(tmp_path):
    assert lg.mutex_path(KEY, tmp_path).name.endswith(".lock")
    assert lg.ledger_dir(KEY, tmp_path).name.endswith(".lock.d")
    assert lg.ledger_dir(KEY, tmp_path).parent == tmp_path


def test_record_round_trips(tmp_path):
    mkrec(tmp_path)
    (got,) = lg.records_for(KEY, tmp_path)
    assert (got.pid, got.usage_pid, got.vram_mb) == (100, 101, 512)
    assert got.cmd == ["python", "t.py"]
    assert got.owner == "me"


def test_write_is_atomic_so_readers_never_see_a_partial(tmp_path):
    """preflight and the reaper read records without the mutex."""
    rec = mkrec(tmp_path)
    assert not list(rec.path.parent.glob("*.part"))
    assert json.loads(rec.path.read_text())["vram_mb"] == 512


def test_garbage_records_are_skipped_not_fatal(tmp_path):
    mkrec(tmp_path)
    bad = lg.ledger_dir(KEY, tmp_path) / "999.zzz.json"
    bad.write_text("{not json")
    assert [r.pid for r in lg.records_for(KEY, tmp_path)] == [100]


def test_exclusive_record_reads_back_as_none(tmp_path):
    mkrec(tmp_path, vram_mb=None)
    assert lg.records_for(KEY, tmp_path)[0].vram_mb is None


def test_reserved_record_has_no_usage_pid(tmp_path):
    mkrec(tmp_path, usage_pid=None)
    assert lg.records_for(KEY, tmp_path)[0].usage_pid is None


def test_set_usage_pid_persists(tmp_path):
    rec = mkrec(tmp_path, usage_pid=None)
    lg.set_usage_pid(rec, 4242)
    assert lg.records_for(KEY, tmp_path)[0].usage_pid == 4242


def test_remove_deletes_the_record(tmp_path):
    lg.remove(mkrec(tmp_path))
    assert lg.records_for(KEY, tmp_path) == []


def test_a_legacy_claim_file_reads_as_an_exclusive_holder(tmp_path):
    """During an upgrade an old gpu-claim still has <key>.lock.json out.
    Read as exclusive and owning its own tree, or the reaper treats its
    trainer as unledgered and kills it."""
    legacy = Path(str(lg.mutex_path(KEY, tmp_path)) + ".json")
    legacy.write_text(json.dumps(
        {"pid": 777, "owner": "alice", "cmd": ["python", "old.py"],
         "started_at": "2026-08-10T00:00:00Z", "key": KEY}))
    (got,) = lg.records_for(KEY, tmp_path)
    assert got.vram_mb is None
    assert got.usage_pid == 777


def test_all_records_spans_every_key(tmp_path):
    mkrec(tmp_path)
    other = lg.Record(path=lg.ledger_dir("other-uuid", tmp_path) / "200.bbb.json",
                      pid=200, usage_pid=201, vram_mb=256, owner="you",
                      cmd=[], started_at="2026-08-10T00:00:00Z",
                      key="other-uuid")
    lg.write_record(other)
    assert sorted(r.pid for r in lg.all_records(tmp_path)) == [100, 200]


def test_all_records_on_a_missing_directory_is_empty(tmp_path):
    assert lg.all_records(tmp_path / "nope") == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_ledger.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'gpuqueue.ledger'`

- [ ] **Step 3: Write the module**

```python
# src/gpuqueue/ledger.py
"""The claim ledger: who holds this card, and for how much VRAM.

`flock` cannot be a counting semaphore, so it changes role. `<key>.lock`
is taken only for the milliseconds needed to read the holders, decide, and
write a record -- it guards the accounting, not the card. Holders live one
file per holder under `<key>.lock.d/`, which is what keeps `ls` able to
show who is on the card and `rm` able to clear one wedged holder. A single
mutated document would give both up exactly when something is stuck, since
a torn write blinds every participant at once.

`vram_mb = None` means exclusive: the whole card. It fits only into an
empty ledger, and nothing fits alongside it. That one rule is what makes
an undeclared claim behave as it did before any of this existed.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from .gpuid import lock_filename
from .procs import pid_alive
from .spec import utcnow_iso

# Held back from admission. Two processes that each fit exactly still have
# their allocators fragmenting the same heap.
DEFAULT_RESERVE_MB = 512


class ClaimBusy(RuntimeError):
    """The card has no room for this claim."""


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

    @property
    def name(self) -> str:
        return self.path.name

    def to_dict(self) -> dict:
        return {"pid": self.pid, "usage_pid": self.usage_pid,
                "vram_mb": self.vram_mb, "owner": self.owner,
                "cmd": self.cmd, "started_at": self.started_at,
                "key": self.key}


def mutex_path(key: str, directory) -> Path:
    return Path(directory) / lock_filename(key)


def ledger_dir(key: str, directory) -> Path:
    return Path(str(mutex_path(key, directory)) + ".d")


def _legacy_path(key: str, directory) -> Path:
    return Path(str(mutex_path(key, directory)) + ".json")


def _load(path: Path) -> Record | None:
    try:
        d = json.loads(path.read_text())
        return Record(
            path=path, pid=int(d["pid"]),
            # `is not None`, not truthiness: a stored 0 is falsy, and for
            # vram_mb None means *exclusive* -- the most permissive value
            # there is. Collapsing 0 into it would flip a record's meaning.
            usage_pid=(int(d["usage_pid"])
                       if d.get("usage_pid") is not None else None),
            vram_mb=(int(d["vram_mb"])
                     if d.get("vram_mb") is not None else None),
            owner=d.get("owner", "?"), cmd=list(d.get("cmd") or []),
            started_at=d.get("started_at", ""), key=d.get("key", ""))
    except Exception:
        return None  # a garbage record must not blind us to the good ones


def _load_legacy(path: Path) -> Record | None:
    """A `<key>.lock.json` from a gpu-claim that predates the ledger.

    It took the whole card and the process on the card is under the pid it
    recorded. Reading it as exclusive is what stops the reaper treating an
    old holder's trainer as unledgered and killing it mid-upgrade.
    """
    rec = _load(path)
    if rec is None:
        return None
    rec.vram_mb = None
    if rec.usage_pid is None:
        rec.usage_pid = rec.pid
    return rec


def records_for(key: str, directory) -> list[Record]:
    d = Path(directory)
    out = []
    ldir = ledger_dir(key, d)
    if ldir.is_dir():
        out.extend(r for r in (_load(p) for p in sorted(ldir.glob("*.json")))
                   if r is not None)
    legacy = _load_legacy(_legacy_path(key, d))
    if legacy is not None:
        out.append(legacy)
    return out


def all_records(directory) -> list[Record]:
    d = Path(directory)
    if not d.is_dir():
        return []
    out = []
    for ldir in sorted(d.glob("*.lock.d")):
        out.extend(r for r in (_load(p) for p in sorted(ldir.glob("*.json")))
                   if r is not None)
    for p in sorted(d.glob("*.lock.json")):
        legacy = _load_legacy(p)
        if legacy is not None:
            out.append(legacy)
    return out


def live_records(records: list[Record]) -> list[Record]:
    return [r for r in records if pid_alive(r.pid)]


def write_record(rec: Record) -> None:
    """Atomic: preflight and the reaper read records without the mutex, so
    a half-written file must never be observable."""
    rec.path.parent.mkdir(parents=True, exist_ok=True)
    tmp = rec.path.with_suffix(".json.part")
    tmp.write_text(json.dumps(rec.to_dict(), indent=2) + "\n")
    os.replace(tmp, rec.path)


def set_usage_pid(rec: Record, pid: int | None) -> None:
    rec.usage_pid = pid
    write_record(rec)


def remove(rec: Record) -> None:
    rec.path.unlink(missing_ok=True)
```

`utcnow_iso` is imported now because Task 7 uses it; leave the import in place.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_ledger.py -q`
Expected: PASS (11 tests)

- [ ] **Step 5: Commit**

```bash
git add src/gpuqueue/ledger.py tests/test_ledger.py
git commit -m "feat: ledger records, one file per holder

One file per holder rather than one shared document: ls shows who is on
the card and rm clears a wedged holder, which a torn write in a single
mutated document would take away exactly when something is stuck."
```

---

### Task 6: Capacity arithmetic

**Files:**
- Modify: `src/gpuqueue/ledger.py` (append)
- Test: `tests/test_ledger.py` (append)

**Interfaces:**
- Consumes: `ledger.Record` (Task 5).
- Produces: `ledger.fits(records, want_mb, usable_mb) -> bool`, `ledger.exceeds_capacity(want_mb, usable_mb) -> bool`, `ledger.free_mb(records, usable_mb) -> int`, `ledger.busy_message(key, records, want_mb, usable_mb) -> str`. `usable_mb is None` means the card could not be queried. Task 7 calls all four; Task 13 calls `exceeds_capacity`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_ledger.py
def rec(vram_mb, pid=100, owner="me"):
    return lg.Record(path=Path(f"/tmp/{pid}.x.json"), pid=pid, usage_pid=pid,
                     vram_mb=vram_mb, owner=owner, cmd=["python", "t.py"],
                     started_at="2026-08-10T00:00:00Z", key=KEY)


def test_a_declared_claim_fits_an_empty_ledger():
    assert lg.fits([], 512, 7676) is True


def test_declarations_sum_against_capacity():
    assert lg.fits([rec(4000)], 3676, 7676) is True
    assert lg.fits([rec(4000)], 3677, 7676) is False


def test_exclusive_fits_only_an_empty_ledger():
    assert lg.fits([], None, 7676) is True
    assert lg.fits([rec(16)], None, 7676) is False


def test_nothing_fits_alongside_an_exclusive_holder():
    assert lg.fits([rec(None)], 16, 7676) is False


def test_a_declaration_larger_than_the_card_never_fits():
    assert lg.fits([], 9000, 7676) is False
    assert lg.exceeds_capacity(9000, 7676) is True
    assert lg.exceeds_capacity(7676, 7676) is False


def test_unknown_capacity_degrades_to_exclusive():
    """A box whose card cannot be queried gets the old behaviour rather
    than arithmetic on a number nobody has."""
    assert lg.fits([], 512, None) is True
    assert lg.fits([rec(16)], 512, None) is False
    assert lg.exceeds_capacity(512, None) is False


def test_free_mb_reports_the_remainder():
    assert lg.free_mb([rec(4000)], 7676) == 3676
    assert lg.free_mb([rec(None)], 7676) == 0
    assert lg.free_mb([], None) == 0


def test_busy_message_names_the_holders_and_the_shortfall():
    msg = lg.busy_message(KEY, [rec(4000, pid=42, owner="gpuq:job-a")],
                          4000, 7676)
    assert "4000" in msg and "3676" in msg
    assert "42" in msg and "gpuq:job-a" in msg
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_ledger.py -q -k "fits or capacity or free_mb or busy"`
Expected: FAIL — `AttributeError: module 'gpuqueue.ledger' has no attribute 'fits'`

- [ ] **Step 3: Implement**

```python
# append to src/gpuqueue/ledger.py
def exceeds_capacity(want_mb: int | None, usable_mb: int | None) -> bool:
    """A declaration that can never be admitted, however empty the card.

    The runner needs this apart from `fits` so it can fail such a job
    instead of leaving it pending forever.
    """
    if want_mb is None or usable_mb is None:
        return False
    return want_mb > usable_mb


def fits(records: list[Record], want_mb: int | None,
         usable_mb: int | None) -> bool:
    """`usable_mb is None` means the card could not be queried, so every
    claim is treated as exclusive -- degraded, and the same posture
    preflight already takes when it cannot enumerate the card."""
    if any(r.vram_mb is None for r in records):
        return False
    if want_mb is None or usable_mb is None:
        return not records
    if want_mb > usable_mb:
        return False
    return sum(r.vram_mb or 0 for r in records) + want_mb <= usable_mb


def free_mb(records: list[Record], usable_mb: int | None) -> int:
    if usable_mb is None or any(r.vram_mb is None for r in records):
        return 0
    return max(0, usable_mb - sum(r.vram_mb or 0 for r in records))


def busy_message(key: str, records: list[Record], want_mb: int | None,
                 usable_mb: int | None) -> str:
    want = "the whole card" if want_mb is None else f"{want_mb} MiB"
    head = (f"GPU {key}: need {want}, "
            f"{free_mb(records, usable_mb)} MiB free"
            + (f" of {usable_mb}" if usable_mb is not None else "")
            + ". Holders:")
    lines = [
        f"  pid {r.pid:>7}  {r.owner:<24} "
        f"{'exclusive' if r.vram_mb is None else str(r.vram_mb) + ' MiB':>12}"
        f"  {' '.join(r.cmd) or '?'}"
        for r in records
    ] or ["  (none -- this claim does not fit the card at all)"]
    return "\n".join([head, *lines])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_ledger.py -q`
Expected: PASS (19 tests)

- [ ] **Step 5: Commit**

```bash
git add src/gpuqueue/ledger.py tests/test_ledger.py
git commit -m "feat: ledger capacity arithmetic

exceeds_capacity is separate from fits so the runner can tell a job that
will never fit (fail it) from one that does not fit yet (wait)."
```

---

### Task 7: Acquire and release under the mutex

**Files:**
- Modify: `src/gpuqueue/ledger.py` (append)
- Test: `tests/test_ledger.py` (append)

**Interfaces:**
- Consumes: Tasks 5 and 6.
- Produces: `ledger.MUTEX_WAIT_S = 10.0`; `ledger.acquire(key, *, vram_mb, owner, cmd, directory, usable_mb, usage_pid=None) -> Record`, raising `ClaimBusy` when the card has no room or the mutex cannot be taken. Task 9 wraps it.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_ledger.py
import os
import subprocess
import sys


def test_acquire_writes_a_live_record(tmp_path):
    got = lg.acquire(KEY, vram_mb=512, owner="me", cmd=["python", "t.py"],
                     directory=tmp_path, usable_mb=7676, usage_pid=os.getpid())
    assert got.pid == os.getpid()
    assert got.usage_pid == os.getpid()
    assert [r.vram_mb for r in lg.records_for(KEY, tmp_path)] == [512]
    lg.remove(got)


def test_two_claims_share_a_card_when_both_fit(tmp_path):
    a = lg.acquire(KEY, vram_mb=3000, owner="a", cmd=[], directory=tmp_path,
                   usable_mb=7676)
    b = lg.acquire(KEY, vram_mb=3000, owner="b", cmd=[], directory=tmp_path,
                   usable_mb=7676)
    assert len(lg.records_for(KEY, tmp_path)) == 2
    lg.remove(a)
    lg.remove(b)


def test_records_do_not_collide_when_one_process_holds_several(tmp_path):
    """The runner holds one record per GPU job, all with its own pid."""
    a = lg.acquire(KEY, vram_mb=100, owner="a", cmd=[], directory=tmp_path,
                   usable_mb=7676)
    b = lg.acquire(KEY, vram_mb=100, owner="b", cmd=[], directory=tmp_path,
                   usable_mb=7676)
    assert a.path != b.path
    assert len(lg.records_for(KEY, tmp_path)) == 2


def test_acquire_refuses_when_the_card_is_full(tmp_path):
    lg.acquire(KEY, vram_mb=7000, owner="a", cmd=[], directory=tmp_path,
               usable_mb=7676)
    with pytest.raises(lg.ClaimBusy, match="MiB free"):
        lg.acquire(KEY, vram_mb=1000, owner="b", cmd=[], directory=tmp_path,
                   usable_mb=7676)


def test_a_dead_holders_record_does_not_reserve_anything(tmp_path):
    mkrec(tmp_path, name="4000000.dead.json", pid=4000000, vram_mb=7000)
    got = lg.acquire(KEY, vram_mb=7000, owner="b", cmd=[], directory=tmp_path,
                     usable_mb=7676)
    assert got.vram_mb == 7000


def test_a_holder_in_another_process_blocks_by_capacity(tmp_path):
    """A real second process is the only honest test: the record has to
    outlive the mutex, which is released the instant acquire returns."""
    holder = subprocess.Popen(
        [sys.executable, "-c",
         "import time;from gpuqueue import ledger as lg;"
         f"lg.acquire({KEY!r},vram_mb=7000,owner='a',cmd=[],"
         f"directory={str(tmp_path)!r},usable_mb=7676);"
         "print('held',flush=True);time.sleep(30)"],
        stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "held"
        with pytest.raises(lg.ClaimBusy):
            lg.acquire(KEY, vram_mb=1000, owner="b", cmd=[],
                       directory=tmp_path, usable_mb=7676)
    finally:
        holder.kill()
        holder.wait()


def test_an_old_style_exclusive_flock_is_reported_as_such(tmp_path, monkeypatch):
    """A gpu-claim from before the ledger holds LOCK_EX for its whole run
    and would otherwise hang us forever."""
    monkeypatch.setattr(lg, "MUTEX_WAIT_S", 0.2)
    holder = subprocess.Popen(
        [sys.executable, "-c",
         "import fcntl,os,time;"
         f"fd=os.open({str(lg.mutex_path(KEY, tmp_path))!r},os.O_CREAT|os.O_RDWR,0o666);"
         "fcntl.flock(fd,fcntl.LOCK_EX);print('held',flush=True);time.sleep(30)"],
        stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "held"
        with pytest.raises(lg.ClaimBusy, match="older gpu-claim"):
            lg.acquire(KEY, vram_mb=512, owner="b", cmd=[],
                       directory=tmp_path, usable_mb=7676)
    finally:
        holder.kill()
        holder.wait()
```

The `mutex_path(...)` call inside the second subprocess string is evaluated in the *parent* before the child starts, which is intended — the child only needs the path.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_ledger.py -q -k acquire`
Expected: FAIL — `AttributeError: module 'gpuqueue.ledger' has no attribute 'acquire'`

- [ ] **Step 3: Implement**

```python
# append to src/gpuqueue/ledger.py, and add `import fcntl`, `import secrets`
# and `import time` to the imports at the top
MUTEX_WAIT_S = 10.0
_MUTEX_POLL_S = 0.05


def _take_mutex(fd: int, timeout_s: float) -> None:
    """Bounded, because a participant only holds this for a directory read
    and one rename. Waiting longer than that means the holder is not
    playing by these rules -- in practice a gpu-claim from before the
    ledger, which takes LOCK_EX for the whole run and would hang us until
    its training finished."""
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except OSError:
            if time.monotonic() >= deadline:
                raise ClaimBusy(
                    f"could not take the ledger mutex within {timeout_s:g}s: "
                    "an older gpu-claim is holding this card exclusively for "
                    "the whole of its run. Wait for it, or upgrade it.")
            time.sleep(_MUTEX_POLL_S)


def acquire(key: str, *, vram_mb: int | None, owner: str,
            cmd: list[str] | None, directory, usable_mb: int | None,
            usage_pid: int | None = None) -> Record:
    """Take a share of the card, or raise ClaimBusy.

    Non-blocking on capacity by design: the caller decides whether to wait,
    and the runner must not, because a single-threaded loop that waits here
    stalls the CPU lane behind whoever holds the card.
    """
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    ldir = ledger_dir(key, d)
    ldir.mkdir(parents=True, exist_ok=True)

    fd = os.open(mutex_path(key, d), os.O_CREAT | os.O_RDWR, 0o666)
    try:
        _take_mutex(fd, MUTEX_WAIT_S)
        live = live_records(records_for(key, d))
        if not fits(live, vram_mb, usable_mb):
            raise ClaimBusy(busy_message(key, live, vram_mb, usable_mb))
        # A token, not just the pid: the runner holds one record per GPU
        # job and every one of them carries the runner's pid.
        rec = Record(path=ldir / f"{os.getpid()}.{secrets.token_hex(3)}.json",
                     pid=os.getpid(), usage_pid=usage_pid, vram_mb=vram_mb,
                     owner=owner, cmd=list(cmd or []),
                     started_at=utcnow_iso(), key=key)
        write_record(rec)
        return rec
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_ledger.py -q`
Expected: PASS (26 tests)

- [ ] **Step 5: Commit**

```bash
git add src/gpuqueue/ledger.py tests/test_ledger.py
git commit -m "feat: ledger acquire/release under a short-lived mutex

The flock now guards the accounting rather than the card, held for a
directory read and one atomic rename. The wait is bounded so a pre-ledger
gpu-claim holding LOCK_EX for its whole run reports that instead of
hanging us until its training finishes."
```

---

### Task 8: Attribution — which record owns this CUDA process

**Files:**
- Modify: `src/gpuqueue/ledger.py` (append)
- Test: `tests/test_ledger.py` (append)

**Interfaces:**
- Consumes: Task 5, `procs.descendants`.
- Produces: `ledger.attribute(apps, records) -> tuple[dict[str, list[dict]], list[dict]]` — `(owned_by_record_name, unledgered)`; `ledger.used_mb(apps) -> int`. Tasks 10, 11 and 12 all call `attribute`, which is the point: preflight, the orphan reaper and the watchdog must never disagree about who owns a pid.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_ledger.py
def app(pid, used_mb=100, name="train.py"):
    return {"pid": pid, "used_mb": used_mb, "name": name}


def test_attribute_charges_a_pid_to_its_records_tree(monkeypatch):
    monkeypatch.setattr(lg, "descendants",
                        lambda pid: {555} if pid == 100 else set())
    r = rec(512, pid=100)
    owned, unledgered = lg.attribute([app(555)], [r])
    assert [a["pid"] for a in owned[r.name]] == [555]
    assert unledgered == []


def test_attribute_charges_the_usage_pid_itself(monkeypatch):
    monkeypatch.setattr(lg, "descendants", lambda pid: set())
    r = rec(512, pid=100)
    owned, unledgered = lg.attribute([app(100)], [r])
    assert [a["pid"] for a in owned[r.name]] == [100]


def test_a_stranger_is_unledgered(monkeypatch):
    monkeypatch.setattr(lg, "descendants", lambda pid: set())
    owned, unledgered = lg.attribute([app(4321)], [rec(512, pid=100)])
    assert owned == {}
    assert [a["pid"] for a in unledgered] == [4321]


def test_a_reserved_record_owns_nothing(monkeypatch):
    """Between acquire and launch there is no process to charge, so the
    record must not silently adopt a stranger's."""
    monkeypatch.setattr(lg, "descendants", lambda pid: {999})
    r = lg.Record(path=Path("/tmp/1.x.json"), pid=100, usage_pid=None,
                  vram_mb=512, owner="me", cmd=[], started_at="", key=KEY)
    owned, unledgered = lg.attribute([app(999)], [r])
    assert owned == {}
    assert [a["pid"] for a in unledgered] == [999]


def test_each_pid_is_charged_to_exactly_one_record(monkeypatch):
    monkeypatch.setattr(lg, "descendants",
                        lambda pid: {pid + 1000})
    a, b = rec(512, pid=100), rec(512, pid=200)
    owned, unledgered = lg.attribute([app(1100), app(1200)], [a, b])
    assert [x["pid"] for x in owned[a.name]] == [1100]
    assert [x["pid"] for x in owned[b.name]] == [1200]
    assert unledgered == []


def test_used_mb_sums_and_tolerates_unknowns():
    assert lg.used_mb([app(1, 200), app(2, 300)]) == 500
    assert lg.used_mb([app(1, None)]) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_ledger.py -q -k "attribute or used_mb"`
Expected: FAIL — `AttributeError: module 'gpuqueue.ledger' has no attribute 'attribute'`

- [ ] **Step 3: Implement**

```python
# append to src/gpuqueue/ledger.py, and add `descendants` to the
# `from .procs import ...` line at the top
def attribute(apps: list[dict],
              records: list[Record]) -> tuple[dict[str, list[dict]], list[dict]]:
    """Charge every visible CUDA process to the record that owns it.

    Returns (owned, unledgered), keyed by record name. Three callers need
    this answer -- preflight, the orphan reaper and the VRAM watchdog --
    and they share one implementation so they cannot disagree about who
    owns a pid, which is the disagreement that gets a legitimate job
    killed.

    A record with no `usage_pid` has been admitted but not launched. It
    owns nothing, and must not adopt a stranger's process.
    """
    trees = {r.name: {r.usage_pid} | descendants(r.usage_pid)
             for r in records if r.usage_pid is not None}
    owned: dict[str, list[dict]] = {}
    unledgered: list[dict] = []
    for app in apps:
        for name, tree in trees.items():
            if app["pid"] in tree:
                owned.setdefault(name, []).append(app)
                break
        else:
            unledgered.append(app)
    return owned, unledgered


def used_mb(apps: list[dict]) -> int:
    """nvidia-smi reports [N/A] for a process it can see but not measure;
    counting that as zero under-reports, which is the safe direction for a
    watchdog that kills."""
    return sum(a.get("used_mb") or 0 for a in apps)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_ledger.py -q`
Expected: PASS (32 tests)

- [ ] **Step 5: Commit**

```bash
git add src/gpuqueue/ledger.py tests/test_ledger.py
git commit -m "feat: attribute a CUDA process to its ledger record

One implementation for preflight, the orphan reaper and the watchdog:
a disagreement between them about who owns a pid is what gets a
legitimate job killed."
```

---

### Task 9: `gpu_claim` on the ledger

**Files:**
- Modify: `src/gpuqueue/claim.py` (replace `gpu_claim`, `list_claims`, `release_stale`, `_paths`; keep `claim_dir`, `read_claim`, `job_orphaned`, `_default_owner`)
- Test: `tests/test_claim.py`

**Interfaces:**
- Consumes: Tasks 4–7.
- Produces: `gpu_claim(key=None, owner=None, cmd=None, wait=False, directory=None, vram_mb=None, usable_mb=None, own_usage=True)`, **yielding a `ledger.Record`** rather than a dict; `claim.default_usable_mb() -> int | None`; `claim.ClaimBusy` (re-exported from `ledger`); `list_claims(directory=None) -> list[tuple[Path, dict]]` unchanged in shape; `release_stale(directory=None) -> list[dict]` unchanged in shape. Task 13 passes `own_usage=False`.

**Breaking change, deliberate:** the context manager yielded a dict and now yields a `Record`, because the runner has to call `ledger.set_usage_pid` on it after the job launches. `tests/test_claim.py:14` uses `c["pid"]` and becomes `c.pid`. The *file* format is what `docs/design.md` pins as the protocol, not the yielded Python object.

- [ ] **Step 1: Write the failing test**

Replace the body of `tests/test_claim.py` above `test_job_orphaned_*` with:

```python
import json
import os
import subprocess
import sys
import pytest
from gpuqueue import ledger as lg
from gpuqueue.claim import (gpu_claim, ClaimBusy, read_claim, pid_alive,
                            list_claims, release_stale, default_usable_mb)

KEY = "4b8f2c1a-0000-0000-0000-000000000001"


def test_claim_writes_a_record_with_pid_and_cmd(tmp_path):
    with gpu_claim(key=KEY, owner="me", cmd=["python", "t.py"],
                   directory=tmp_path, usable_mb=7676) as c:
        assert c.pid == os.getpid()
        (path, body), = list_claims(tmp_path)
        assert body["owner"] == "me"
        assert body["cmd"] == ["python", "t.py"]
        assert body["started_at"].endswith("Z")


def test_an_undeclared_claim_is_exclusive(tmp_path):
    """The default is the whole card, which is what keeps every caller
    written before --vram-mb behaving exactly as it did."""
    with gpu_claim(key=KEY, directory=tmp_path, usable_mb=7676) as c:
        assert c.vram_mb is None
        with pytest.raises(ClaimBusy):
            with gpu_claim(key=KEY, directory=tmp_path, vram_mb=16,
                           usable_mb=7676):
                pass


def test_two_declared_claims_share_the_card(tmp_path):
    with gpu_claim(key=KEY, owner="a", directory=tmp_path, vram_mb=3000,
                   usable_mb=7676):
        with gpu_claim(key=KEY, owner="b", directory=tmp_path, vram_mb=3000,
                       usable_mb=7676):
            assert len(list_claims(tmp_path)) == 2


def test_a_declared_claim_is_refused_when_the_card_is_full(tmp_path):
    with gpu_claim(key=KEY, directory=tmp_path, vram_mb=7000, usable_mb=7676):
        with pytest.raises(ClaimBusy, match="MiB free"):
            with gpu_claim(key=KEY, directory=tmp_path, vram_mb=1000,
                           usable_mb=7676):
                pass


def test_claim_charges_its_own_tree_by_default(tmp_path):
    with gpu_claim(key=KEY, directory=tmp_path, usable_mb=7676) as c:
        assert c.usage_pid == os.getpid()


def test_own_usage_false_leaves_the_record_unattributed(tmp_path):
    """The runner takes the card before the job process exists."""
    with gpu_claim(key=KEY, directory=tmp_path, usable_mb=7676,
                   own_usage=False) as c:
        assert c.usage_pid is None


def test_record_removed_on_exit(tmp_path):
    with gpu_claim(key=KEY, directory=tmp_path, usable_mb=7676):
        pass
    assert list_claims(tmp_path) == []


def test_record_removed_on_exception(tmp_path):
    with pytest.raises(ValueError):
        with gpu_claim(key=KEY, directory=tmp_path, usable_mb=7676):
            raise ValueError("boom")
    assert list_claims(tmp_path) == []


def test_different_keys_do_not_collide(tmp_path):
    with gpu_claim(key=KEY, directory=tmp_path, usable_mb=7676):
        with gpu_claim(key="other-uuid", directory=tmp_path, usable_mb=7676):
            assert len(list_claims(tmp_path)) == 2


def test_wait_blocks_until_there_is_room(tmp_path, monkeypatch):
    """`wait` polls capacity now rather than blocking on flock, because the
    mutex is released the instant acquire returns."""
    monkeypatch.setattr("gpuqueue.claim.WAIT_POLL_S", 0.01)
    holder = lg.acquire(KEY, vram_mb=7000, owner="a", cmd=[],
                        directory=tmp_path, usable_mb=7676)
    calls = []
    real_sleep = __import__("time").sleep

    def freeing_sleep(s):
        calls.append(s)
        if len(calls) == 2:
            lg.remove(holder)
        real_sleep(0)

    monkeypatch.setattr("gpuqueue.claim.time.sleep", freeing_sleep)
    with gpu_claim(key=KEY, directory=tmp_path, vram_mb=1000,
                   usable_mb=7676, wait=True) as c:
        assert c.vram_mb == 1000


def test_release_stale_removes_dead_pid_records(tmp_path):
    lg.write_record(lg.Record(
        path=lg.ledger_dir(KEY, tmp_path) / "4000000.dead.json",
        pid=4000000, usage_pid=4000000, vram_mb=512, owner="ghost",
        cmd=["x"], started_at="2026-08-10T00:00:00Z", key=KEY))
    assert [r["owner"] for r in release_stale(tmp_path)] == ["ghost"]
    assert list_claims(tmp_path) == []


def test_release_stale_keeps_live_records(tmp_path):
    with gpu_claim(key=KEY, directory=tmp_path, usable_mb=7676):
        assert release_stale(tmp_path) == []
        assert len(list_claims(tmp_path)) == 1


def test_read_claim_returns_none_on_garbage(tmp_path):
    p = tmp_path / "bad.lock.json"
    p.write_text("{not json")
    assert read_claim(p) is None


def test_default_usable_mb_holds_back_a_reserve(monkeypatch):
    monkeypatch.setattr("gpuqueue.claim.total_vram_mb", lambda: 8188)
    assert default_usable_mb() == 8188 - lg.DEFAULT_RESERVE_MB


def test_default_usable_mb_is_none_when_the_card_cannot_be_queried(monkeypatch):
    monkeypatch.setattr("gpuqueue.claim.total_vram_mb", lambda: None)
    assert default_usable_mb() is None
```

Keep `test_pid_alive_*` and the four `test_job_orphaned_*` tests as they are.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_claim.py -q`
Expected: FAIL — `ImportError: cannot import name 'default_usable_mb'`

- [ ] **Step 3: Rewrite the claim surface**

In `src/gpuqueue/claim.py`, replace the imports, `ClaimBusy`, `_paths`, `list_claims`, `release_stale` and `gpu_claim` with:

```python
import getpass
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

from . import ledger
from .gpuid import gpu_key, total_vram_mb
from .ledger import ClaimBusy          # re-exported: callers import it here
from .procs import pid_alive           # re-exported for the same reason

DEFAULT_CLAIM_DIR = "/var/lock/gpu"
WAIT_POLL_S = 0.5


def default_usable_mb() -> int | None:
    """What a standalone `gpu-claim` may admit against.

    None when the card cannot be queried, which `ledger.fits` turns into
    exclusive-only admission -- degraded, and the same posture preflight
    already takes when it cannot enumerate the card.
    """
    total = total_vram_mb()
    return None if total is None else total - ledger.DEFAULT_RESERVE_MB


def list_claims(directory: Path | None = None) -> list[tuple[Path, dict]]:
    d = Path(directory) if directory else claim_dir()
    return [(r.path, r.to_dict()) for r in ledger.all_records(d)]


def release_stale(directory: Path | None = None) -> list[dict]:
    """Remove records whose owning pid is gone. Returns what it freed."""
    d = Path(directory) if directory else claim_dir()
    released = []
    for r in ledger.all_records(d):
        if not pid_alive(r.pid):
            released.append(r.to_dict())
            ledger.remove(r)
    return released


@contextmanager
def gpu_claim(key: str | None = None, owner: str | None = None,
              cmd: list[str] | None = None, wait: bool = False,
              directory: Path | None = None, vram_mb: int | None = None,
              usable_mb: int | None = None, own_usage: bool = True):
    """Hold a share of the card. `vram_mb=None` means the whole of it.

    `own_usage=False` is for the runner, which takes the card before the
    job process exists and fills the usage pid in after launch.

    `wait` polls capacity rather than blocking on flock: the mutex is
    released the instant `acquire` returns, so there is no longer a kernel
    queue to wait in.
    """
    d = Path(directory) if directory else claim_dir()
    key = key or gpu_key()
    if usable_mb is None:
        usable_mb = default_usable_mb()
    while True:
        try:
            rec = ledger.acquire(
                key, vram_mb=vram_mb, owner=owner or _default_owner(),
                cmd=cmd, directory=d, usable_mb=usable_mb,
                usage_pid=os.getpid() if own_usage else None)
            break
        except ClaimBusy:
            if not wait:
                raise
            time.sleep(WAIT_POLL_S)
    try:
        yield rec
    finally:
        ledger.remove(rec)
```

Keep `claim_dir`, `read_claim`, `job_orphaned` and `_default_owner` exactly as they are.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_claim.py -q`
Expected: PASS

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: `tests/test_cli_claim.py` and `tests/test_runner.py` may fail where they assert on the yielded dict or call `gpu_claim` without `usable_mb`. Fix those call sites to use attribute access; do not change behaviour to accommodate a test.

- [ ] **Step 6: Commit**

```bash
git add src/gpuqueue/claim.py tests/test_claim.py tests/test_cli_claim.py tests/test_runner.py
git commit -m "feat: gpu_claim admits against ledger capacity

Undeclared claims stay exclusive, so every existing caller behaves as it
did. The context manager now yields a ledger.Record rather than a dict --
the runner needs to set the usage pid on it after the job launches, and
the protocol design.md pins is the file, not this object."
```

---

### Task 10: Preflight refuses on unledgered, not foreign

**Files:**
- Modify: `src/gpuqueue/preflight.py:53-100`
- Test: `tests/test_preflight.py`

**Interfaces:**
- Consumes: Tasks 8 and 9.
- Produces: `preflight.unledgered_processes(directory=None) -> list[dict]`; `preflight.preflight(allow=None, directory=None)` unchanged in signature-compatible shape. `foreign_processes` stays as an alias so nothing that imports it breaks.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_preflight.py
from gpuqueue import ledger as lg

KEY = "4b8f2c1a-0000-0000-0000-000000000001"


def _record(tmp_path, pid, usage_pid, vram_mb=512):
    rec = lg.Record(path=lg.ledger_dir(KEY, tmp_path) / f"{pid}.aaa.json",
                    pid=pid, usage_pid=usage_pid, vram_mb=vram_mb,
                    owner="co-tenant", cmd=["python", "t.py"],
                    started_at="2026-08-10T00:00:00Z", key=KEY)
    lg.write_record(rec)
    return rec


def test_a_ledgered_co_tenant_is_not_contention(tmp_path, monkeypatch):
    """Sharing is the point. A declared holder's process must not read as
    an intruder, or no second job can ever start."""
    _record(tmp_path, os.getpid(), os.getpid())
    monkeypatch.setattr(pf, "compute_apps",
                        lambda: [{"pid": 5150, "used_mb": 400, "name": "co.py"}])
    monkeypatch.setattr(pf, "descendants",
                        lambda pid: {5150} if pid == os.getpid() else set())
    pf.preflight(directory=tmp_path)


def test_an_unledgered_process_still_refuses(tmp_path, monkeypatch):
    monkeypatch.setattr(pf, "compute_apps",
                        lambda: [{"pid": 4321, "used_mb": 900, "name": "train.py"}])
    monkeypatch.setattr(pf, "descendants", lambda pid: set())
    with pytest.raises(PreflightFailed) as e:
        pf.preflight(directory=tmp_path)
    assert "4321" in str(e.value) and "train.py" in str(e.value)


def test_a_dead_holders_record_does_not_shelter_anyone(tmp_path, monkeypatch):
    _record(tmp_path, 4000000, 4000000)
    monkeypatch.setattr(pf, "compute_apps",
                        lambda: [{"pid": 4321, "used_mb": 900, "name": "t.py"}])
    monkeypatch.setattr(pf, "descendants", lambda pid: {4321})
    with pytest.raises(PreflightFailed):
        pf.preflight(directory=tmp_path)
```

Add `import os` and `from gpuqueue.preflight import PreflightFailed` if not already imported at the top of the file.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_preflight.py -q -k "ledgered or shelter"`
Expected: FAIL — `TypeError: preflight() got an unexpected keyword argument 'directory'`

- [ ] **Step 3: Implement**

Replace `own_pids`, `_foreign`, `foreign_processes` and `preflight` in `src/gpuqueue/preflight.py` with:

```python
from . import ledger
from .claim import claim_dir, list_claims          # noqa: F401 (list_claims
                                                   # kept for callers)
from .procs import descendants, pid_alive


def own_pids(directory=None) -> set[int]:
    """Every pid the claim protocol accounts for, plus this process's own.

    Kept because `reaper` and callers outside this package use it. The
    finer question -- which *record* owns a pid -- is `ledger.attribute`.
    """
    pids = {os.getpid(), os.getppid()} | descendants(os.getpid())
    d = Path(directory) if directory else claim_dir()
    for rec in ledger.live_records(ledger.all_records(d)):
        pids.add(rec.pid)
        if rec.usage_pid is not None:
            pids.add(rec.usage_pid)
            pids.update(descendants(rec.usage_pid))
        else:
            pids.update(descendants(rec.pid))
    return pids


def unledgered_processes(allow: set[int] | None = None,
                         directory=None) -> list[dict]:
    """CUDA processes no live ledger record accounts for.

    This replaces "foreign", which was the right question when the card
    admitted one job: with sharing, a declared co-tenant's process is
    someone else's and entirely legitimate. What is still contention is a
    process nobody has claimed capacity for.
    """
    apps = compute_apps()
    if apps is None:
        return []
    d = Path(directory) if directory else claim_dir()
    records = ledger.live_records(ledger.all_records(d))
    _, unledgered = ledger.attribute(apps, records)
    exempt = set(allow or set()) | {os.getpid(), os.getppid()}
    exempt |= descendants(os.getpid())
    return [a for a in unledgered if a["pid"] not in exempt]


# The old name, kept so nothing importing it breaks.
foreign_processes = unledgered_processes


def preflight(allow: set[int] | None = None, directory=None) -> None:
    apps = compute_apps()
    if apps is None:
        print("gpu-claim: warning: cannot enumerate CUDA processes on this "
              "box; proceeding on the advisory lock alone", file=sys.stderr)
        return
    d = Path(directory) if directory else claim_dir()
    records = ledger.live_records(ledger.all_records(d))
    _, unledgered = ledger.attribute(apps, records)
    exempt = set(allow or set()) | {os.getpid(), os.getppid()}
    exempt |= descendants(os.getpid())
    stray = [a for a in unledgered if a["pid"] not in exempt]
    if stray:
        lines = [f"  pid {a['pid']:>7}  {a['used_mb'] or '?'} MiB  {a['name']}"
                 for a in stray]
        raise PreflightFailed(
            "CUDA processes hold this GPU with no claim on it:\n"
            + "\n".join(lines))
```

Add `from pathlib import Path` to the imports.

Note `preflight` still calls `compute_apps()` exactly once — `test_preflight_queries_the_card_once` guards that, and the reason is unchanged: two queries let the visibility check and the contention check disagree about what they saw.

**Two stubs elsewhere must gain the new keyword**, or they raise `TypeError` the moment Task 13 starts passing it:

- `tests/test_runner.py:34` — `monkeypatch.setattr(rn, "preflight", lambda allow=None: None)` becomes `lambda allow=None, directory=None: None`.
- Any `pf.preflight` stub in `tests/test_cli_claim.py` likewise.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_preflight.py -q`
Expected: PASS. Existing tests that stub `pf.own_pids` still pass because `preflight` no longer calls it; if any assert on the old "foreign CUDA processes" wording, update the string to match the new message.

- [ ] **Step 5: Commit**

```bash
git add src/gpuqueue/preflight.py tests/test_preflight.py
git commit -m "feat: preflight refuses on unledgered, not foreign

With sharing, a declared co-tenant's process is legitimate and 'foreign'
is the wrong question. What is still contention is a process nobody has
claimed capacity for."
```

---

### Task 11: The orphan sweep uses attribution

**Files:**
- Modify: `src/gpuqueue/reaper.py:47-56` (`kill_orphan_cuda`), `src/gpuqueue/reaper.py:83-103` (`reap`)
- Test: `tests/test_reaper.py`

**Interfaces:**
- Consumes: Tasks 8 and 9.
- Produces: `reaper.kill_orphan_cuda(protect, records, apps) -> list[int]` — now taking the records and the already-fetched apps rather than calling `compute_apps` itself, so `reap` makes one `nvidia-smi` call that both the sweep and Task 12's watchdog share.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_reaper.py
def test_a_ledgered_co_tenants_process_is_not_an_orphan(q, tmp_path, monkeypatch):
    """The whole point of sharing: a declared holder's CUDA process is
    someone else's job, not debris."""
    from gpuqueue import ledger as lg
    cfg = RunnerConfig(queue_root=q.root, kill_orphan_cuda=True,
                       claim_dir=tmp_path)
    lg.write_record(lg.Record(
        path=lg.ledger_dir("k", tmp_path) / f"{_os.getpid()}.aaa.json",
        pid=_os.getpid(), usage_pid=_os.getpid(), vram_mb=512, owner="co",
        cmd=[], started_at="2026-08-10T00:00:00Z", key="k"))
    monkeypatch.setattr(rp, "compute_apps",
                        lambda: [{"pid": 5150, "used_mb": 400, "name": "co.py"}])
    monkeypatch.setattr(rp, "descendants",
                        lambda pid: {5150} if pid == _os.getpid() else set())
    monkeypatch.setattr(rp, "_kill", lambda pid: pytest.fail("killed a co-tenant"))
    assert reap(q, cfg)["killed_pids"] == []


def test_an_unledgered_process_is_still_killed(q, tmp_path, monkeypatch):
    cfg = RunnerConfig(queue_root=q.root, kill_orphan_cuda=True,
                       claim_dir=tmp_path)
    monkeypatch.setattr(rp, "compute_apps",
                        lambda: [{"pid": 4321, "used_mb": 900, "name": "x.py"}])
    monkeypatch.setattr(rp, "descendants", lambda pid: set())
    killed = []
    monkeypatch.setattr(rp, "_kill", lambda pid: killed.append(pid) or True)
    assert reap(q, cfg)["killed_pids"] == [4321]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_reaper.py -q -k "co_tenant or unledgered"`
Expected: FAIL — the co-tenant's process is killed, because `own_pids` exempts by pid set rather than by record ownership and the stubbed `descendants` is not consulted.

- [ ] **Step 3: Implement**

In `src/gpuqueue/reaper.py`, change the imports and `kill_orphan_cuda`:

```python
from . import ledger
from .claim import release_stale, claim_dir
from .config import RunnerConfig
from .preflight import compute_apps, own_pids
from .procs import descendants, pid_alive
from .queue import QueueRoot


def kill_orphan_cuda(protect: set[int], records: list, apps: list[dict]) -> list[int]:
    """Kill CUDA processes no live claim accounts for.

    Takes `apps` rather than fetching them so one nvidia-smi call serves
    both this and the VRAM watchdog; takes `records` so ownership is
    decided by the same `ledger.attribute` preflight uses, rather than by
    a second, subtly different pid set.
    """
    _, unledgered = ledger.attribute(apps, records)
    exempt = set(protect) | own_pids()
    killed = []
    for app in unledgered:
        if app["pid"] not in exempt and _kill(app["pid"]):
            killed.append(app["pid"])
    return killed
```

In `reap`, fetch the apps and records once:

```python
    stale = release_stale(cfg.claim_dir)
    requeued, failed = requeue_orphans(queue, active_ids)
    killed = []
    apps = None
    records = []
    if include_orphan_cuda:
        apps = compute_apps()
        d = cfg.claim_dir if cfg.claim_dir else claim_dir()
        records = ledger.live_records(ledger.all_records(d))
        if apps is not None and cfg.kill_orphan_cuda:
            protect = {s.pid for s in queue.list_state("running") if s.pid}
            killed = kill_orphan_cuda(protect, records, apps)
    cleaned = clean_partials(queue)
    return {"stale_claims": stale, "requeued": requeued, "failed": failed,
            "killed_pids": killed, "cleaned_paths": cleaned}
```

`apps is None` still means "cannot see the list", and killing blind stays worse than leaking.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_reaper.py -q`
Expected: PASS. `test_kills_orphan_cuda_when_enabled` and `test_does_not_kill_pids_of_running_jobs` need `rp.descendants` stubbed to `lambda pid: set()` since they now go through attribution; add that to the autouse `no_gpu_calls` fixture.

- [ ] **Step 5: Commit**

```bash
git add src/gpuqueue/reaper.py tests/test_reaper.py
git commit -m "feat: the orphan sweep decides ownership by ledger record

One nvidia-smi call now serves the sweep and (next) the watchdog, and
ownership uses the same attribute() preflight does -- two subtly
different pid sets is how a legitimate job gets killed."
```

---

### Task 12: The VRAM watchdog

**Files:**
- Modify: `src/gpuqueue/reaper.py` (add `check_vram`, `_kill_tree`; extend `reap`)
- Test: `tests/test_reaper.py`

**Interfaces:**
- Consumes: Tasks 3, 8, 11.
- Produces: `reaper.WATCHDOG_STRIKES = 2`; `reaper.check_vram(records, apps, strikes) -> list[dict]` where each conviction is `{"owner", "declared", "used", "usage_pid", "record"}`; `reap(..., vram_strikes: dict[str, int] | None = None)` returning `"convicted": [...]` in its dict. Task 13 owns the `strikes` dict on the `Runner` and reads `convicted`.

**Why the strike count:** PyTorch's caching allocator moves its high-water mark in steps, so a single sample over the line is not evidence of a persistent overage. Two consecutive sweeps at the default 60s interval means conviction takes up to two minutes — the victim has already OOMed by then. That is accepted and it is the point: **this is attribution, not prevention** (§6). What it buys is that the over-user dies naming its own declaration instead of two jobs sharing a bare CUDA OOM.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_reaper.py
def _rec(tmp_path, name, usage_pid, vram_mb, owner="gpuq:j1"):
    from gpuqueue import ledger as lg
    return lg.Record(path=lg.ledger_dir("k", tmp_path) / name, pid=_os.getpid(),
                     usage_pid=usage_pid, vram_mb=vram_mb, owner=owner,
                     cmd=["python", "t.py"], started_at="2026-08-10T00:00:00Z",
                     key="k")


def test_one_sweep_over_the_line_does_not_convict(tmp_path, monkeypatch):
    """The caching allocator's high-water mark moves in steps; one sample
    over is not evidence of a persistent overage."""
    monkeypatch.setattr(rp, "descendants", lambda pid: set())
    r = _rec(tmp_path, "1.a.json", 500, 512)
    strikes = {}
    apps = [{"pid": 500, "used_mb": 3070, "name": "t.py"}]
    assert rp.check_vram([r], apps, strikes) == []
    assert strikes[str(r.path)] == 1


def test_two_consecutive_sweeps_convict(tmp_path, monkeypatch):
    monkeypatch.setattr(rp, "descendants", lambda pid: set())
    r = _rec(tmp_path, "1.a.json", 500, 512)
    strikes = {}
    apps = [{"pid": 500, "used_mb": 3070, "name": "t.py"}]
    rp.check_vram([r], apps, strikes)
    (guilty,) = rp.check_vram([r], apps, strikes)
    assert guilty["declared"] == 512 and guilty["used"] == 3070
    assert guilty["owner"] == "gpuq:j1" and guilty["usage_pid"] == 500


def test_a_sweep_back_under_the_line_clears_the_strike(tmp_path, monkeypatch):
    monkeypatch.setattr(rp, "descendants", lambda pid: set())
    r = _rec(tmp_path, "1.a.json", 500, 512)
    strikes = {}
    rp.check_vram([r], [{"pid": 500, "used_mb": 3070, "name": "t"}], strikes)
    rp.check_vram([r], [{"pid": 500, "used_mb": 400, "name": "t"}], strikes)
    assert strikes == {}


def test_an_exclusive_holder_is_never_over(tmp_path, monkeypatch):
    monkeypatch.setattr(rp, "descendants", lambda pid: set())
    r = _rec(tmp_path, "1.a.json", 500, None)
    apps = [{"pid": 500, "used_mb": 8000, "name": "t.py"}]
    rp.check_vram([r], apps, {})
    assert rp.check_vram([r], apps, {}) == []


def test_a_holders_children_count_toward_its_declaration(tmp_path, monkeypatch):
    """A trainer's dataloader workers hold VRAM under the same record."""
    monkeypatch.setattr(rp, "descendants",
                        lambda pid: {501, 502} if pid == 500 else set())
    r = _rec(tmp_path, "1.a.json", 500, 512)
    apps = [{"pid": 500, "used_mb": 200, "name": "t"},
            {"pid": 501, "used_mb": 200, "name": "w"},
            {"pid": 502, "used_mb": 200, "name": "w"}]
    strikes = {}
    rp.check_vram([r], apps, strikes)
    (guilty,) = rp.check_vram([r], apps, strikes)
    assert guilty["used"] == 600


def test_broken_attribution_convicts_nobody(tmp_path, monkeypatch):
    """Under MPS nvidia-smi reports the server, not its clients, so every
    process looks unledgered. That is a broken measurement, not a box full
    of intruders."""
    monkeypatch.setattr(rp, "descendants", lambda pid: set())
    r = _rec(tmp_path, "1.a.json", 500, 512)
    apps = [{"pid": 9999, "used_mb": 8000, "name": "nvidia-cuda-mps-server"}]
    strikes = {}
    rp.check_vram([r], apps, strikes)
    assert rp.check_vram([r], apps, strikes) == []
    assert strikes == {}


def test_a_departed_holder_stops_accruing_strikes(tmp_path, monkeypatch):
    monkeypatch.setattr(rp, "descendants", lambda pid: set())
    r = _rec(tmp_path, "1.a.json", 500, 512)
    strikes = {}
    rp.check_vram([r], [{"pid": 500, "used_mb": 3070, "name": "t"}], strikes)
    assert rp.check_vram([], [], strikes) == []
    assert strikes == {}


def test_enforce_vram_off_convicts_nobody(q, tmp_path, monkeypatch):
    cfg = RunnerConfig(queue_root=q.root, claim_dir=tmp_path,
                       kill_orphan_cuda=False, enforce_vram=False)
    monkeypatch.setattr(rp, "compute_apps",
                        lambda: [{"pid": 500, "used_mb": 8000, "name": "t"}])
    monkeypatch.setattr(rp, "descendants", lambda pid: set())
    strikes = {}
    reap(q, cfg, vram_strikes=strikes)
    assert reap(q, cfg, vram_strikes=strikes)["convicted"] == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_reaper.py -q -k "convict or over_the_line or exclusive_holder or declaration"`
Expected: FAIL — `AttributeError: module 'gpuqueue.reaper' has no attribute 'check_vram'`

- [ ] **Step 3: Implement**

```python
# append to src/gpuqueue/reaper.py
WATCHDOG_STRIKES = 2


def _kill_tree(pid: int) -> bool:
    """Kill a holder's whole tree, by pid rather than by process group.

    killpg would be shorter and is wrong here: a `gpu-claim` launched from
    a script shares its group, so the group is not reliably the holder's
    own. Enumerating descendants kills exactly what the record is charged
    for and nothing else.
    """
    ok = False
    for p in {pid} | descendants(pid):
        ok = _kill(p) or ok
    return ok


def check_vram(records: list, apps: list[dict],
               strikes: dict[str, int]) -> list[dict]:
    """Convict holders using more than they declared.

    Attribution, not prevention. The victim of an overage OOMs in
    milliseconds and this convicts in up to two sweeps -- what it buys is
    that the failure is legible afterwards, instead of two jobs sharing a
    bare CUDA OOM with nothing to say whose fault it was.
    """
    owned, unledgered = ledger.attribute(apps, records)
    if unledgered and not owned and any(r.usage_pid for r in records):
        # Every visible process is unattributable while records claim to
        # own trees: the measurement is broken, not the box overrun. Under
        # MPS nvidia-smi reports the server rather than its clients, which
        # looks exactly like this. Convicting here would kill the box's
        # own work.
        strikes.clear()
        return []

    convicted, seen = [], set()
    for rec in records:
        if rec.vram_mb is None:
            continue  # declared the whole card, so it cannot exceed it
        # Keyed by full path, matching ledger.attribute. A bare filename
        # is not unique across <key>.lock.d directories, and these strikes
        # persist across sweeps -- a collision would charge one holder's
        # strikes to another and kill the wrong job.
        key = str(rec.path)
        seen.add(key)
        used = ledger.used_mb(owned.get(key, []))
        if used <= rec.vram_mb:
            strikes.pop(key, None)
            continue
        strikes[key] = strikes.get(key, 0) + 1
        if strikes[key] >= WATCHDOG_STRIKES:
            strikes.pop(key, None)
            convicted.append({"owner": rec.owner, "declared": rec.vram_mb,
                              "used": used, "usage_pid": rec.usage_pid,
                              "record": rec.name})
    for key in list(strikes):
        if key not in seen:
            strikes.pop(key)  # the holder is gone; its strikes go with it
    return convicted
```

Extend `reap` — replace the block added in Task 11 with:

```python
    stale = release_stale(cfg.claim_dir)
    requeued, failed = requeue_orphans(queue, active_ids)
    killed, convicted = [], []
    if include_orphan_cuda:
        apps = compute_apps()
        d = cfg.claim_dir if cfg.claim_dir else claim_dir()
        records = ledger.live_records(ledger.all_records(d))
        if apps is not None:
            if cfg.kill_orphan_cuda:
                protect = {s.pid for s in queue.list_state("running") if s.pid}
                killed = kill_orphan_cuda(protect, records, apps)
            if cfg.enforce_vram and vram_strikes is not None:
                convicted = check_vram(records, apps, vram_strikes)
                for c in convicted:
                    if c["usage_pid"]:
                        _kill_tree(c["usage_pid"])
    cleaned = clean_partials(queue)
    return {"stale_claims": stale, "requeued": requeued, "failed": failed,
            "killed_pids": killed, "cleaned_paths": cleaned,
            "convicted": convicted}
```

and change the signature to `def reap(queue, cfg, active_ids=None, include_orphan_cuda=True, vram_strikes=None) -> dict:`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_reaper.py -q`
Expected: PASS. Existing tests calling `reap(q, cfg)` still work — `vram_strikes=None` disables the watchdog.

- [ ] **Step 5: Commit**

```bash
git add src/gpuqueue/reaper.py tests/test_reaper.py
git commit -m "feat: VRAM watchdog convicts a holder over its declaration

Two consecutive sweeps, because the caching allocator's high-water mark
moves in steps. This is attribution rather than prevention: the victim
has already OOMed, but the over-user now dies naming its own declaration
instead of both jobs sharing an anonymous CUDA OOM."
```

---

### Task 13: Runner admission

**Files:**
- Modify: `src/gpuqueue/runner.py:38-44` (`Active`), `:226-230` (`_capacity`), `:264-291` (`_take_card`), `:293-326` (`_launch`), `:172-191` (`_reap`), `:411-418` (`_describe_failure`)
- Test: `tests/test_runner.py`

**Interfaces:**
- Consumes: Tasks 3, 4, 9, 12.
- Produces: `Runner._usable_mb() -> int | None` (cached), `Active.claim_record`, `Active.started_mono`, `Runner._vram_strikes`, `Runner._convicted: dict[str, dict]`. Task 14 uses `started_mono` and `_convicted`.

- [ ] **Step 1: Write the failing test**

First extend the file's existing `submit` helper (`tests/test_runner.py:38`) to carry a declaration:

```python
def submit(r, sha, job_id, cmd, lane="cpu", artifacts=(), timeout_s=30,
           vram_mb=None):
    r.queue.submit(JobSpec.from_dict(dict(
        id=job_id, lane=lane, project="p", commit=sha, branch="main",
        cmd=list(cmd), artifacts=list(artifacts), timeout_s=timeout_s,
        vram_mb=vram_mb)))
```

Then append, using the file's existing `env` fixture (`tests/test_runner.py:23`), which yields `(runner, sha)`:

```python
def test_gpu_lane_admits_up_to_gpu_max_jobs(env):
    r, _ = env
    r.cfg.gpu_max_jobs = 2
    assert r._capacity("gpu") == 2


def test_usable_mb_holds_back_the_reserve(env):
    r, _ = env
    r.cfg.gpu_vram_mb = 8188
    r.cfg.gpu_vram_reserve_mb = 512
    assert r._usable_mb() == 7676


def test_usable_mb_asks_the_card_when_unconfigured(env, monkeypatch):
    r, _ = env
    r.cfg.gpu_vram_mb = None
    monkeypatch.setattr(rn, "total_vram_mb", lambda: 8188)
    assert r._usable_mb() == 8188 - r.cfg.gpu_vram_reserve_mb


def test_usable_mb_is_queried_once(env, monkeypatch):
    """Otherwise this is an nvidia-smi subprocess on every admit, on the
    single loop that also polls every running job."""
    r, _ = env
    r.cfg.gpu_vram_mb = None
    calls = []
    monkeypatch.setattr(rn, "total_vram_mb", lambda: calls.append(1) or 8188)
    r._usable_mb()
    r._usable_mb()
    assert len(calls) == 1


def test_a_declaration_bigger_than_the_card_fails_rather_than_queues(env):
    """A permanent condition. Leaving it pending queues it forever, which
    is the mistake `_take_card` already avoids for a box with no GPU."""
    r, sha = env
    r.cfg.gpu_vram_mb = 1024
    r.cfg.gpu_vram_reserve_mb = 512
    submit(r, sha, "j1", ["true"], lane="gpu", vram_mb=4096)
    r.admit()
    state, got = r.queue.find("j1")
    assert state == "failed"
    assert "never be admitted" in got.error


def test_launch_charges_the_record_to_the_job(env, monkeypatch):
    """The card is taken before the job process exists, so the record is
    only attributable once there is a pid to charge it to."""
    from gpuqueue import ledger as lg
    r, sha = env
    r.cfg.gpu_vram_mb = 8188
    monkeypatch.setattr(r, "_prepare_workdir",
                        lambda spec, project: r.queue.work_dir(spec.id))
    (r.queue.work_dir("j1")).mkdir(parents=True, exist_ok=True)
    submit(r, sha, "j1", ["sleep", "5"], lane="gpu", vram_mb=512)
    assert r.admit() == ["j1"]
    try:
        (record,) = lg.all_records(r.cfg.claim_dir)
        assert record.usage_pid == r.active["j1"].running.pid
        assert record.vram_mb == 512
    finally:
        r.shutdown()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_runner.py -q -k "gpu_max_jobs or usable_mb or bigger_than_the_card"`
Expected: FAIL — `AttributeError: 'Runner' object has no attribute '_usable_mb'`

- [ ] **Step 3: Implement**

Imports in `runner.py`:

```python
from . import ledger
from .claim import gpu_claim, ClaimBusy
from .gpuid import gpu_key, total_vram_mb, GpuIdError
```

`Active` gains two fields:

```python
@dataclass
class Active:
    running: RunningJob
    project: ProjectConfig
    workdir: Path
    claim_cm: object | None = None       # entered gpu_claim, released on settle
    claim_record: object | None = None   # its ledger.Record, for usage_pid
    started_mono: float = 0.0            # for §7's victim retry
```

`__init__` gains:

```python
        self._usable_mb_cache: int | None = None
        self._usable_asked = False
        # record name -> consecutive sweeps over its declaration
        self._vram_strikes: dict[str, int] = {}
        # job id -> the conviction that killed it, so _describe_failure can
        # say "declared 512 MiB, using 3070" instead of "exit -9"
        self._convicted: dict[str, dict] = {}
        self._last_conviction: float | None = None
```

Capacity and usable memory:

```python
    def _usable_mb(self) -> int | None:
        """Cached: a card's total does not change, and this would otherwise
        run nvidia-smi on every admit, on the loop that also polls jobs."""
        if not self._usable_asked:
            self._usable_asked = True
            total = self.cfg.gpu_vram_mb
            if total is None:
                total = total_vram_mb()
            self._usable_mb_cache = (None if total is None
                                     else total - self.cfg.gpu_vram_reserve_mb)
        return self._usable_mb_cache

    def _capacity(self, lane: str) -> int:
        limit = (self.cfg.cpu_slots if lane == "cpu"
                 else self.cfg.gpu_max_jobs)
        in_lane = sum(1 for a in self.active.values()
                      if a.running.spec.lane == lane)
        return limit - in_lane
```

`_take_card` — its first line changes from `preflight()` to:

```python
            # The runner's configured claim dir, not $GPU_CLAIM_DIR. These
            # were already two different answers to "where are the
            # claims?"; now that preflight decides contention by reading
            # them, disagreeing means a co-tenant reads as an intruder.
            preflight(directory=self.cfg.claim_dir)
```

Then, after the `gpu_key()` block and before building the claim:

```python
        usable = self._usable_mb()
        if ledger.exceeds_capacity(spec.vram_mb, usable):
            # Permanent, so failing beats queueing forever -- the same call
            # this function already makes for a box with no GPU.
            self._fail_pending(
                spec, f"declared {spec.vram_mb} MiB but only {usable} MiB is "
                      "usable on this card; it can never be admitted")
            return None
        cm = gpu_claim(key=key, owner=f"gpuq:{spec.id}", cmd=spec.cmd,
                       wait=False, directory=self.cfg.claim_dir,
                       vram_mb=spec.vram_mb, usable_mb=usable,
                       own_usage=False)
        try:
            record = cm.__enter__()
        except ClaimBusy as e:
            log.info("%s waiting: %s", spec.id, e)
            return None
        return cm, record
```

Every `_take_card` caller changes from `claim_cm = self._take_card(spec)` to:

```python
            claim_cm = claim_record = None
            if spec.lane == "gpu":
                taken = self._take_card(spec)
                if taken is None:
                    continue
                claim_cm, claim_record = taken
```

and `_launch(claimed, project, claim_cm)` becomes `_launch(claimed, project, claim_cm, claim_record)`. In `_launch`, after `self.active[spec.id] = Active(...)`:

```python
        self.active[spec.id] = Active(running=running, project=project,
                                      workdir=workdir, claim_cm=claim_cm,
                                      claim_record=claim_record,
                                      started_mono=time.monotonic())
        if claim_record is not None:
            # The card was taken before this process existed. Charging the
            # record to it now is what lets the watchdog and the orphan
            # sweep tell this job's VRAM from a co-tenant's.
            ledger.set_usage_pid(claim_record, running.pid)
```

`_reap` records convictions:

```python
        result = reap(self.queue, self.cfg, active_ids=set(self.active),
                      include_orphan_cuda=sweep,
                      vram_strikes=self._vram_strikes)
        for c in result.get("convicted", []):
            self._last_conviction = time.monotonic()
            log.warning("killed %s: declared %s MiB, using %s MiB",
                        c["owner"], c["declared"], c["used"])
            if c["owner"].startswith("gpuq:"):
                self._convicted[c["owner"][len("gpuq:"):]] = c
```

`_describe_failure` gains a first branch:

```python
    def _describe_failure(self, spec: JobSpec, result: JobResult) -> str:
        guilty = self._convicted.pop(spec.id, None)
        if guilty:
            return (f"killed for exceeding its declaration: --vram-mb "
                    f"{guilty['declared']}, actually using {guilty['used']} "
                    "MiB. Declare what the job really needs, measured as "
                    "nvidia-smi reports it. Not retried.")
        if result.timed_out:
            return (f"timeout after {spec.timeout_s}s; killed. A hung job "
                    "is a bug, not a transient — not retried.")
        if result.oom:
            return ("CUDA out of memory — a configuration error, not a "
                    f"transient; not retried.\n{result.stderr_tail}")
        return f"exit {result.exit_code}\n{result.stderr_tail}"
```

The three trailing branches are the existing body, unchanged — reproduced here so the new branch's position is unambiguous.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_runner.py -q`
Expected: PASS

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/gpuqueue/runner.py tests/test_runner.py
git commit -m "feat: admit GPU jobs against declared capacity

The GPU lane is bounded by gpu_max_jobs and by summed declarations
rather than by a literal 1. A declaration bigger than the card fails the
job instead of queueing it forever."
```

---

### Task 14: Retry the victim of a convicted co-tenant

Separable (§7). Cutting this task leaves the design correct and the victim's OOM merely unexplained; it is the difference between blame being *assignable* and blame being *acted on*.

**Files:**
- Modify: `src/gpuqueue/runner.py:360-391` (`_settle`)
- Test: `tests/test_runner.py`

**Interfaces:**
- Consumes: Task 13's `Active.started_mono`, `Runner._last_conviction`, `Runner._convicted`; `reaper.MAX_ATTEMPTS`.
- Produces: `Runner._hit_by_a_convicted_co_tenant(spec, active, result) -> bool`.

- [ ] **Step 1: Write the failing test**

These drive a real job through `admit()` and `collect()` rather than calling `_settle` with a hand-built `Active`, so the `started_mono` comparison is against a real clock. `collect()` is used instead of `tick()` because `tick()` would re-admit the requeued job in the same call and the assertion could not see it in `pending`.

```python
# append to tests/test_runner.py
OOMS = ["sh", "-c", "sleep 0.3; echo 'CUDA out of memory' >&2; exit 1"]


def _run_until_settled(r, job_id, limit=15.0):
    deadline = time.monotonic() + limit
    while job_id in r.active and time.monotonic() < deadline:
        r.collect()
        time.sleep(0.02)
    assert job_id not in r.active, "job never settled"


def test_an_oom_beside_a_convicted_co_tenant_is_retried(env, monkeypatch):
    """With sharing, 'a CUDA OOM is your own configuration error' is only
    true if the two cases can be told apart. Here they can."""
    r, sha = env
    r.cfg.gpu_vram_mb = 8188
    monkeypatch.setattr(r, "_prepare_workdir",
                        lambda spec, project: r.queue.work_dir(spec.id))
    r.queue.work_dir("j1").mkdir(parents=True, exist_ok=True)
    submit(r, sha, "j1", OOMS, lane="gpu", vram_mb=512)
    assert r.admit() == ["j1"]
    r._last_conviction = time.monotonic()   # a co-tenant convicted mid-run
    _run_until_settled(r, "j1")
    state, spec = r.queue.find("j1")
    assert state == "pending"
    assert spec.attempts == 1


def test_an_ordinary_oom_is_still_not_retried(env, monkeypatch):
    r, sha = env
    r.cfg.gpu_vram_mb = 8188
    monkeypatch.setattr(r, "_prepare_workdir",
                        lambda spec, project: r.queue.work_dir(spec.id))
    r.queue.work_dir("j1").mkdir(parents=True, exist_ok=True)
    submit(r, sha, "j1", OOMS, lane="gpu", vram_mb=512)
    r.admit()
    _run_until_settled(r, "j1")
    state, spec = r.queue.find("j1")
    assert state == "failed"
    assert "out of memory" in (spec.error or "").lower()


def test_a_conviction_before_the_job_started_does_not_excuse_it(env, monkeypatch):
    r, sha = env
    r.cfg.gpu_vram_mb = 8188
    monkeypatch.setattr(r, "_prepare_workdir",
                        lambda spec, project: r.queue.work_dir(spec.id))
    r.queue.work_dir("j1").mkdir(parents=True, exist_ok=True)
    r._last_conviction = time.monotonic()   # before the job existed
    submit(r, sha, "j1", OOMS, lane="gpu", vram_mb=512)
    r.admit()
    _run_until_settled(r, "j1")
    assert r.queue.find("j1")[0] == "failed"


def test_the_convicted_job_is_not_its_own_victim(env, monkeypatch):
    """The over-user is killed, not retried: exceeding your own
    declaration is a configuration error, the same class as an OOM."""
    r, sha = env
    r.cfg.gpu_vram_mb = 8188
    monkeypatch.setattr(r, "_prepare_workdir",
                        lambda spec, project: r.queue.work_dir(spec.id))
    r.queue.work_dir("j1").mkdir(parents=True, exist_ok=True)
    submit(r, sha, "j1", OOMS, lane="gpu", vram_mb=512)
    r.admit()
    r._last_conviction = time.monotonic()
    r._convicted["j1"] = {"declared": 512, "used": 3070, "owner": "gpuq:j1"}
    _run_until_settled(r, "j1")
    state, spec = r.queue.find("j1")
    assert state == "failed"
    assert "exceeding its declaration" in spec.error


def test_the_victim_is_retried_only_once(env, monkeypatch):
    r, sha = env
    r.cfg.gpu_vram_mb = 8188
    monkeypatch.setattr(r, "_prepare_workdir",
                        lambda spec, project: r.queue.work_dir(spec.id))
    r.queue.work_dir("j1").mkdir(parents=True, exist_ok=True)
    submit(r, sha, "j1", OOMS, lane="gpu", vram_mb=512)
    r.admit()
    r.active["j1"].running.spec.attempts = 1   # already used its retry
    r._last_conviction = time.monotonic()
    _run_until_settled(r, "j1")
    assert r.queue.find("j1")[0] == "failed"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_runner.py -q -k "convicted_co_tenant or ordinary_oom or excuse or own_victim"`
Expected: FAIL — the first test finds the job in `failed`, not `pending`.

- [ ] **Step 3: Implement**

Add to `runner.py`, and change the `reap` import to `from .reaper import reap, MAX_ATTEMPTS`:

```python
    def _hit_by_a_convicted_co_tenant(self, spec: JobSpec, active: Active,
                                      result: JobResult) -> bool:
        """An OOM this job did not cause.

        `docs/design.md` says a CUDA OOM is a configuration error and is
        never retried blindly. That stays true, and sharing does not
        weaken it -- it only adds one case where the premise is false: the
        job OOMed while the watchdog convicted a *different* holder of
        over-using the card. That is a genuine transient, so it gets the
        single retry `attempts` already bounds, and every other OOM
        behaves exactly as before.
        """
        if not result.oom or spec.id in self._convicted:
            return False
        if self._last_conviction is None:
            return False
        return (self._last_conviction > active.started_mono
                and spec.attempts < MAX_ATTEMPTS)
```

In `_settle`, immediately after `ok = result.exit_code == 0 and not result.timed_out` and *before* the artifact collection block:

```python
        if not ok and self._hit_by_a_convicted_co_tenant(spec, active, result):
            self._remove_worktree(active)
            self.queue.requeue(spec)
            log.info("%s requeued: OOMed while a co-tenant was convicted of "
                     "exceeding its declaration", spec.id)
            return
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_runner.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/gpuqueue/runner.py tests/test_runner.py
git commit -m "feat: retry a job that OOMed beside a convicted co-tenant

The one case where 'a CUDA OOM is your own configuration error' is
false. Bounded by the existing attempts counter, so it is still one
retry and no more."
```

---

### Task 15: `--vram-mb` on both CLIs

**Files:**
- Modify: `src/gpuqueue/cli_gpuq.py:34-47` (spec construction), `:180-200` (submit parser)
- Modify: `src/gpuqueue/cli_claim.py:17-29` (parser), `:62-67` (claim call)
- Test: `tests/test_cli_gpuq.py`, `tests/test_cli_claim.py`

**Interfaces:**
- Consumes: Tasks 2 and 9.
- Produces: `gpuq submit --vram-mb N`, `gpu-claim --vram-mb N`. Omitting either means the whole card.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_cli_gpuq.py
def test_submit_records_the_declaration(tmp_path, capsys):
    rc = main(["--queue-root", str(tmp_path), "submit", "--project", "p",
               "--commit", "abc", "--branch", "main", "--lane", "gpu",
               "--vram-mb", "512", "--", "python", "t.py"])
    assert rc == 0
    job_id = capsys.readouterr().out.strip()
    body = json.loads((tmp_path / "pending" / f"{job_id}.json").read_text())
    assert body["vram_mb"] == 512


def test_submit_without_a_declaration_takes_the_whole_card(tmp_path, capsys):
    main(["--queue-root", str(tmp_path), "submit", "--project", "p",
          "--commit", "abc", "--branch", "main", "--lane", "gpu",
          "--", "python", "t.py"])
    job_id = capsys.readouterr().out.strip()
    body = json.loads((tmp_path / "pending" / f"{job_id}.json").read_text())
    assert body["vram_mb"] is None


def test_submit_rejects_a_nonsense_declaration(tmp_path, capsys):
    rc = main(["--queue-root", str(tmp_path), "submit", "--project", "p",
               "--commit", "abc", "--branch", "main", "--lane", "gpu",
               "--vram-mb", "0", "--", "python", "t.py"])
    assert rc == 2
    assert "vram_mb" in capsys.readouterr().err
```

```python
# append to tests/test_cli_claim.py
def test_gpu_claim_passes_the_declaration_through(tmp_path, monkeypatch):
    seen = {}
    from contextlib import contextmanager

    @contextmanager
    def fake_claim(**kw):
        seen.update(kw)
        yield None

    monkeypatch.setattr(cc, "gpu_claim", fake_claim)
    monkeypatch.setattr(cc, "gpu_key", lambda index=0: "k")
    monkeypatch.setattr(cc, "preflight", lambda: None)
    cc.main(["--vram-mb", "512", "--", "true"])
    assert seen["vram_mb"] == 512


def test_gpu_claim_without_a_declaration_takes_the_whole_card(tmp_path, monkeypatch):
    seen = {}
    from contextlib import contextmanager

    @contextmanager
    def fake_claim(**kw):
        seen.update(kw)
        yield None

    monkeypatch.setattr(cc, "gpu_claim", fake_claim)
    monkeypatch.setattr(cc, "gpu_key", lambda index=0: "k")
    monkeypatch.setattr(cc, "preflight", lambda: None)
    cc.main(["--", "true"])
    assert seen["vram_mb"] is None
```

Match the existing import alias for `cli_claim` in that file rather than introducing `cc` if it is already bound under another name.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_cli_gpuq.py tests/test_cli_claim.py -q -k vram`
Expected: FAIL — `unrecognized arguments: --vram-mb`

- [ ] **Step 3: Implement**

In `cli_gpuq.py`, add to the `submit` parser:

```python
    s.add_argument("--vram-mb", dest="vram_mb", type=int, default=None,
                   help="VRAM this job needs, in MiB as nvidia-smi reports "
                        "it (so including the ~250 MiB CUDA context and the "
                        "allocator's high-water mark, not torch's "
                        "max_memory_allocated). Omit to take the whole card.")
```

and pass `vram_mb=args.vram_mb` into the `JobSpec(...)` construction in `_cmd_submit`.

In `cli_claim.py`, add to the parser:

```python
    p.add_argument("--vram-mb", dest="vram_mb", type=int, default=None,
                   help="VRAM this command needs, in MiB as nvidia-smi "
                        "reports it. Omit to take the whole card.")
```

and pass it through, keeping every call keyword-style so the test above can read them:

```python
        with gpu_claim(key=key, owner=args.owner, cmd=cmd, wait=args.wait,
                       vram_mb=args.vram_mb):
            return subprocess.run(cmd).returncode
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_cli_gpuq.py tests/test_cli_claim.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/gpuqueue/cli_gpuq.py src/gpuqueue/cli_claim.py tests/test_cli_gpuq.py tests/test_cli_claim.py
git commit -m "feat: gpuq submit --vram-mb and gpu-claim --vram-mb"
```

---

### Task 16: End-to-end — two GPU jobs on one card

The proof the feature works. No GPU required: `gpu_key` and `preflight` are stubbed, and the jobs are trivial commands.

**Files:**
- Test: `tests/test_runner.py` (append)

**Interfaces:**
- Consumes: every task above.
- Produces: nothing.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_runner.py
@pytest.fixture
def gpu_env(env, monkeypatch):
    """`env` with a card big enough to share and git stubbed out, so these
    assert on admission rather than on checkout."""
    r, sha = env
    r.cfg.gpu_vram_mb = 8188
    r.cfg.gpu_vram_reserve_mb = 512
    r.cfg.gpu_max_jobs = 2
    monkeypatch.setattr(r, "_prepare_workdir",
                        lambda spec, project: r.queue.work_dir(spec.id))
    try:
        yield r, sha
    finally:
        r.shutdown()   # no `sleep 5` outliving the test


def test_two_declared_gpu_jobs_run_at_once(gpu_env):
    """Issue #8, end to end: two small jobs share the card where one used
    to hold all of it."""
    from gpuqueue import ledger as lg
    r, sha = gpu_env
    for i in (1, 2):
        r.queue.work_dir(f"j{i}").mkdir(parents=True, exist_ok=True)
        submit(r, sha, f"j{i}", ["sleep", "5"], lane="gpu", vram_mb=3000)

    assert sorted(r.admit()) == ["j1", "j2"]
    assert len(r.active) == 2
    records = lg.all_records(r.cfg.claim_dir)
    assert sorted(x.vram_mb for x in records) == [3000, 3000]
    assert all(x.usage_pid for x in records), "records must be attributable"


def test_a_third_job_waits_on_gpu_max_jobs(gpu_env):
    r, sha = gpu_env
    for i in (1, 2, 3):
        r.queue.work_dir(f"j{i}").mkdir(parents=True, exist_ok=True)
        submit(r, sha, f"j{i}", ["sleep", "5"], lane="gpu", vram_mb=100)
    assert len(r.admit()) == 2
    assert r.queue.find("j3")[0] == "pending"


def test_a_fourth_job_waits_on_vram_even_under_the_job_cap(gpu_env):
    """The safety axis, distinct from the latency one."""
    r, sha = gpu_env
    r.cfg.gpu_max_jobs = 4
    for i in (1, 2, 3):
        r.queue.work_dir(f"j{i}").mkdir(parents=True, exist_ok=True)
        submit(r, sha, f"j{i}", ["sleep", "5"], lane="gpu", vram_mb=3000)
    assert len(r.admit()) == 2          # 3000 + 3000 fits 7676; a third does not
    assert r.queue.find("j3")[0] == "pending"


def test_an_undeclared_gpu_job_still_runs_alone(gpu_env):
    """The backward-compatibility guarantee, byte for byte."""
    r, sha = gpu_env
    for i in (1, 2):
        r.queue.work_dir(f"j{i}").mkdir(parents=True, exist_ok=True)
    submit(r, sha, "j1", ["sleep", "5"], lane="gpu")            # undeclared
    submit(r, sha, "j2", ["sleep", "5"], lane="gpu", vram_mb=100)
    assert r.admit() == ["j1"]
    assert r.queue.find("j2")[0] == "pending"
```

- [ ] **Step 2: Run and confirm they pass**

Run: `.venv/bin/python -m pytest tests/test_runner.py -q -k "at_once or third_job or undeclared"`
Expected: PASS. These are written last deliberately — if any fails, the defect is in Tasks 9–13, not here.

- [ ] **Step 3: Run the full suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_runner.py
git commit -m "test: two declared GPU jobs share one card, end to end"
```

---

### Task 17: Documentation

**Files:**
- Modify: `docs/design.md:32-54` (Two lanes), `:161-185` (Lock protocol), `:210-221` (Not in scope)
- Modify: `README.md` (the `gpuq-runner` row and the closing VRAM paragraph)
- Modify: `gpuq.example.toml`
- Modify: `docs/deploying.md`

**Interfaces:** none.

- [ ] **Step 1: `docs/design.md` — Two lanes**

Change the diagram's GPU line from `└──► gpu lane ── 1 slot, behind gpu-claim` to:

```
      └──► gpu lane ── admitted against declared VRAM,
                       capped at gpu_max_jobs (default 2)
```

and add after the CPU-default paragraph:

```markdown
The GPU lane admits against capacity rather than a count. A job declares
`--vram-mb`; admission sums the declarations of current holders against the
card's total less a reserve. A job that declares nothing takes the whole
card, which is what makes the change invisible to anything written before
it.

Two dimensions, doing different jobs. Declared VRAM is a **safety** budget:
it is what stops a co-tenant turning into an OOM. `gpu_max_jobs` is a
**latency** budget: VRAM alone would admit sixteen 500 MiB jobs onto an 8 GB
card, all time-slicing, each slower than it would have been queued — and
with independent submitters that cost lands on a stranger.
```

- [ ] **Step 2: `docs/design.md` — Lock protocol**

Replace the three-row table with:

```markdown
| | |
|---|---|
| Lock path | fixed directory (`$GPU_CLAIM_DIR`, default `/var/lock/gpu`), file named by GPU UUID |
| Key derivation | `torch.cuda.get_device_properties(dev).uuid`; fall back to `name-index` on builds without `.uuid` |
| Ledger | `<key>.lock.d/<pid>.<token>.json` per holder: `pid`, `usage_pid`, `vram_mb`, `owner`, `cmd`, `started_at`, `key` |
| Mutex | `<key>.lock`, `flock`ed only while reading the ledger and writing one record |

`flock` guards the accounting, not the card. It is taken for the
milliseconds needed to read the holders, decide, and rename one record into
place — never for the duration of a run. `vram_mb: null` means exclusive:
it fits only into an empty ledger and nothing fits alongside it.

One record per holder rather than one document listing them, because the
property that matters when something is stuck is that `ls` shows who is on
the card and `rm` clears one wedged holder. A shared mutated document gives
both up exactly then, since a torn write blinds every participant at once.

Enforcement stays advisory, with two additions. Preflight refuses to start
when it finds a CUDA process no live record accounts for. And a watchdog on
the reaper's sweep kills a holder using more than it declared, on two
consecutive samples. Neither prevents an overage — the victim OOMs in
milliseconds and conviction takes up to two sweeps. What they convert is an
anonymous CUDA OOM into a named one.
```

- [ ] **Step 3: `docs/design.md` — Not in scope**

Add:

```markdown
- **Hard per-process VRAM caps.** MPS (`CUDA_MPS_PINNED_DEVICE_MEM_LIMIT`)
  or MIG would prevent an overage rather than convict it. MPS needs a
  daemon; MIG is unavailable on consumer cards; and injecting
  `torch.cuda.set_per_process_memory_fraction` would end this tool's
  assuming nothing about what it runs.
- **Compute or SM-share accounting.** No portable way to declare or measure
  it. `gpu_max_jobs` is the crude substitute.
```

- [ ] **Step 4: `README.md`**

Change the `gpuq-runner` row to: *"Supervisor-managed daemon. Admits CPU jobs concurrently and GPU jobs against their declared VRAM, reaps dead claims, commits artifacts."*

Replace the closing paragraph — currently *"Per-job VRAM limits are not implemented: a job that holds the card holds all of it."* — with:

```markdown
A GPU job may declare what it needs, and jobs are admitted while their
declarations fit the card:

    gpuq submit --lane gpu --vram-mb 512 ... 

Declare nothing and you get the whole card, as before. Declarations are in
MiB as `nvidia-smi` reports them, and a job using more than it declared is
killed and told so. See `docs/specs/2026-08-10-vram-admission-design.md`.
```

- [ ] **Step 5: `gpuq.example.toml`**

Add under `[queue]`, matching the file's existing commented style:

```toml
# Capacity for the GPU lane. Omit gpu_vram_mb to ask the card.
# gpu_vram_mb = 8188
# Held back from admission: two jobs that each fit exactly still have
# their allocators fragmenting the same heap.
gpu_vram_reserve_mb = 512
# A latency budget, not a safety one. VRAM accounting alone would admit
# many small jobs that then merely time-slice, each slower than it would
# have been queued. 2 is what has been measured; raise it after measuring.
gpu_max_jobs = 2
# Kill a job using more VRAM than it declared. Turning this off does not
# make over-use safe; it makes it unattributable.
enforce_vram = true
```

- [ ] **Step 6: `docs/deploying.md`**

Add to the upgrade guidance:

```markdown
### Upgrading past the VRAM ledger

Upgrade the whole installation in one pass — `bootstrap.sh` does this, and
it is why the README argues for one shared installation and never a
vendored copy. A runner from before the ledger reads `<key>.lock.json` and
cannot see `<key>.lock.d/`, so it treats a new `gpu-claim` holder's trainer
as an orphan and kills it. The reverse direction is safe: new code reads an
old holder's file as an exclusive claim, and a pre-ledger `gpu-claim`
holding `flock` for its whole run is reported as such rather than hung on.
```

- [ ] **Step 7: Verify the docs match the code**

Run: `.venv/bin/python -m pytest tests/ -q` (the bootstrap test reads `gpuq.example.toml`)
Expected: PASS. Also confirm `grep -rn "not implemented" README.md docs/design.md` returns nothing about VRAM.

- [ ] **Step 8: Commit**

```bash
git add docs/design.md README.md gpuq.example.toml docs/deploying.md
git commit -m "docs: capacity-based GPU admission

The lock protocol section changes substantively: flock now guards the
ledger rather than the card, which is the part an independent
implementation has to match."
```

---

## Done when

- `.venv/bin/python -m pytest tests/ -q` passes.
- Two declared GPU jobs run concurrently; a third waits on `gpu_max_jobs`; an undeclared job still runs alone (Task 16).
- `gpu-claim --status` shows a ledger with declarations.
- `grep -rn "else 1" src/gpuqueue/runner.py` finds nothing.
