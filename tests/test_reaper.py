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
    # The job has to still be in `running` when reap() builds the protect set:
    # requeue_orphans() runs first and drops any running job whose process is
    # gone. Without this the test asserts nothing on a box where pid 4321 is
    # dead -- j1 gets requeued, protect comes out empty, and the kill is real.
    monkeypatch.setattr(rp, "pid_alive", lambda pid: True)
    monkeypatch.setattr(rp, "compute_apps",
                        lambda: [{"pid": 4321, "used_mb": 900, "name": "t.py"}])
    monkeypatch.setattr(rp, "_kill", lambda pid: pytest.fail("killed a live job"))
    assert reap(q, cfg)["killed_pids"] == []

def test_does_not_kill_when_cuda_list_is_invisible(q, monkeypatch):
    cfg = RunnerConfig(queue_root=q.root, kill_orphan_cuda=True)
    monkeypatch.setattr(rp, "compute_apps", lambda: None)
    monkeypatch.setattr(rp, "_kill", lambda pid: pytest.fail("killed blind"))
    assert reap(q, cfg)["killed_pids"] == []


def test_clean_partials_leaves_a_live_jobs_files_alone(tmp_path):
    """Reaping now runs on every tick, so it must not sweep inside a job that
    is still going -- a .part file there is that job's business, not debris."""
    from gpuqueue.reaper import clean_partials
    q = QueueRoot(tmp_path / "q"); q.ensure_dirs()
    q.submit(mkspec("live"))
    q.claim("live")                       # now in running/
    live_part = q.work_dir("live") / "checkpoint.part"
    live_part.parent.mkdir(parents=True, exist_ok=True)
    live_part.write_text("half a checkpoint")
    dead_part = q.work_dir("gone") / "leftover.part"
    dead_part.parent.mkdir(parents=True, exist_ok=True)
    dead_part.write_text("debris")

    clean_partials(q)

    assert live_part.exists(), "swept a running job's own file"
    assert not dead_part.exists()

def test_reap_can_skip_the_expensive_cuda_sweep(tmp_path, monkeypatch):
    from gpuqueue.reaper import reap
    import gpuqueue.reaper as rp
    called = []
    monkeypatch.setattr(rp, "kill_orphan_cuda", lambda protect: called.append(1) or [])
    q = QueueRoot(tmp_path / "q"); q.ensure_dirs()
    cfg = RunnerConfig(queue_root=q.root, claim_dir=tmp_path / "c")
    reap(q, cfg, include_orphan_cuda=False)
    assert called == []
    reap(q, cfg, include_orphan_cuda=True)
    assert called == [1]


# --- a direct `gpu-claim` run must survive the sweep --------------------
#
# The README advertises `gpu-claim -- <cmd>` as usable on its own. Its CUDA
# process is the *child* of the pid recorded in the claim file, and
# `preflight.own_pids` expands `_descendants()` only for the runner's own
# pid -- so that child is exempted by nothing and `kill_orphan_cuda`
# SIGKILLs a legitimate run. Every other test in this file stubs
# `own_pids` to set(), which is why the suite cannot currently see this.

import os as _os
import signal as _signal
import subprocess as _sp
import sys as _sys
import time as _time
from pathlib import Path as _Path
from gpuqueue import preflight as _pf

# Detaches before spawning its child, so the pair is NOT in pytest's own
# process tree. Without the double fork `_descendants(os.getpid())` exempts
# them and these tests pass while the bug is fully present.
_HOLDER = r'''
import json, os, subprocess, sys, time
out = sys.argv[1]
if os.fork() != 0: os._exit(0)
os.setsid()
if os.fork() != 0: os._exit(0)
child = subprocess.Popen(["sleep", "60"])
tmp = out + ".part"
with open(tmp, "w") as f:
    json.dump({"holder": os.getpid(), "child": child.pid}, f)
os.rename(tmp, out)
time.sleep(60)
'''


def _live(pid: int) -> bool:
    """Not `claim.pid_alive`: that uses os.kill(pid, 0), which succeeds on a
    zombie. A SIGKILLed child whose parent is still sleeping is exactly a
    zombie, so pid_alive would report the kill as survival."""
    try:
        status = _Path(f"/proc/{pid}/status").read_text()
    except OSError:
        return False
    for line in status.splitlines():
        if line.startswith("State:"):
            return line.split()[1] != "Z"
    return False


@pytest.fixture
def direct_claim(tmp_path, monkeypatch):
    """A detached `gpu-claim`-shaped process pair plus its claim file."""
    claim_dir = tmp_path / "claims"
    claim_dir.mkdir()
    monkeypatch.setenv("GPU_CLAIM_DIR", str(claim_dir))

    out = tmp_path / "pids.json"
    _sp.run([_sys.executable, "-c", _HOLDER, str(out)], check=True, timeout=30)
    deadline = _time.monotonic() + 10
    while not out.exists():
        if _time.monotonic() > deadline:
            pytest.fail("helper never reported its pids")
        _time.sleep(0.05)
    pids = json.loads(out.read_text())
    holder, child = pids["holder"], pids["child"]

    (claim_dir / "GPU-test.lock.json").write_text(json.dumps({
        "pid": holder, "owner": "someone", "cmd": ["python", "train.py"],
        "started_at": "2026-08-10T00:00:00Z", "key": "GPU-test"}))

    # The whole reproduction rests on these two not being in pytest's tree.
    assert holder not in _pf._descendants(_os.getpid())
    assert child not in _pf._descendants(_os.getpid())
    try:
        yield holder, child
    finally:
        for pid in (child, holder):
            try:
                _os.kill(pid, _signal.SIGKILL)
            except OSError:
                pass


def test_own_pids_covers_a_claim_holders_children(direct_claim):
    """The root cause. The claim names the gpu-claim process; the process
    actually on the card is its child."""
    holder, child = direct_claim
    own = _pf.own_pids()
    assert holder in own, "the recorded pid itself is exempt"
    assert child in own, "a claim holder's CUDA child is not exempt"


def test_does_not_kill_a_direct_gpu_claim_run(q, direct_claim, monkeypatch):
    """The consequence: a legitimate `gpu-claim` run is SIGKILLed within
    orphan_cuda_interval_s on any box where the runner is up."""
    holder, child = direct_claim
    cfg = RunnerConfig(queue_root=q.root, kill_orphan_cuda=True)
    monkeypatch.setattr(rp, "own_pids", _pf.own_pids)   # undo the autouse stub
    monkeypatch.setattr(rp, "compute_apps",
                        lambda: [{"pid": child, "used_mb": 900,
                                  "name": "train.py"}])

    result = reap(q, cfg)

    deadline = _time.monotonic() + 2
    while _live(child) and _time.monotonic() < deadline:
        _time.sleep(0.05)
    assert _live(child), "SIGKILLed a legitimate direct gpu-claim run"
    assert result["killed_pids"] == []
