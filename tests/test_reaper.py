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
