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


def test_job_orphaned_when_the_runner_is_gone():
    """A live job whose runner is dead: nothing enforces its timeout and
    nothing will collect its result."""
    from gpuqueue.claim import job_orphaned
    assert job_orphaned(os.getpid(), 4000000) is True

def test_job_not_orphaned_while_its_runner_lives():
    from gpuqueue.claim import job_orphaned
    assert job_orphaned(os.getpid(), os.getpid()) is False

def test_job_not_orphaned_once_it_has_exited():
    """A dead job is the reaper's business, not an orphan."""
    from gpuqueue.claim import job_orphaned
    assert job_orphaned(4000000, 4000001) is False

def test_unknown_owner_is_not_reported_as_orphaned():
    """A spec written before runner_pid existed cannot be judged; an unknown
    owner is not evidence of an absent one."""
    from gpuqueue.claim import job_orphaned
    assert job_orphaned(os.getpid(), None) is False
