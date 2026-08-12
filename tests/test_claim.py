import json
import os
import subprocess
import sys
import pytest
from gpuqueue import ledger as lg
from gpuqueue.claim import (gpu_claim, ClaimBusy, MutexTimeout, read_claim,
                            pid_alive, list_claims, release_stale,
                            default_usable_mb)

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


def _spawn_lock_ex_holder(tmp_path, sleep_s=30):
    """A detached process holding the real `LOCK_EX` on the mutex path --
    what a pre-ledger gpu-claim looks like from the outside. A real
    second process is the only honest way to exercise `_take_mutex`'s
    timeout: an in-process flock can't contend with itself."""
    return subprocess.Popen(
        [sys.executable, "-c",
         "import fcntl,os,time;"
         f"fd=os.open({str(lg.mutex_path(KEY, tmp_path))!r},os.O_CREAT|os.O_RDWR,0o666);"
         f"fcntl.flock(fd,fcntl.LOCK_EX);print('held',flush=True);time.sleep({sleep_s})"],
        stdout=subprocess.PIPE, text=True)


def test_wait_true_warns_once_across_several_mutex_timeouts(
        tmp_path, monkeypatch, capsys):
    """A MutexTimeout means an old-style gpu-claim holds the mutex itself,
    which can last hours -- worth one explanation, not one every retry.

    `time.sleep` is not mocked here: `ledger._take_mutex`'s own internal
    poll shares the one process-wide `time` module with `gpu_claim`'s
    retry loop, so patching `time.sleep` globally (as the capacity-wait
    test above does, safely, because that path never touches a real
    flock) would also collapse `_take_mutex`'s internal polling and
    change which branch fires. Real, scaled-down durations avoid that
    trap entirely: the holder lives just long enough for several real
    `MutexTimeout`s to occur before it exits on its own and the claim
    goes through.
    """
    monkeypatch.setattr(lg, "MUTEX_WAIT_S", 0.03)
    monkeypatch.setattr("gpuqueue.claim.MUTEX_WAIT_POLL_S", 0.03)
    holder = _spawn_lock_ex_holder(tmp_path, sleep_s=0.4)

    attempts = []
    real_acquire = lg.acquire

    def counting_acquire(*a, **kw):
        attempts.append(1)
        return real_acquire(*a, **kw)

    monkeypatch.setattr(lg, "acquire", counting_acquire)
    try:
        assert holder.stdout.readline().strip() == "held"
        with gpu_claim(key=KEY, directory=tmp_path, usable_mb=7676,
                       wait=True) as c:
            assert c.vram_mb is None  # the claim eventually went through
    finally:
        if holder.poll() is None:
            holder.kill()
        holder.wait()
    assert len(attempts) >= 3  # several retries happened before it succeeded
    err = capsys.readouterr().err
    assert err.count("gpu-claim: warning:") == 1


def test_wait_false_raises_immediately_on_mutex_timeout(tmp_path, monkeypatch):
    monkeypatch.setattr(lg, "MUTEX_WAIT_S", 0.05)
    holder = _spawn_lock_ex_holder(tmp_path)
    try:
        assert holder.stdout.readline().strip() == "held"
        with pytest.raises(MutexTimeout):
            with gpu_claim(key=KEY, directory=tmp_path, usable_mb=7676):
                pass
    finally:
        holder.kill()
        holder.wait()


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


def test_default_usable_mb_is_none_when_the_reserve_swallows_the_card(
        monkeypatch):
    """A card smaller than DEFAULT_RESERVE_MB must not hand a negative
    usable_mb to ledger.fits/exceeds_capacity, where `want_mb > usable_mb`
    is then true for every claim -- silently admitting nothing. Same
    guard as Runner._usable_mb, applied to this sibling path."""
    monkeypatch.setattr("gpuqueue.claim.total_vram_mb", lambda: 256)
    assert default_usable_mb() is None


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
