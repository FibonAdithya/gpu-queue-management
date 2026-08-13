# tests/test_ledger.py
import json
import os
import subprocess
import sys
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


def test_zero_is_not_misread_as_none(tmp_path):
    """0 is falsy but not absent: `d.get(...) if ... else None` would
    silently turn a real 0 into None, and for vram_mb that flips the
    record's meaning to exclusive -- the most permissive value there is."""
    path = lg.ledger_dir(KEY, tmp_path) / "100.aaa.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        {"pid": 100, "usage_pid": 0, "vram_mb": 0, "owner": "me",
         "cmd": [], "started_at": "2026-08-10T00:00:00Z", "key": KEY}))
    (got,) = lg.records_for(KEY, tmp_path)
    assert got.usage_pid == 0
    assert got.vram_mb == 0


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


def test_mutex_timeout_is_caught_by_except_claim_busy():
    """The whole point of `MutexTimeout(ClaimBusy)`: every existing
    `except ClaimBusy` -- the runner, gpu-claim, callers not yet written
    -- must keep catching this without being told about it."""
    try:
        raise lg.MutexTimeout("an old gpu-claim is holding the mutex")
    except lg.ClaimBusy as e:
        assert isinstance(e, lg.MutexTimeout)
    else:
        pytest.fail("MutexTimeout escaped an `except ClaimBusy`")


def app(pid, used_mb=100, name="train.py"):
    return {"pid": pid, "used_mb": used_mb, "name": name}


def test_attribute_charges_a_pid_to_its_records_tree(monkeypatch):
    monkeypatch.setattr(lg, "descendants",
                        lambda pid: {555} if pid == 100 else set())
    r = rec(512, pid=100)
    owned, unledgered = lg.attribute([app(555)], [r])
    assert [a["pid"] for a in owned[str(r.path)]] == [555]
    assert unledgered == []


def test_attribute_charges_the_usage_pid_itself(monkeypatch):
    monkeypatch.setattr(lg, "descendants", lambda pid: set())
    r = rec(512, pid=100)
    owned, unledgered = lg.attribute([app(100)], [r])
    assert [a["pid"] for a in owned[str(r.path)]] == [100]


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
    assert [x["pid"] for x in owned[str(a.path)]] == [1100]
    assert [x["pid"] for x in owned[str(b.path)]] == [1200]
    assert unledgered == []


def test_used_mb_sums_and_tolerates_unknowns():
    assert lg.used_mb([app(1, 200), app(2, 300)]) == 500
    assert lg.used_mb([app(1, None)]) == 0


def test_records_with_the_same_name_in_different_dirs_both_own(monkeypatch):
    """`all_records()` spans every `<key>.lock.d/` directory, so a bare
    filename is only unique within one directory, not across the whole
    ledger. Keying `owned` by name would let the second record's tree
    silently overwrite the first's, and the first holder's live process
    would read as a stranger to the reaper and the watchdog -- both of
    which kill."""
    monkeypatch.setattr(lg, "descendants", lambda pid: set())
    a = lg.Record(path=Path("/tmp/keyA.lock.d/100.aaa.json"), pid=100,
                  usage_pid=100, vram_mb=512, owner="a", cmd=[],
                  started_at="", key=KEY)
    b = lg.Record(path=Path("/tmp/keyB.lock.d/100.aaa.json"), pid=200,
                  usage_pid=200, vram_mb=512, owner="b", cmd=[],
                  started_at="", key=KEY)
    assert a.name == b.name  # same basename, different directories
    owned, unledgered = lg.attribute([app(100), app(200)], [a, b])
    assert [x["pid"] for x in owned[str(a.path)]] == [100]
    assert [x["pid"] for x in owned[str(b.path)]] == [200]
    assert unledgered == []


# --- the holder cap: a latency budget the whole box shares ---------------

def test_fits_refuses_past_the_holder_cap(tmp_path):
    """VRAM accounting alone admits sixteen 500 MiB jobs onto an 8 GB card.
    `gpu_max_jobs` is the latency budget that stops it -- and it has to live
    here, not in the runner, or a hand-run gpu-claim walks straight past it."""
    recs = [mkrec(tmp_path, "100.a.json", vram_mb=500),
            mkrec(tmp_path, "101.b.json", vram_mb=500)]
    assert lg.fits(recs, 500, 8000) is True            # uncapped, as before
    assert lg.fits(recs, 500, 8000, max_holders=3) is True
    assert lg.fits(recs, 500, 8000, max_holders=2) is False


def test_the_cap_does_not_refuse_the_first_holder(tmp_path):
    assert lg.fits([], None, 8000, max_holders=1) is True


def test_busy_message_names_the_cap_rather_than_the_free_space(tmp_path):
    """Refused for the count while 7 GB sits free, the VRAM wording sends
    the reader looking for a memory problem that is not there."""
    recs = [mkrec(tmp_path, "100.a.json", vram_mb=500),
            mkrec(tmp_path, "101.b.json", vram_mb=500)]
    msg = lg.busy_message(KEY, recs, 500, 8000, max_holders=2)
    assert "2-job limit" in msg
    assert "me" in msg          # still lists who is on the card


def test_acquire_refuses_past_the_holder_cap(tmp_path):
    live = os.getpid()
    for n in ("a", "b"):
        mkrec(tmp_path, f"{live}.{n}.json", pid=live, usage_pid=live,
              vram_mb=500)
    with pytest.raises(lg.ClaimBusy, match="2-job limit"):
        lg.acquire(KEY, vram_mb=500, owner="third", cmd=["train"],
                   directory=tmp_path, usable_mb=8000, max_holders=2)


def test_acquire_admits_under_the_cap(tmp_path):
    live = os.getpid()
    mkrec(tmp_path, f"{live}.a.json", pid=live, usage_pid=live, vram_mb=500)
    rec = lg.acquire(KEY, vram_mb=500, owner="second", cmd=["train"],
                     directory=tmp_path, usable_mb=8000, max_holders=2)
    assert rec.owner == "second"


def test_a_dead_holder_does_not_spend_a_slot(tmp_path):
    """`acquire` counts live records, the same set it sums VRAM over."""
    mkrec(tmp_path, "4000000.dead.json", pid=4000000, vram_mb=500)
    rec = lg.acquire(KEY, vram_mb=500, owner="live", cmd=["train"],
                     directory=tmp_path, usable_mb=8000, max_holders=1)
    assert rec.owner == "live"
