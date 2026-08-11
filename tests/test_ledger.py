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
