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
