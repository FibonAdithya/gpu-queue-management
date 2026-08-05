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
