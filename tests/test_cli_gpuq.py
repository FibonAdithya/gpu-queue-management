import json
import threading
import time
import pytest
from gpuqueue.cli_gpuq import main, generate_id
from gpuqueue.queue import QueueRoot

@pytest.fixture
def root(tmp_path, monkeypatch):
    r = tmp_path / "queue"
    monkeypatch.setenv("QUEUE_ROOT", str(r))
    return r

def test_submit_creates_pending_job(root, capsys):
    rc = main(["submit", "--project", "p", "--commit", "abc", "--branch", "main",
               "--lane", "cpu", "--id", "j1", "--", "python", "-c", "print(1)"])
    assert rc == 0
    assert capsys.readouterr().out.strip() == "j1"
    body = json.loads((root / "pending" / "j1.json").read_text())
    assert body["cmd"] == ["python", "-c", "print(1)"]
    assert body["lane"] == "cpu"

def test_submit_generates_id_when_omitted(root, capsys):
    assert main(["submit", "--project", "p", "--commit", "abc",
                 "--branch", "main", "--", "true"]) == 0
    job_id = capsys.readouterr().out.strip()
    assert (root / "pending" / f"{job_id}.json").exists()

def test_submit_default_lane_is_cpu(root, capsys):
    main(["submit", "--project", "p", "--commit", "abc", "--branch", "main",
          "--id", "j1", "--", "true"])
    assert json.loads((root / "pending" / "j1.json").read_text())["lane"] == "cpu"

def test_submit_dedupe_prints_existing_id(root, capsys):
    args = ["submit", "--project", "p", "--commit", "abc", "--branch", "main",
            "--dedupe-key", "k", "--", "true"]
    main(args)
    first = capsys.readouterr().out.strip()
    main(args)
    assert capsys.readouterr().out.strip() == first

def test_submit_invalid_lane_exits_nonzero(root, capsys):
    rc = main(["submit", "--project", "p", "--commit", "abc", "--branch", "main",
               "--lane", "tpu", "--", "true"])
    assert rc == 2
    assert "lane" in capsys.readouterr().err

def test_submit_requires_cmd(root, capsys):
    assert main(["submit", "--project", "p", "--commit", "abc",
                 "--branch", "main", "--"]) == 2

def test_list_json_output(root, capsys):
    main(["submit", "--project", "p", "--commit", "abc", "--branch", "main",
          "--id", "j1", "--", "true"])
    capsys.readouterr()
    assert main(["list", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows[0]["id"] == "j1" and rows[0]["state"] == "pending"

def test_list_filters_by_state(root, capsys):
    main(["submit", "--project", "p", "--commit", "abc", "--branch", "main",
          "--id", "j1", "--", "true"])
    capsys.readouterr()
    main(["list", "--state", "done", "--json"])
    assert json.loads(capsys.readouterr().out) == []

def test_show_prints_spec(root, capsys):
    main(["submit", "--project", "p", "--commit", "abc", "--branch", "main",
          "--id", "j1", "--", "true"])
    capsys.readouterr()
    assert main(["show", "j1"]) == 0
    assert json.loads(capsys.readouterr().out)["id"] == "j1"

def test_show_missing_job_exits_1(root, capsys):
    assert main(["show", "nope"]) == 1

def test_cancel_pending(root, capsys):
    main(["submit", "--project", "p", "--commit", "abc", "--branch", "main",
          "--id", "j1", "--", "true"])
    assert main(["cancel", "j1"]) == 0
    assert (root / "failed" / "j1.json").exists()

def test_cancel_unknown_exits_1(root):
    assert main(["cancel", "nope"]) == 1

def test_generate_id_is_unique_and_safe():
    a, b = generate_id("job"), generate_id("job")
    assert a != b
    assert "/" not in a and a.startswith("job-")
def _submit(root, job_id="j1"):
    main(["submit", "--project", "p", "--commit", "abc", "--branch", "main",
          "--id", job_id, "--", "true"])

def test_wait_returns_zero_for_an_already_finished_job(root, capsys):
    _submit(root)
    q = QueueRoot(root)
    q.finish(q.claim("j1"), ok=True)
    capsys.readouterr()
    start = time.monotonic()
    assert main(["wait", "j1"]) == 0
    assert time.monotonic() - start < 1.0  # immediate, not one poll interval

def test_wait_returns_one_for_a_failed_job(root):
    _submit(root)
    q = QueueRoot(root)
    spec = q.claim("j1")
    spec.error = "boom"
    q.finish(spec, ok=False)
    assert main(["wait", "j1"]) == 1

def test_wait_blocks_until_the_job_finishes(root):
    _submit(root)
    q = QueueRoot(root)
    q.claim("j1")

    def finish_soon():
        time.sleep(0.3)
        state, spec = q.find("j1")
        q.finish(spec, ok=True)

    t = threading.Thread(target=finish_soon, daemon=True)
    t.start()
    assert main(["wait", "j1", "--poll", "0.05"]) == 0
    t.join()

def test_wait_timeout_exits_124_and_leaves_the_job_alone(root):
    _submit(root)
    QueueRoot(root).claim("j1")
    assert main(["wait", "j1", "--timeout", "0.2", "--poll", "0.05"]) == 124
    assert (root / "running" / "j1.json").exists()  # not cancelled

def test_wait_unknown_job_exits_2(root):
    assert main(["wait", "nope", "--timeout", "0.2"]) == 2

def test_wait_rides_out_a_requeue(root):
    """The reaper moves a job running -> pending. A scan that passes pending
    before the move and reads running after it sees neither; that is a
    transient, not a missing job."""
    _submit(root)
    q = QueueRoot(root)
    spec = q.claim("j1")

    def requeue_then_finish():
        time.sleep(0.2)
        q.requeue(spec)
        time.sleep(0.2)
        again = q.claim("j1")
        q.finish(again, ok=True)

    t = threading.Thread(target=requeue_then_finish, daemon=True)
    t.start()
    assert main(["wait", "j1", "--poll", "0.05"]) == 0
    t.join()

def test_submit_wait_does_both(root, capsys):
    q = QueueRoot(root)
    q.ensure_dirs()

    def finish_soon():
        time.sleep(0.3)
        found = q.find("j1")
        while found is None or found[0] == "pending":
            if found and found[0] == "pending":
                q.claim("j1")
            time.sleep(0.05)
            found = q.find("j1")
        q.finish(found[1], ok=True)

    t = threading.Thread(target=finish_soon, daemon=True)
    t.start()
    rc = main(["submit", "--project", "p", "--commit", "abc", "--branch", "main",
               "--id", "j1", "--wait", "--poll", "0.05", "--", "true"])
    t.join()
    assert rc == 0
