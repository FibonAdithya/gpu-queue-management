import json
import pytest
from gpuqueue.cli_gpuq import main, generate_id

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
