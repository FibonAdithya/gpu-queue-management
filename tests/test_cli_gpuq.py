import json
import threading
import time
import pytest
from gpuqueue.cli_gpuq import main, generate_id
from gpuqueue.queue import QueueRoot

import io
from gpuqueue import bugfiler

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


def test_list_flags_an_orphaned_job(root, capsys):
    """Decision D: surface it, do not act on it."""
    import os
    _submit(root)
    q = QueueRoot(root)
    spec = q.claim("j1")
    spec.pid = os.getpid()       # a live "job"
    spec.runner_pid = 4000000    # whose runner is gone
    q.update(spec)
    capsys.readouterr()
    main(["list", "--json"])
    row = [r for r in json.loads(capsys.readouterr().out) if r["id"] == "j1"][0]
    assert row["orphaned"] is True

def test_list_does_not_flag_a_normally_running_job(root, capsys):
    import os
    _submit(root)
    q = QueueRoot(root)
    spec = q.claim("j1")
    spec.pid = os.getpid()
    spec.runner_pid = os.getpid()
    q.update(spec)
    capsys.readouterr()
    main(["list", "--json"])
    row = [r for r in json.loads(capsys.readouterr().out) if r["id"] == "j1"][0]
    assert row["orphaned"] is False

def test_show_reports_orphaned(root, capsys):
    import os
    _submit(root)
    q = QueueRoot(root)
    spec = q.claim("j1")
    spec.pid, spec.runner_pid = os.getpid(), 4000000
    q.update(spec)
    capsys.readouterr()
    main(["show", "j1"])
    assert json.loads(capsys.readouterr().out)["orphaned"] is True

def test_plain_list_marks_orphans_visibly(root, capsys):
    import os
    _submit(root)
    q = QueueRoot(root)
    spec = q.claim("j1")
    spec.pid, spec.runner_pid = os.getpid(), 4000000
    q.update(spec)
    capsys.readouterr()
    main(["list"])
    assert "ORPHANED" in capsys.readouterr().out


CONFIG = """
[queue]
root = "{root}"

[autofix]
enabled = true
repo = "you/gpu-queue-management"
"""


@pytest.fixture
def cfg_file(tmp_path):
    p = tmp_path / "gpuq.toml"
    p.write_text(CONFIG.format(root=tmp_path / "queue"))
    return p


@pytest.fixture
def reports(monkeypatch):
    sent = []
    monkeypatch.setattr(bugfiler, "file_agent_report",
                        lambda cfg, title, body: sent.append((title, body)) or 42)
    return sent


def test_bug_files_a_report(cfg_file, reports, monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO("gpuq wait never returns\n"))
    assert main(["bug", "--config", str(cfg_file), "wait hangs"]) == 0
    assert reports == [("wait hangs", "gpuq wait never returns\n")]
    assert "42" in capsys.readouterr().out


def test_bug_takes_the_body_from_a_flag(cfg_file, reports):
    main(["bug", "--config", str(cfg_file), "wait hangs", "--body", "inline"])
    assert reports[0][1] == "inline"


def test_bug_refuses_when_autofix_is_disabled(tmp_path, reports, capsys):
    p = tmp_path / "gpuq.toml"
    p.write_text('[queue]\nroot = "/q"\n')
    assert main(["bug", "--config", str(p), "t", "--body", "b"]) == 2
    assert reports == []
    assert "autofix" in capsys.readouterr().err


def test_bug_reports_a_gh_failure_without_a_traceback(cfg_file, monkeypatch,
                                                      capsys):
    monkeypatch.setattr(bugfiler, "file_agent_report",
                        lambda *a: (_ for _ in ()).throw(
                            bugfiler.GhError("HTTP 403")))
    assert main(["bug", "--config", str(cfg_file), "t", "--body", "b"]) == 1
    assert "403" in capsys.readouterr().err


def test_bug_refuses_an_empty_body(cfg_file, reports, capsys):
    """A report with no prose is not a report; the auto path exists for
    reports with no prose."""
    assert main(["bug", "--config", str(cfg_file), "t", "--body", "  "]) == 2
    assert reports == []


def test_submit_records_the_declaration(tmp_path, capsys):
    rc = main(["--queue-root", str(tmp_path), "submit", "--project", "p",
               "--commit", "abc", "--branch", "main", "--lane", "gpu",
               "--vram-mb", "512", "--", "python", "t.py"])
    assert rc == 0
    job_id = capsys.readouterr().out.strip()
    body = json.loads((tmp_path / "pending" / f"{job_id}.json").read_text())
    assert body["vram_mb"] == 512


def test_submit_without_a_declaration_takes_the_whole_card(tmp_path, capsys):
    main(["--queue-root", str(tmp_path), "submit", "--project", "p",
          "--commit", "abc", "--branch", "main", "--lane", "gpu",
          "--", "python", "t.py"])
    job_id = capsys.readouterr().out.strip()
    body = json.loads((tmp_path / "pending" / f"{job_id}.json").read_text())
    assert body["vram_mb"] is None


def test_submit_rejects_a_nonsense_declaration(tmp_path, capsys):
    rc = main(["--queue-root", str(tmp_path), "submit", "--project", "p",
               "--commit", "abc", "--branch", "main", "--lane", "gpu",
               "--vram-mb", "0", "--", "python", "t.py"])
    assert rc == 2
    assert "vram_mb" in capsys.readouterr().err


def test_kills_prints_recent_kills(tmp_path, monkeypatch, capsys):
    # The thing an agent that sees `killed by signal 9` can be TOLD to
    # run. A file nobody thinks to open is barely better than the runner
    # log nobody thinks to open, which is #24's actual complaint.
    from gpuqueue import killlog
    monkeypatch.setenv("QUEUE_ROOT", str(tmp_path))
    killlog.append(tmp_path, [{"pid": 2791919, "name": "tig-runtime",
                               "used_mb": 900,
                               "cgroup": "/system.slice/docker-abc.scope"}],
                   ["/workspace/lock/gpu"])
    assert main(["kills"]) == 0
    out = capsys.readouterr().out
    assert "2791919" in out
    assert "docker-abc.scope" in out


def test_kills_with_no_kills_says_so(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("QUEUE_ROOT", str(tmp_path))
    assert main(["kills"]) == 0
    assert "no kills" in capsys.readouterr().out.lower()


def test_kills_notes_truncation_with_both_counts(tmp_path, monkeypatch,
                                                  capsys):
    # `entries` and `total` must come from one read (fix round 1): a
    # separate second read for `total` could straddle a concurrent
    # sweep's append and report a window that never existed.
    from gpuqueue import killlog
    monkeypatch.setenv("QUEUE_ROOT", str(tmp_path))
    for i in range(5):
        killlog.append(tmp_path, [{"pid": i, "name": "p", "used_mb": 1,
                                   "cgroup": "/c"}], [])
    assert main(["kills", "--limit", "2"]) == 0
    out = capsys.readouterr().out
    assert "showing the most recent 2 of 5" in out
    assert "--limit 5" in out


def test_kills_omits_truncation_notice_when_everything_fits(tmp_path,
                                                             monkeypatch,
                                                             capsys):
    from gpuqueue import killlog
    monkeypatch.setenv("QUEUE_ROOT", str(tmp_path))
    for i in range(3):
        killlog.append(tmp_path, [{"pid": i, "name": "p", "used_mb": 1,
                                   "cgroup": "/c"}], [])
    assert main(["kills", "--limit", "20"]) == 0
    out = capsys.readouterr().out
    assert "showing the most recent" not in out


def test_kills_limit_zero_prints_no_entries(tmp_path, monkeypatch, capsys):
    # The falsy-zero trap, now guarded at the CLI layer too: `--limit 0`
    # must not fall back to "everything".
    from gpuqueue import killlog
    monkeypatch.setenv("QUEUE_ROOT", str(tmp_path))
    killlog.append(tmp_path, [{"pid": 2791919, "name": "tig-runtime",
                               "used_mb": 900,
                               "cgroup": "/system.slice/docker-abc.scope"}],
                   [])
    assert main(["kills", "--limit", "0"]) == 0
    out = capsys.readouterr().out
    assert "2791919" not in out
    assert "docker-abc.scope" not in out
