import os
import time
from pathlib import Path
import pytest
from gpuqueue.executor import (start_job, poll_job, kill_job, looks_like_oom,
                               StartFailed, STDERR_TAIL_BYTES)
from gpuqueue.spec import JobSpec

def mkspec(cmd, timeout_s=30, **over):
    d = dict(id="j1", lane="cpu", project="p", commit="abc", branch="main",
             cmd=cmd, artifacts=[], timeout_s=timeout_s)
    d.update(over)
    return JobSpec.from_dict(d)

def _logs(tmp_path):
    return tmp_path / "j1.out", tmp_path / "j1.err"

def finish(running, limit=30.0):
    """Poll to completion the way the runner's tick loop would."""
    deadline = time.monotonic() + limit
    while time.monotonic() < deadline:
        result = poll_job(running)
        if result is not None:
            return result
        time.sleep(0.02)
    raise AssertionError("job did not finish")

def test_success_returns_zero_and_captures_stdout(tmp_path):
    out, err = _logs(tmp_path)
    r = finish(start_job(mkspec(["sh", "-c", "echo hello"]), tmp_path, out, err))
    assert r.exit_code == 0 and r.timed_out is False
    assert out.read_text().strip() == "hello"

def test_poll_returns_none_while_running(tmp_path):
    out, err = _logs(tmp_path)
    running = start_job(mkspec(["sleep", "5"]), tmp_path, out, err)
    assert poll_job(running) is None
    kill_job(running)

def test_start_reports_the_pid_immediately(tmp_path):
    """The runner writes this into running/<id>.json before the next tick,
    so the reaper can tell a live job from an abandoned one."""
    out, err = _logs(tmp_path)
    running = start_job(mkspec(["sleep", "5"]), tmp_path, out, err)
    assert running.pid > 0
    kill_job(running)

def test_failure_captures_stderr_tail(tmp_path):
    out, err = _logs(tmp_path)
    r = finish(start_job(mkspec(["sh", "-c", "echo boom >&2; exit 7"]),
                         tmp_path, out, err))
    assert r.exit_code == 7
    assert "boom" in r.stderr_tail

def test_stderr_tail_is_bounded(tmp_path):
    out, err = _logs(tmp_path)
    r = finish(start_job(
        mkspec(["sh", "-c", "head -c 100000 /dev/zero | tr '\\0' 'x' >&2; exit 1"]),
        tmp_path, out, err))
    assert len(r.stderr_tail) <= STDERR_TAIL_BYTES

def test_timeout_kills_and_flags(tmp_path):
    out, err = _logs(tmp_path)
    r = finish(start_job(mkspec(["sleep", "30"], timeout_s=1), tmp_path, out, err))
    assert r.timed_out is True and r.exit_code != 0

def test_timeout_kills_the_whole_process_group(tmp_path):
    """A trainer that spawns dataloader workers must not leave them behind
    holding the card."""
    out, err = _logs(tmp_path)
    marker = tmp_path / "child.pid"
    cmd = ["sh", "-c", f"sleep 30 & echo $! > {marker}; sleep 30"]
    r = finish(start_job(mkspec(cmd, timeout_s=1), tmp_path, out, err))
    assert r.timed_out is True
    child = int(marker.read_text().strip())
    with pytest.raises(OSError):
        os.kill(child, 0)

def test_kill_job_terminates_a_live_job(tmp_path):
    """Runner shutdown must not leave a job holding the card."""
    out, err = _logs(tmp_path)
    running = start_job(mkspec(["sleep", "30"]), tmp_path, out, err)
    r = kill_job(running)
    assert r.exit_code != 0
    assert running.proc.poll() is not None

def test_runs_in_the_given_workdir(tmp_path):
    out, err = _logs(tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    (work / "here.txt").write_text("x")
    finish(start_job(mkspec(["sh", "-c", "ls"]), work, out, err))
    assert "here.txt" in out.read_text()

def test_extra_env_is_passed(tmp_path):
    out, err = _logs(tmp_path)
    finish(start_job(mkspec(["sh", "-c", "echo $GPUQ_JOB_ID"]), tmp_path, out, err,
                     extra_env={"GPUQ_JOB_ID": "j1"}))
    assert out.read_text().strip() == "j1"

def test_venv_bin_is_prepended_to_path(tmp_path):
    from gpuqueue.config import ProjectConfig
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    out, err = _logs(tmp_path)
    proj = ProjectConfig(name="p", remote="r", checkout=tmp_path, venv=venv)
    finish(start_job(mkspec(["sh", "-c", "echo $PATH"]), tmp_path, out, err,
                     project=proj))
    assert out.read_text().startswith(str(venv / "bin"))

def test_missing_executable_raises_start_failed_naming_the_binary(tmp_path):
    out, err = _logs(tmp_path)
    with pytest.raises(StartFailed, match="definitely-not-a-real-binary"):
        start_job(mkspec(["definitely-not-a-real-binary"]), tmp_path, out, err)

def test_looks_like_oom_detects_cuda_oom():
    assert looks_like_oom("RuntimeError: CUDA out of memory. Tried to allocate")
    assert looks_like_oom("torch.cuda.OutOfMemoryError: CUDA out of memory")
    assert not looks_like_oom("ValueError: bad config")

def test_oom_flag_set_from_stderr(tmp_path):
    out, err = _logs(tmp_path)
    r = finish(start_job(mkspec(["sh", "-c", "echo 'CUDA out of memory' >&2; exit 1"]),
                         tmp_path, out, err))
    assert r.oom is True
