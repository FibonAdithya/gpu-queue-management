"""Start, poll and kill one job's subprocess.

Knows nothing about queues, lanes or git. Deliberately non-blocking: the
runner is single-threaded, so nothing here may wait for a job to finish.
`start_job` hands back a handle, `poll_job` answers "done yet?" and returns
None while the answer is no.
"""
from __future__ import annotations

import os
import re
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from .config import ProjectConfig
from .spec import JobSpec

STDERR_TAIL_BYTES = 4000
_OOM = re.compile(r"cuda out of memory|outofmemoryerror|cublas_status_alloc_failed",
                  re.IGNORECASE)


class StartFailed(RuntimeError):
    """The job's command could not be executed at all.

    Carries the OSError's errno, because the message alone cannot tell
    "the binary you asked for is not there" (the caller's mistake) from
    "gpuq handed Popen a working directory that does not exist" (ours).
    """

    def __init__(self, message: str, errno: int | None = None):
        super().__init__(message)
        self.errno = errno


def looks_like_oom(text: str) -> bool:
    return bool(_OOM.search(text or ""))


@dataclass
class JobResult:
    exit_code: int
    timed_out: bool
    oom: bool
    stderr_tail: str


@dataclass
class RunningJob:
    spec: JobSpec
    proc: subprocess.Popen
    out_log: Path
    err_log: Path
    out_fh: IO[bytes]
    err_fh: IO[bytes]
    deadline: float  # time.monotonic() value past which it is killed

    @property
    def pid(self) -> int:
        return self.proc.pid


def _tail(path: Path, n: int = STDERR_TAIL_BYTES) -> str:
    try:
        data = path.read_bytes()
    except OSError:
        return ""
    return data[-n:].decode("utf-8", "replace")


def _env_for(project: ProjectConfig | None, extra: dict | None) -> dict:
    env = dict(os.environ)
    if project and project.venv:
        bin_dir = str(Path(project.venv) / "bin")
        env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")
        env["VIRTUAL_ENV"] = str(project.venv)
    env.update(extra or {})
    return env


def start_job(spec: JobSpec, workdir: Path, out_log: Path, err_log: Path,
              project: ProjectConfig | None = None,
              extra_env: dict | None = None) -> RunningJob:
    out_log.parent.mkdir(parents=True, exist_ok=True)
    fo = open(out_log, "wb")
    fe = open(err_log, "wb")
    try:
        proc = subprocess.Popen(
            spec.cmd, cwd=str(workdir), stdout=fo, stderr=fe,
            env=_env_for(project, extra_env),
            start_new_session=True,  # own process group, so we can kill it all
        )
    except OSError as e:
        # Record it where a consumer looks for failures, then tell the caller.
        fe.write(f"{e}\n".encode())
        fo.close()
        fe.close()
        raise StartFailed(f"cannot execute {spec.cmd[0]!r}: {e}",
                          errno=e.errno) from e
    return RunningJob(spec=spec, proc=proc, out_log=out_log, err_log=err_log,
                      out_fh=fo, err_fh=fe,
                      deadline=time.monotonic() + spec.timeout_s)


def poll_job(running: RunningJob) -> JobResult | None:
    """None while the job is still running.

    The wall-clock watchdog lives here because this is the one function
    called about a job on every tick.
    """
    if running.proc.poll() is None:
        if time.monotonic() < running.deadline:
            return None
        _kill_group(running.proc)
        return _result(running, timed_out=True)
    return _result(running, timed_out=False)


def kill_job(running: RunningJob) -> JobResult:
    """Stop a live job now. Used when the runner is shutting down."""
    if running.proc.poll() is None:
        _kill_group(running.proc)
    return _result(running, timed_out=False)


def _result(running: RunningJob, timed_out: bool) -> JobResult:
    for fh in (running.out_fh, running.err_fh):
        if not fh.closed:
            fh.close()  # flush before reading the tail back
    tail = _tail(running.err_log)
    code = running.proc.returncode if running.proc.returncode is not None else -1
    return JobResult(exit_code=code, timed_out=timed_out,
                     oom=looks_like_oom(tail), stderr_tail=tail)


def _kill_group(proc: subprocess.Popen) -> None:
    """SIGTERM the group, then SIGKILL what survives. A trainer's dataloader
    workers must not outlive it holding VRAM.

    This does block, for up to the two grace periods. That is deliberate: a
    job being killed is not the moment to admit another one, and the runner
    has nothing useful to do until the card is actually free.
    """
    try:
        pgid = os.getpgid(proc.pid)
    except OSError:
        return
    for sig, grace in ((signal.SIGTERM, 10.0), (signal.SIGKILL, 5.0)):
        try:
            os.killpg(pgid, sig)
        except OSError:
            return
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                return
            time.sleep(0.1)
