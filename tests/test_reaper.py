import json
import pytest
from gpuqueue import reaper as rp
from gpuqueue.reaper import reap, MAX_ATTEMPTS
from gpuqueue.queue import QueueRoot
from gpuqueue.spec import JobSpec
from gpuqueue.config import RunnerConfig

def mkspec(job_id="j1", **over):
    d = dict(id=job_id, lane="gpu", project="p", commit="abc", branch="main",
             cmd=["true"], artifacts=[], timeout_s=60)
    d.update(over)
    return JobSpec.from_dict(d)

@pytest.fixture
def q(tmp_path):
    qr = QueueRoot(tmp_path / "queue")
    qr.ensure_dirs()
    return qr

@pytest.fixture
def cfg(q):
    return RunnerConfig(queue_root=q.root, kill_orphan_cuda=False)

@pytest.fixture(autouse=True)
def no_gpu_calls(monkeypatch):
    monkeypatch.setattr(rp, "release_stale", lambda directory=None: [])
    monkeypatch.setattr(rp, "compute_apps", lambda: [])
    monkeypatch.setattr(rp, "own_pids", lambda: set())
    # Attribution now walks each record's process tree; without this every
    # test using the stubbed compute_apps() would also need to know about
    # ledger.attribute's internals.
    monkeypatch.setattr(rp, "descendants", lambda pid: set())

def test_requeues_running_job_with_dead_pid(q, cfg):
    q.submit(mkspec())
    spec = q.claim("j1")
    spec.pid = 4000000
    q._write(q.path_for("running", "j1"), spec)
    result = reap(q, cfg)
    assert result["requeued"] == ["j1"]
    assert json.loads((q.root / "pending" / "j1.json").read_text())["attempts"] == 1

def test_second_reap_fails_instead_of_requeueing(q, cfg):
    """Without an attempt ceiling a crash-looping job occupies the only
    card indefinitely."""
    q.submit(mkspec(attempts=MAX_ATTEMPTS))
    spec = q.claim("j1")
    spec.pid = 4000000
    q._write(q.path_for("running", "j1"), spec)
    result = reap(q, cfg)
    assert result["failed"] == ["j1"]
    body = json.loads((q.root / "failed" / "j1.json").read_text())
    assert "attempt" in (body["error"] or "").lower()

def test_leaves_running_job_with_live_pid(q, cfg):
    import os
    q.submit(mkspec())
    spec = q.claim("j1")
    spec.pid = os.getpid()
    q._write(q.path_for("running", "j1"), spec)
    assert reap(q, cfg)["requeued"] == []
    assert (q.root / "running" / "j1.json").exists()

def test_leaves_jobs_this_runner_is_actively_executing(q, cfg):
    """A job dispatched microseconds ago has no pid yet; reaping it would
    run it twice."""
    q.submit(mkspec())
    q.claim("j1")
    assert reap(q, cfg, active_ids={"j1"})["requeued"] == []

def test_requeues_running_job_with_no_pid_when_not_active(q, cfg):
    q.submit(mkspec())
    q.claim("j1")
    assert reap(q, cfg, active_ids=set())["requeued"] == ["j1"]

def test_releases_stale_claims(q, cfg, monkeypatch):
    monkeypatch.setattr(rp, "release_stale",
                        lambda directory=None: [{"pid": 999, "owner": "ghost"}])
    assert reap(q, cfg)["stale_claims"] == [{"pid": 999, "owner": "ghost"}]

def test_removes_part_files(q, cfg):
    (q.root / "work" / "j1").mkdir(parents=True)
    stray = q.root / "work" / "j1" / "out.json.part"
    stray.write_text("half")
    assert str(stray) in reap(q, cfg)["cleaned_paths"]
    assert not stray.exists()

def test_kills_orphan_cuda_when_enabled(q, monkeypatch):
    cfg = RunnerConfig(queue_root=q.root, kill_orphan_cuda=True)
    monkeypatch.setattr(rp, "compute_apps",
                        lambda: [{"pid": 4321, "used_mb": 900, "name": "t.py"}])
    killed = []
    monkeypatch.setattr(rp, "_kill", lambda pid: killed.append(pid) or True)
    assert reap(q, cfg)["killed_pids"] == [4321]
    assert killed == [4321]

def test_does_not_kill_pids_of_running_jobs(q, monkeypatch):
    cfg = RunnerConfig(queue_root=q.root, kill_orphan_cuda=True)
    q.submit(mkspec())
    spec = q.claim("j1")
    spec.pid = 4321
    q._write(q.path_for("running", "j1"), spec)
    # The job has to still be in `running` when reap() builds the protect set:
    # requeue_orphans() runs first and drops any running job whose process is
    # gone. Without this the test asserts nothing on a box where pid 4321 is
    # dead -- j1 gets requeued, protect comes out empty, and the kill is real.
    monkeypatch.setattr(rp, "pid_alive", lambda pid: True)
    monkeypatch.setattr(rp, "compute_apps",
                        lambda: [{"pid": 4321, "used_mb": 900, "name": "t.py"}])
    monkeypatch.setattr(rp, "_kill", lambda pid: pytest.fail("killed a live job"))
    assert reap(q, cfg)["killed_pids"] == []

def test_does_not_kill_when_cuda_list_is_invisible(q, monkeypatch):
    cfg = RunnerConfig(queue_root=q.root, kill_orphan_cuda=True)
    monkeypatch.setattr(rp, "compute_apps", lambda: None)
    monkeypatch.setattr(rp, "_kill", lambda pid: pytest.fail("killed blind"))
    assert reap(q, cfg)["killed_pids"] == []


def test_clean_partials_leaves_a_live_jobs_files_alone(tmp_path):
    """Reaping now runs on every tick, so it must not sweep inside a job that
    is still going -- a .part file there is that job's business, not debris."""
    from gpuqueue.reaper import clean_partials
    q = QueueRoot(tmp_path / "q"); q.ensure_dirs()
    q.submit(mkspec("live"))
    q.claim("live")                       # now in running/
    live_part = q.work_dir("live") / "checkpoint.part"
    live_part.parent.mkdir(parents=True, exist_ok=True)
    live_part.write_text("half a checkpoint")
    dead_part = q.work_dir("gone") / "leftover.part"
    dead_part.parent.mkdir(parents=True, exist_ok=True)
    dead_part.write_text("debris")

    clean_partials(q)

    assert live_part.exists(), "swept a running job's own file"
    assert not dead_part.exists()

def test_reap_can_skip_the_expensive_cuda_sweep(tmp_path, monkeypatch):
    from gpuqueue.reaper import reap
    import gpuqueue.reaper as rp
    called = []
    monkeypatch.setattr(rp, "kill_orphan_cuda",
                        lambda protect, records, apps: called.append(1) or [])
    q = QueueRoot(tmp_path / "q"); q.ensure_dirs()
    cfg = RunnerConfig(queue_root=q.root, claim_dir=tmp_path / "c")
    reap(q, cfg, include_orphan_cuda=False)
    assert called == []
    reap(q, cfg, include_orphan_cuda=True)
    assert called == [1]


# --- a direct `gpu-claim` run must survive the sweep --------------------
#
# The README advertises `gpu-claim -- <cmd>` as usable on its own. Its CUDA
# process is the *child* of the pid recorded in the claim file, and
# `preflight.own_pids` expands `descendants()` only for the runner's own
# pid -- so that child is exempted by nothing and `kill_orphan_cuda`
# SIGKILLs a legitimate run. Every other test in this file stubs
# `own_pids` to set(), which is why the suite cannot currently see this.

import os as _os
import signal as _signal
import subprocess as _sp
import sys as _sys
import time as _time
from pathlib import Path as _Path
from gpuqueue import preflight as _pf
from gpuqueue import claim as _cl

# Detaches before spawning its child, so the pair is NOT in pytest's own
# process tree. Without the double fork `descendants(os.getpid())` exempts
# them and these tests pass while the bug is fully present.
_HOLDER = r'''
import json, os, subprocess, sys, time
out = sys.argv[1]
if os.fork() != 0: os._exit(0)
os.setsid()
if os.fork() != 0: os._exit(0)
child = subprocess.Popen(["sleep", "60"])
tmp = out + ".part"
with open(tmp, "w") as f:
    json.dump({"holder": os.getpid(), "child": child.pid}, f)
os.rename(tmp, out)
time.sleep(60)
'''


def _live(pid: int) -> bool:
    """Not `claim.pid_alive`: that uses os.kill(pid, 0), which succeeds on a
    zombie. A SIGKILLed child whose parent is still sleeping is exactly a
    zombie, so pid_alive would report the kill as survival."""
    try:
        status = _Path(f"/proc/{pid}/status").read_text()
    except OSError:
        return False
    for line in status.splitlines():
        if line.startswith("State:"):
            return line.split()[1] != "Z"
    return False


def _write_claim(directory, holder: int) -> None:
    """A hand-run `gpu-claim`'s record, in whichever directory that run's
    own `$GPU_CLAIM_DIR` resolved to.

    Separate from the fixture below because *which* directory it lands in
    is the variable under test in the two-environment case: the claim
    writer is an interactive shell and the reaper is a supervisor unit, and
    they do not resolve that variable to the same place (issue #19).
    """
    directory = _Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "GPU-test.lock.json").write_text(json.dumps({
        "pid": holder, "owner": "someone", "cmd": ["python", "train.py"],
        "started_at": "2026-08-10T00:00:00Z", "key": "GPU-test"}))


@pytest.fixture
def holder_process(tmp_path):
    """A detached `gpu-claim`-shaped process pair, with no claim written.

    Detached on purpose, and the two assertions below are what make every
    test built on this honest: `own_pids` exempts `os.getpid()` and its
    whole tree unconditionally, so a helper inside pytest's tree would be
    exempt no matter what the claim directory logic did.
    """
    out = tmp_path / "pids.json"
    _sp.run([_sys.executable, "-c", _HOLDER, str(out)], check=True, timeout=30)
    deadline = _time.monotonic() + 10
    while not out.exists():
        if _time.monotonic() > deadline:
            pytest.fail("helper never reported its pids")
        _time.sleep(0.05)
    pids = json.loads(out.read_text())
    holder, child = pids["holder"], pids["child"]

    # The whole reproduction rests on these two not being in pytest's tree.
    assert holder not in _pf.descendants(_os.getpid())
    assert child not in _pf.descendants(_os.getpid())
    try:
        yield holder, child
    finally:
        for pid in (child, holder):
            try:
                _os.kill(pid, _signal.SIGKILL)
            except OSError:
                pass


@pytest.fixture
def direct_claim(tmp_path, monkeypatch, holder_process):
    """The single-environment case: the claim lands in the same
    `$GPU_CLAIM_DIR` the reaper reads."""
    claim_dir = tmp_path / "claims"
    claim_dir.mkdir()
    monkeypatch.setenv("GPU_CLAIM_DIR", str(claim_dir))
    holder, child = holder_process
    _write_claim(claim_dir, holder)
    return holder, child


def test_own_pids_covers_a_claim_holders_children(direct_claim):
    """The root cause. The claim names the gpu-claim process; the process
    actually on the card is its child."""
    holder, child = direct_claim
    own = _pf.own_pids()
    assert holder in own, "the recorded pid itself is exempt"
    assert child in own, "a claim holder's CUDA child is not exempt"


def test_own_pids_exempts_a_claim_written_under_the_default_dir(
        holder_process, tmp_path, monkeypatch):
    """Two environments, which is what issue #19 turns on.

    The daemon got `$GPU_CLAIM_DIR` from its supervisor unit. The
    interactive shell that ran `gpu-claim` did not inherit it, so the claim
    landed on `DEFAULT_CLAIM_DIR`. `own_pids` resolves the variable in the
    *reaper's* process, so before the union it read a directory the claim
    was never written to -- not a superset of what `attribute` owns, but
    disjoint from it.
    """
    holder, child = holder_process
    daemon_dir = tmp_path / "daemon-claims"          # what supervisor gave it
    daemon_dir.mkdir()
    monkeypatch.setenv("GPU_CLAIM_DIR", str(daemon_dir))
    _write_claim(_cl.DEFAULT_CLAIM_DIR, holder)      # where the shell wrote

    from gpuqueue import ledger as lg
    assert lg.all_records(daemon_dir) == [], \
        "the two environments must genuinely diverge or this proves nothing"

    own = _pf.own_pids()

    assert holder in own, "the hand-run claim's own pid is not exempt"
    assert child in own, "the trainer under a hand-run claim is not exempt"


def test_own_pids_given_a_directory_reads_only_that_one(
        holder_process, tmp_path, monkeypatch):
    """The widening is for the bare call. A caller that names a directory
    is asking about that directory, and answering about a different one
    would make `directory=` mean nothing."""
    holder, _child = holder_process
    empty = tmp_path / "empty"
    empty.mkdir()
    _write_claim(_cl.DEFAULT_CLAIM_DIR, holder)

    assert holder not in _pf.own_pids(directory=empty)
    assert holder in _pf.own_pids(), "the bare call must still widen"


def test_a_claim_the_daemons_environment_cannot_see_is_still_spared(
        q, tmp_path, holder_process, monkeypatch):
    """The reproduction, end to end: 14 of 29 runs on the deployment box,
    `exit -9` with empty stderr, at exactly `orphan_cuda_interval_s`.

    `test_a_divergent_runner_claim_dir_still_spares_a_direct_run` cannot
    express this. It varies `cfg.claim_dir` while the claim is still
    written into the same `$GPU_CLAIM_DIR` that `own_pids()` reads; here
    `cfg.claim_dir` and the daemon's `$GPU_CLAIM_DIR` *agree*, and it is
    the interactive user who diverges.
    """
    holder, child = holder_process
    daemon_dir = tmp_path / "daemon-claims"
    daemon_dir.mkdir()
    monkeypatch.setenv("GPU_CLAIM_DIR", str(daemon_dir))
    _write_claim(_cl.DEFAULT_CLAIM_DIR, holder)

    cfg = RunnerConfig(queue_root=q.root, kill_orphan_cuda=True,
                       claim_dir=daemon_dir)
    monkeypatch.setattr(rp, "own_pids", _pf.own_pids)   # undo the autouse stub
    monkeypatch.setattr(rp, "compute_apps",
                        lambda: [{"pid": child, "used_mb": 900,
                                  "name": "train.py"}])

    from gpuqueue import ledger as lg
    assert lg.all_records(daemon_dir) == [], \
        "the config and the daemon's environment must agree with each other"

    result = reap(q, cfg)

    deadline = _time.monotonic() + 2
    while _live(child) and _time.monotonic() < deadline:
        _time.sleep(0.05)
    assert _live(child), ("SIGKILLed a direct gpu-claim run whose claim went "
                          "to the default directory because the reaper's "
                          "$GPU_CLAIM_DIR came from its supervisor unit")
    assert result["killed_pids"] == []


def test_does_not_kill_a_direct_gpu_claim_run(q, direct_claim, monkeypatch):
    """The consequence: a legitimate `gpu-claim` run is SIGKILLed within
    orphan_cuda_interval_s on any box where the runner is up."""
    holder, child = direct_claim
    cfg = RunnerConfig(queue_root=q.root, kill_orphan_cuda=True)
    monkeypatch.setattr(rp, "own_pids", _pf.own_pids)   # undo the autouse stub
    monkeypatch.setattr(rp, "compute_apps",
                        lambda: [{"pid": child, "used_mb": 900,
                                  "name": "train.py"}])

    result = reap(q, cfg)

    deadline = _time.monotonic() + 2
    while _live(child) and _time.monotonic() < deadline:
        _time.sleep(0.05)
    assert _live(child), "SIGKILLed a legitimate direct gpu-claim run"
    assert result["killed_pids"] == []


def test_a_divergent_runner_claim_dir_still_spares_a_direct_run(
        q, tmp_path, direct_claim, monkeypatch):
    """The superset invariant `reaper.kill_orphan_cuda` rests on, made
    observable.

    The test above passes `claim_dir=None`, so `cfg.claim_dir` and
    `$GPU_CLAIM_DIR` are the same directory and the two ownership
    computations cannot disagree. Here they are deliberately different --
    which a real box reaches by being bootstrapped before `claim_dir`
    was templated into `gpuq.toml`, or by hand-editing the config.

    So `ledger.attribute` sees no records at all and the holder's CUDA
    child is unledgered, i.e. a kill candidate. The only thing standing
    between it and SIGKILL is bare `own_pids()` reading `$GPU_CLAIM_DIR`.
    Both "cleanups" `kill_orphan_cuda` warns against -- passing
    `cfg.claim_dir` into `own_pids()`, or routing it through
    `attribute()` -- point that read at the empty directory and turn this
    red.
    """
    holder, child = direct_claim
    runner_dir = tmp_path / "runner-claims"      # empty, and not the env one
    runner_dir.mkdir()
    cfg = RunnerConfig(queue_root=q.root, kill_orphan_cuda=True,
                       claim_dir=runner_dir)
    monkeypatch.setattr(rp, "own_pids", _pf.own_pids)   # undo the autouse stub
    monkeypatch.setattr(rp, "compute_apps",
                        lambda: [{"pid": child, "used_mb": 900,
                                  "name": "train.py"}])

    from gpuqueue import ledger as lg
    assert lg.all_records(runner_dir) == [], \
        "the two directories must genuinely diverge or this proves nothing"

    result = reap(q, cfg)

    deadline = _time.monotonic() + 2
    while _live(child) and _time.monotonic() < deadline:
        _time.sleep(0.05)
    assert _live(child), ("SIGKILLed a direct gpu-claim run because the "
                          "runner's claim_dir disagreed with $GPU_CLAIM_DIR")
    assert result["killed_pids"] == []


def test_a_ledgered_co_tenants_process_is_not_an_orphan(q, tmp_path, monkeypatch):
    """The whole point of sharing: a declared holder's CUDA process is
    someone else's job, not debris."""
    from gpuqueue import ledger as lg
    cfg = RunnerConfig(queue_root=q.root, kill_orphan_cuda=True,
                       claim_dir=tmp_path)
    lg.write_record(lg.Record(
        path=lg.ledger_dir("k", tmp_path) / f"{_os.getpid()}.aaa.json",
        pid=_os.getpid(), usage_pid=_os.getpid(), vram_mb=512, owner="co",
        cmd=[], started_at="2026-08-10T00:00:00Z", key="k"))
    monkeypatch.setattr(rp, "compute_apps",
                        lambda: [{"pid": 5150, "used_mb": 400, "name": "co.py"}])
    monkeypatch.setattr(rp, "descendants",
                        lambda pid: {5150} if pid == _os.getpid() else set())
    # ledger.attribute() walks its *own* `descendants` (imported directly
    # in ledger.py), never reaper's -- patching rp.descendants above
    # cannot reach it. This is what actually puts 5150 in the co-tenant's
    # tree; see the identical note in tests/test_preflight.py, which hit
    # the same thing.
    monkeypatch.setattr(lg, "descendants",
                        lambda pid: {5150} if pid == _os.getpid() else set())
    monkeypatch.setattr(rp, "_kill", lambda pid: pytest.fail("killed a co-tenant"))
    assert reap(q, cfg)["killed_pids"] == []


def test_an_unledgered_process_is_still_killed(q, tmp_path, monkeypatch):
    cfg = RunnerConfig(queue_root=q.root, kill_orphan_cuda=True,
                       claim_dir=tmp_path)
    monkeypatch.setattr(rp, "compute_apps",
                        lambda: [{"pid": 4321, "used_mb": 900, "name": "x.py"}])
    monkeypatch.setattr(rp, "descendants", lambda pid: set())
    killed = []
    monkeypatch.setattr(rp, "_kill", lambda pid: killed.append(pid) or True)
    assert reap(q, cfg)["killed_pids"] == [4321]


# --- the VRAM watchdog ---------------------------------------------------

def _rec(tmp_path, name, usage_pid, vram_mb, owner="gpuq:j1"):
    from gpuqueue import ledger as lg
    return lg.Record(path=lg.ledger_dir("k", tmp_path) / name, pid=_os.getpid(),
                     usage_pid=usage_pid, vram_mb=vram_mb, owner=owner,
                     cmd=["python", "t.py"], started_at="2026-08-10T00:00:00Z",
                     key="k")


def test_one_sweep_over_the_line_does_not_convict(tmp_path, monkeypatch):
    """The caching allocator's high-water mark moves in steps; one sample
    over is not evidence of a persistent overage."""
    monkeypatch.setattr(rp, "descendants", lambda pid: set())
    r = _rec(tmp_path, "1.a.json", 500, 512)
    strikes = {}
    apps = [{"pid": 500, "used_mb": 3070, "name": "t.py"}]
    assert rp.check_vram([r], apps, strikes) == []
    assert strikes[str(r.path)] == 1


def test_two_consecutive_sweeps_convict(tmp_path, monkeypatch):
    monkeypatch.setattr(rp, "descendants", lambda pid: set())
    r = _rec(tmp_path, "1.a.json", 500, 512)
    strikes = {}
    apps = [{"pid": 500, "used_mb": 3070, "name": "t.py"}]
    rp.check_vram([r], apps, strikes)
    (guilty,) = rp.check_vram([r], apps, strikes)
    assert guilty["declared"] == 512 and guilty["used"] == 3070
    assert guilty["owner"] == "gpuq:j1" and guilty["usage_pid"] == 500


def test_a_sweep_back_under_the_line_clears_the_strike(tmp_path, monkeypatch):
    monkeypatch.setattr(rp, "descendants", lambda pid: set())
    r = _rec(tmp_path, "1.a.json", 500, 512)
    strikes = {}
    rp.check_vram([r], [{"pid": 500, "used_mb": 3070, "name": "t"}], strikes)
    rp.check_vram([r], [{"pid": 500, "used_mb": 400, "name": "t"}], strikes)
    assert strikes == {}


def test_an_exclusive_holder_is_never_over(tmp_path, monkeypatch):
    monkeypatch.setattr(rp, "descendants", lambda pid: set())
    r = _rec(tmp_path, "1.a.json", 500, None)
    apps = [{"pid": 500, "used_mb": 8000, "name": "t.py"}]
    rp.check_vram([r], apps, {})
    assert rp.check_vram([r], apps, {}) == []


def test_a_holders_children_count_toward_its_declaration(tmp_path, monkeypatch):
    """A trainer's dataloader workers hold VRAM under the same record."""
    from gpuqueue import ledger as lg
    # ledger.attribute() resolves `descendants` from ledger's own module
    # namespace (`from .procs import descendants`), not reaper's -- so the
    # reaper-side stub below never reaches it, and the tree must be stubbed
    # there too or the test would pass on a check_vram that ignores
    # descendants entirely. See the identical note in
    # test_a_ledgered_co_tenants_process_is_not_an_orphan above.
    monkeypatch.setattr(rp, "descendants",
                        lambda pid: {501, 502} if pid == 500 else set())
    monkeypatch.setattr(lg, "descendants",
                        lambda pid: {501, 502} if pid == 500 else set())
    r = _rec(tmp_path, "1.a.json", 500, 512)
    apps = [{"pid": 500, "used_mb": 200, "name": "t"},
            {"pid": 501, "used_mb": 200, "name": "w"},
            {"pid": 502, "used_mb": 200, "name": "w"}]
    strikes = {}
    rp.check_vram([r], apps, strikes)
    (guilty,) = rp.check_vram([r], apps, strikes)
    assert guilty["used"] == 600


def test_broken_attribution_convicts_nobody(tmp_path, monkeypatch):
    """Under MPS nvidia-smi reports the server, not its clients, so every
    process looks unledgered. That is a broken measurement, not a box full
    of intruders."""
    monkeypatch.setattr(rp, "descendants", lambda pid: set())
    r = _rec(tmp_path, "1.a.json", 500, 512)
    apps = [{"pid": 9999, "used_mb": 8000, "name": "nvidia-cuda-mps-server"}]
    strikes = {}
    rp.check_vram([r], apps, strikes)
    assert rp.check_vram([r], apps, strikes) == []
    assert strikes == {}


def test_a_departed_holder_stops_accruing_strikes(tmp_path, monkeypatch):
    monkeypatch.setattr(rp, "descendants", lambda pid: set())
    r = _rec(tmp_path, "1.a.json", 500, 512)
    strikes = {}
    rp.check_vram([r], [{"pid": 500, "used_mb": 3070, "name": "t"}], strikes)
    assert rp.check_vram([], [], strikes) == []
    assert strikes == {}


def test_enforce_vram_off_convicts_nobody(q, tmp_path, monkeypatch):
    cfg = RunnerConfig(queue_root=q.root, claim_dir=tmp_path,
                       kill_orphan_cuda=False, enforce_vram=False)
    monkeypatch.setattr(rp, "compute_apps",
                        lambda: [{"pid": 500, "used_mb": 8000, "name": "t"}])
    monkeypatch.setattr(rp, "descendants", lambda pid: set())
    strikes = {}
    reap(q, cfg, vram_strikes=strikes)
    assert reap(q, cfg, vram_strikes=strikes)["convicted"] == []


def test_a_convicted_holder_is_sigtermed_before_sigkill(monkeypatch):
    """The spec's §6: a convicted holder is SIGTERMed then SIGKILLed. Kill
    it outright and a trainer flushes no logs and writes no checkpoint, so
    the run and the evidence are both lost."""
    monkeypatch.setattr(rp, "descendants", lambda pid: set())
    sent = []
    real = rp._signal
    monkeypatch.setattr(rp, "_signal",
                        lambda pid, sig: sent.append(sig) or real(pid, sig))
    proc = _sp.Popen(["sleep", "30"])
    try:
        assert rp._kill_tree(proc.pid) is True
        proc.wait(timeout=5)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
    assert sent == [_signal.SIGTERM], "escalated past SIGTERM needlessly"


def test_kill_tree_does_not_wait_out_a_zombie(monkeypatch):
    """`pid_alive` is kill(pid, 0), which a zombie answers. The runner is
    the parent of what it convicts and does not reap it until `collect()`,
    so without the /proc check every conviction would burn both grace
    periods waiting for a process that has already exited."""
    monkeypatch.setattr(rp, "descendants", lambda pid: set())
    proc = _sp.Popen(["sh", "-c", "exit 0"])
    deadline = _time.monotonic() + 5
    while _live(proc.pid) and _time.monotonic() < deadline:
        _time.sleep(0.02)          # a zombie: exited, not yet waited for
    try:
        started = _time.monotonic()
        rp._kill_tree(proc.pid)
        assert _time.monotonic() - started < 1.0
    finally:
        proc.wait(timeout=5)


def test_an_unreadable_proc_stat_is_not_read_as_exited(monkeypatch):
    """The zombie check above must not turn into a blanket amnesty.

    Under a `hidepid` /proc mount the stat file cannot be opened at all.
    Reading that as "already exited" empties `_kill_tree`'s alive list on
    the SIGKILL pass, so a convicted trainer that blocks SIGTERM survives
    the watchdog outright. `pid_alive` is the authority on existence; an
    unreadable stat only means the *state* is unknown.
    """
    def denied(path, *a, **k):
        raise PermissionError(path)
    monkeypatch.setattr(rp, "open", denied, raising=False)
    assert rp._exited(_os.getpid()) is False


def test_a_conviction_whose_kill_failed_is_not_reported_as_killed(
        q, tmp_path, monkeypatch):
    """Records live in a claim directory shared with hand-run `gpu-claim`
    jobs, so a convicted holder can belong to another user. `_signal`
    swallows the EPERM, so that holder keeps running and keeps over-using
    the card -- while the runner logs `killed ...` and the sweep reports a
    conviction indistinguishable from one that landed.
    """
    from gpuqueue import ledger as lg
    cfg = RunnerConfig(queue_root=q.root, claim_dir=tmp_path,
                       kill_orphan_cuda=False, enforce_vram=True)
    rec = _rec(tmp_path, "1.a.json", 500, 512, owner="alice")
    monkeypatch.setattr(lg, "all_records", lambda d: [rec])
    monkeypatch.setattr(rp, "compute_apps",
                        lambda: [{"pid": 500, "used_mb": 3070, "name": "t"}])
    monkeypatch.setattr(rp, "_kill_tree", lambda pid: False)  # EPERM

    strikes = {}
    reap(q, cfg, vram_strikes=strikes)                    # first strike
    (guilty,) = reap(q, cfg, vram_strikes=strikes)["convicted"]

    assert guilty["killed"] is False


def test_a_holder_that_had_already_exited_is_not_reported_as_unkillable(
        monkeypatch):
    """`killed` is what stamps the runner's co-tenant window, and an
    over-using trainer usually OOMs itself within milliseconds of its
    victim -- around the very sweep that convicts it. Reading "nothing left
    to signal" as "could not kill" therefore denied the victim the one
    retry `_hit_by_a_convicted_co_tenant` exists to grant, and logged
    `COULD NOT KILL` over a process that was already gone.
    """
    monkeypatch.setattr(rp, "descendants", lambda pid: set())
    sent = []
    monkeypatch.setattr(rp, "_signal", lambda pid, sig: sent.append(sig))
    proc = _sp.Popen(["sh", "-c", "exit 0"])
    proc.wait(timeout=5)

    assert rp._kill_tree(proc.pid) is True
    assert sent == [], "signalled a process that had already exited"


def test_a_blind_sweep_does_not_bank_a_vram_strike(q, tmp_path, monkeypatch):
    """`WATCHDOG_STRIKES` counts *consecutive* sweeps over the declaration.
    A sweep whose nvidia-smi call fails measured nothing at all, so a
    strike left banked across it lets one spike now and one spike an hour
    later add up to a conviction -- effectively killing on a single sample,
    which is exactly what the strike count exists to prevent.
    """
    from gpuqueue import ledger as lg
    cfg = RunnerConfig(queue_root=q.root, claim_dir=tmp_path,
                       kill_orphan_cuda=False, enforce_vram=True)
    rec = _rec(tmp_path, "1.a.json", 500, 512, owner="alice")
    monkeypatch.setattr(lg, "all_records", lambda d: [rec])
    monkeypatch.setattr(rp, "_kill_tree", lambda pid: True)
    over = [{"pid": 500, "used_mb": 3070, "name": "t"}]
    seen = iter([over, None, over])
    monkeypatch.setattr(rp, "compute_apps", lambda: next(seen))

    strikes = {}
    reap(q, cfg, vram_strikes=strikes)          # over the line: one strike
    reap(q, cfg, vram_strikes=strikes)          # nvidia-smi says nothing
    assert strikes == {}
    assert reap(q, cfg, vram_strikes=strikes)["convicted"] == []


def test_a_conviction_whose_kill_landed_says_so(q, tmp_path, monkeypatch):
    from gpuqueue import ledger as lg
    cfg = RunnerConfig(queue_root=q.root, claim_dir=tmp_path,
                       kill_orphan_cuda=False, enforce_vram=True)
    rec = _rec(tmp_path, "1.a.json", 500, 512, owner="alice")
    monkeypatch.setattr(lg, "all_records", lambda d: [rec])
    monkeypatch.setattr(rp, "compute_apps",
                        lambda: [{"pid": 500, "used_mb": 3070, "name": "t"}])
    monkeypatch.setattr(rp, "_kill_tree", lambda pid: True)

    strikes = {}
    reap(q, cfg, vram_strikes=strikes)
    (guilty,) = reap(q, cfg, vram_strikes=strikes)["convicted"]

    assert guilty["killed"] is True


def test_sweep_spares_the_whole_tree_of_a_running_job(q, tmp_path, monkeypatch):
    """A runner restart splits the two halves of one reap() call.

    `release_stale` deletes the job's ledger record, because that record
    carries the *dead runner's* pid; `requeue_orphans` deliberately spares
    the job, because `spec.pid` is still alive. Between them the sweep sees
    the job's trainer -- a child of `spec.pid`, since spec.pid is a venv or
    shell wrapper, a torchrun, a dataloader parent -- with no record to
    charge it to. Protecting only the top-level pid SIGKILLs it.
    """
    import os
    cfg = RunnerConfig(queue_root=q.root, claim_dir=tmp_path / "claims",
                       kill_orphan_cuda=True, enforce_vram=False)
    q.submit(mkspec())
    spec = q.claim("j1")
    spec.pid = os.getpid()          # the job survived; its runner did not
    q._write(q.path_for("running", "j1"), spec)

    monkeypatch.setattr(rp, "compute_apps",
                        lambda: [{"pid": 4242, "used_mb": 100}])
    monkeypatch.setattr(rp, "descendants",
                        lambda pid: {4242} if pid == os.getpid() else set())
    killed = []
    monkeypatch.setattr(rp, "_kill", lambda pid: killed.append(pid) or True)

    assert reap(q, cfg)["killed_pids"] == []
    assert killed == []


def test_sweep_still_kills_a_process_under_nobody(q, tmp_path, monkeypatch):
    """The exemption above must not turn into a blanket amnesty."""
    import os
    cfg = RunnerConfig(queue_root=q.root, claim_dir=tmp_path / "claims",
                       kill_orphan_cuda=True, enforce_vram=False)
    q.submit(mkspec())
    spec = q.claim("j1")
    spec.pid = os.getpid()
    q._write(q.path_for("running", "j1"), spec)

    monkeypatch.setattr(rp, "compute_apps",
                        lambda: [{"pid": 9999, "used_mb": 100}])
    monkeypatch.setattr(rp, "descendants",
                        lambda pid: {4242} if pid == os.getpid() else set())
    monkeypatch.setattr(rp, "_kill", lambda pid: True)

    assert reap(q, cfg)["killed_pids"] == [9999]


# --- Which ledgers the sweep consulted (issue #19) -----------------------
#
# A SIGKILL from `kill_orphan_cuda` reaches the caller as `exit -9` with an
# empty stderr. The reporter of #19 spent a session ruling out OOM, host
# memory and the trainer itself before the reaper was even a suspect,
# because nothing anywhere said a kill had happened, let alone which
# exemption set was consulted before it.

def test_reap_reports_the_ledgers_it_exempted_from(q, tmp_path, monkeypatch):
    cfg = RunnerConfig(queue_root=q.root, kill_orphan_cuda=True,
                       claim_dir=tmp_path)
    result = reap(q, cfg)
    assert result["exemption_dirs"] == [str(d) for d in _cl.all_claim_dirs()]
    assert len(result["exemption_dirs"]) == 2, \
        "the environment directory and the default, which conftest keeps apart"


def test_no_exemption_dirs_when_the_sweep_did_not_run(q, tmp_path):
    cfg = RunnerConfig(queue_root=q.root, kill_orphan_cuda=True,
                       claim_dir=tmp_path)
    assert reap(q, cfg, include_orphan_cuda=False)["exemption_dirs"] == []


def test_no_exemption_dirs_when_only_the_vram_watchdog_ran(q, tmp_path):
    """The sweep runs for `enforce_vram` too, and that path consults no
    exemption at all. Reporting one here would have the runner name
    ledgers nothing read."""
    cfg = RunnerConfig(queue_root=q.root, kill_orphan_cuda=False,
                       enforce_vram=True, claim_dir=tmp_path)
    assert reap(q, cfg, vram_strikes={})["exemption_dirs"] == []


def test_no_exemption_dirs_when_the_process_list_is_invisible(
        q, tmp_path, monkeypatch):
    """A sweep that could not see the process list exempted nothing
    because it examined nothing."""
    monkeypatch.setattr(rp, "compute_apps", lambda: None)
    cfg = RunnerConfig(queue_root=q.root, kill_orphan_cuda=True,
                       claim_dir=tmp_path)
    assert reap(q, cfg)["exemption_dirs"] == []
