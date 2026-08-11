import json
import threading
import time
import pytest
from pathlib import Path
from gpuqueue import runner as rn
from gpuqueue.runner import Runner
from gpuqueue.config import RunnerConfig, ProjectConfig
from gpuqueue.spec import JobSpec
from gpuqueue.git_ops import git

def _origin(tmp_path):
    o = tmp_path / "origin"
    o.mkdir()
    git(["init", "-q", "-b", "main"], cwd=o)
    git(["config", "user.email", "t@t"], cwd=o)
    git(["config", "user.name", "t"], cwd=o)
    (o / "a.txt").write_text("one\n")
    git(["add", "a.txt"], cwd=o)
    git(["commit", "-qm", "first"], cwd=o)
    return o, git(["rev-parse", "HEAD"], cwd=o).strip()

@pytest.fixture
def env(tmp_path, monkeypatch):
    origin, sha = _origin(tmp_path)
    monkeypatch.setenv("GPU_CLAIM_DIR", str(tmp_path / "claims"))
    cfg = RunnerConfig(
        queue_root=tmp_path / "queue", cpu_slots=2, poll_interval_s=0.01,
        claim_dir=tmp_path / "claims", kill_orphan_cuda=False,
        projects={"p": ProjectConfig(name="p", remote=str(origin),
                                     checkout=tmp_path / "checkout",
                                     commit_artifacts=True)})
    monkeypatch.setattr(rn, "gpu_key", lambda index=0: "test-uuid")
    monkeypatch.setattr(rn, "preflight", lambda allow=None: None)
    r = Runner(cfg)
    return r, sha

def submit(r, sha, job_id, cmd, lane="cpu", artifacts=(), timeout_s=30):
    r.queue.submit(JobSpec.from_dict(dict(
        id=job_id, lane=lane, project="p", commit=sha, branch="main",
        cmd=list(cmd), artifacts=list(artifacts), timeout_s=timeout_s)))

def drain(r, limit=30.0):
    """Tick until nothing is pending or running — what run_forever does,
    without the sleep."""
    deadline = time.monotonic() + limit
    while time.monotonic() < deadline:
        r.tick()
        if not r.active and not r.queue.list_state("pending"):
            return True
        time.sleep(0.02)
    raise AssertionError("runner did not drain the queue in time")

def test_runs_a_cpu_job_to_done(env):
    r, sha = env
    submit(r, sha, "j1", ["sh", "-c", "echo hi"])
    drain(r)
    assert (r.queue.root / "done" / "j1.json").exists()

def test_job_runs_in_a_worktree_at_the_pinned_commit(env):
    r, sha = env
    submit(r, sha, "j1", ["sh", "-c", "cat a.txt"])
    drain(r)
    out, _ = r.queue.log_paths("j1")
    assert out.read_text().strip() == "one"

def test_failing_job_lands_in_failed_with_stderr_tail(env):
    r, sha = env
    submit(r, sha, "j1", ["sh", "-c", "echo bad >&2; exit 4"])
    drain(r)
    body = json.loads((r.queue.root / "failed" / "j1.json").read_text())
    assert body["exit_code"] == 4 and "bad" in body["error"]

def test_cpu_slots_are_respected(env):
    r, sha = env
    for i in range(4):
        submit(r, sha, f"j{i}", ["sleep", "5"])
    assert len(r.admit()) == 2
    assert r.admit() == []
    r.shutdown()

def test_only_one_gpu_job_runs_at_a_time(env):
    r, sha = env
    submit(r, sha, "g1", ["sleep", "5"], lane="gpu")
    submit(r, sha, "g2", ["sleep", "5"], lane="gpu")
    assert len(r.admit()) == 1
    assert r.admit() == []
    r.shutdown()

def test_gpu_and_cpu_lanes_run_concurrently(env):
    r, sha = env
    submit(r, sha, "g1", ["sleep", "5"], lane="gpu")
    submit(r, sha, "c1", ["sleep", "5"], lane="cpu")
    assert sorted(r.admit()) == ["c1", "g1"]
    r.shutdown()

def test_gpu_job_holds_a_claim_file_while_running(env, tmp_path):
    r, sha = env
    submit(r, sha, "g1", ["sleep", "5"], lane="gpu")
    r.admit()
    assert list((tmp_path / "claims").glob("*.lock.json"))
    r.shutdown()

def test_gpu_claim_is_released_when_the_job_ends(env, tmp_path):
    r, sha = env
    submit(r, sha, "g1", ["true"], lane="gpu")
    drain(r)
    assert list((tmp_path / "claims").glob("*.lock.json")) == []

def test_a_busy_card_leaves_the_job_pending_not_failed(env, tmp_path):
    """An outside gpu-claim holder must not consume the queued job."""
    from gpuqueue.claim import gpu_claim
    r, sha = env
    submit(r, sha, "g1", ["true"], lane="gpu")
    with gpu_claim(key="test-uuid", owner="outsider", directory=tmp_path / "claims"):
        assert r.admit() == []
        assert (r.queue.root / "pending" / "g1.json").exists()
    drain(r)
    assert (r.queue.root / "done" / "g1.json").exists()

def test_running_job_records_its_pid(env):
    r, sha = env
    submit(r, sha, "j1", ["sleep", "5"])
    r.admit()
    body = json.loads((r.queue.root / "running" / "j1.json").read_text())
    assert body["pid"] > 0
    r.shutdown()

def test_timeout_marks_failed_and_does_not_retry(env):
    r, sha = env
    submit(r, sha, "j1", ["sleep", "30"], timeout_s=1)
    drain(r, limit=60)
    body = json.loads((r.queue.root / "failed" / "j1.json").read_text())
    assert "timeout" in body["error"].lower()
    assert body["attempts"] == 0

def test_artifacts_are_committed_by_the_main_loop(env):
    r, sha = env
    submit(r, sha, "j1", ["sh", "-c", "mkdir -p runs && echo '{}' > runs/s.json"],
           artifacts=["runs/s.json"])
    drain(r)
    assert (r.cfg.projects["p"].checkout / "runs" / "s.json").exists()

def test_missing_artifact_fails_the_job(env):
    r, sha = env
    submit(r, sha, "j1", ["true"], artifacts=["runs/never.json"])
    drain(r)
    body = json.loads((r.queue.root / "failed" / "j1.json").read_text())
    assert "never.json" in body["error"]

def test_worktree_removed_after_job(env):
    r, sha = env
    submit(r, sha, "j1", ["true"])
    drain(r)
    assert not r.queue.work_dir("j1").exists()

def test_unknown_project_fails_fast(env):
    r, sha = env
    r.queue.submit(JobSpec.from_dict(dict(
        id="j1", lane="cpu", project="nope", commit=sha, branch="main",
        cmd=["true"], artifacts=[], timeout_s=30)))
    r.tick()
    body = json.loads((r.queue.root / "failed" / "j1.json").read_text())
    assert "nope" in body["error"]

def test_unstartable_command_fails_the_job(env):
    r, sha = env
    submit(r, sha, "j1", ["definitely-not-a-real-binary"])
    drain(r)
    body = json.loads((r.queue.root / "failed" / "j1.json").read_text())
    assert "definitely-not-a-real-binary" in body["error"]

def test_shutdown_leaves_killed_jobs_for_the_reaper(env):
    """Graceful stop does not decide the job's fate — it clears the pid and
    leaves it in running/, which is exactly the state the reaper already
    knows how to requeue once."""
    r, sha = env
    submit(r, sha, "j1", ["sleep", "30"])
    r.admit()
    r.shutdown()
    body = json.loads((r.queue.root / "running" / "j1.json").read_text())
    assert body["pid"] is None
    assert r.active == {}

def test_the_runner_spawns_no_threads(env):
    """Concurrency is Popen, not threads. A thread here would put a worker
    and the loop on the same queue files and the same JobSpec objects."""
    r, sha = env
    before = threading.active_count()
    for i in range(3):
        submit(r, sha, f"j{i}", ["sh", "-c", "sleep 0.2; echo ok"])
    r.tick()
    assert threading.active_count() == before
    drain(r)
    assert threading.active_count() == before

def test_shutdown_survives_a_job_file_removed_by_hand(env):
    """The queue is advertised as repairable with mv. A spec that moved out
    from under the runner must not crash shutdown, and above all must not
    strand another job's hold on the card."""
    r, sha = env
    submit(r, sha, "g1", ["sleep", "30"], lane="gpu")
    submit(r, sha, "c1", ["sleep", "30"], lane="cpu")
    r.admit()
    (r.queue.root / "running" / "c1.json").unlink()

    r.shutdown()
    assert r.active == {}
    # the GPU claim was released despite the other job's missing file
    assert list((r.cfg.claim_dir).glob("*.lock.json")) == []


def test_an_idle_runner_still_reaps_an_abandoned_job(env):
    """The failure this fixes: a job left in running/ by a dead runner was
    only recovered when some *other* job completed or the runner restarted.
    On an idle box -- precisely when there is something to recover -- nothing
    ever happened."""
    r, sha = env
    submit(r, sha, "j1", ["true"])
    spec = r.queue.claim("j1")
    spec.pid = 4000000          # a runner that is gone
    spec.runner_pid = 4000001
    r.queue.update(spec)

    r.tick()                    # nothing else happening, nothing completing

    # requeued once and picked straight back up, all in the one tick
    state, recovered = r.queue.find("j1")
    assert recovered.attempts == 1
    assert state in ("pending", "running", "done")
    assert drain(r)

def test_the_cuda_sweep_is_throttled(env, monkeypatch):
    """The cheap reaps run every tick; the nvidia-smi sweep must not."""
    r, sha = env
    calls = []
    monkeypatch.setattr(rn, "reap",
                        lambda q, c, active_ids=None, include_orphan_cuda=True:
                        calls.append(include_orphan_cuda) or {})
    r.cfg.orphan_cuda_interval_s = 3600
    for _ in range(5):
        r.tick()
    assert len(calls) == 5, "the cheap reap must run on every tick"
    assert calls[0] is True, "the first tick sweeps"
    assert not any(calls[1:]), "later ticks inside the interval must not sweep"

def test_the_cuda_sweep_runs_again_once_the_interval_passes(env, monkeypatch):
    r, sha = env
    calls = []
    monkeypatch.setattr(rn, "reap",
                        lambda q, c, active_ids=None, include_orphan_cuda=True:
                        calls.append(include_orphan_cuda) or {})
    r.cfg.orphan_cuda_interval_s = 0    # everything is always due
    r.tick(); r.tick()
    assert calls == [True, True]

def test_a_missing_artifact_raises_caller_error_not_a_gpuq_bug(env):
    """A declared artifact the job never produced is the caller's mistake
    wearing a gpuq traceback -- the one case the classifier cannot infer."""
    from gpuqueue.bugreport import CallerError, is_gpuq_fault
    r, sha = env
    project = r.cfg.projects["p"]
    with pytest.raises(CallerError) as caught:
        r._collect_artifacts(
            JobSpec(id="j1", lane="cpu", project="p", commit=sha,
                    branch="main", cmd=["true"], artifacts=["runs/never.json"]),
            project, r.queue.work_dir("j1"))
    assert "never.json" in str(caught.value)
    assert is_gpuq_fault(caught.value) is False

@pytest.fixture
def filed(monkeypatch):
    """Capture what the runner would have filed, without touching GitHub."""
    calls = []

    def fake(cfg, exc, phase, spec=None, queue_counts=None, now=None):
        calls.append({"phase": phase, "exc": exc, "spec": spec,
                      "counts": queue_counts})
        return "filed"

    monkeypatch.setattr(rn.bugfiler, "file_bug", fake)
    return calls


def _enable(r):
    from gpuqueue.config import AutofixConfig
    r.cfg.autofix = AutofixConfig(enabled=True, repo="you/gpuq")


def test_a_checkout_failure_files_a_bug(env, filed, monkeypatch):
    r, sha = env
    _enable(r)
    monkeypatch.setattr(r, "_prepare_workdir",
                        lambda spec, project: (_ for _ in ()).throw(
                            rn.git_ops.GitError("worktree add failed")))
    submit(r, sha, "j1", ["true"])
    r.admit()
    assert [c["phase"] for c in filed] == ["checkout"]
    assert filed[0]["spec"].id == "j1"


def test_the_job_still_fails_normally_when_a_bug_is_filed(env, filed,
                                                          monkeypatch):
    """Filing is additive. The job's own fate is unchanged."""
    r, sha = env
    _enable(r)
    monkeypatch.setattr(r, "_prepare_workdir",
                        lambda spec, project: (_ for _ in ()).throw(
                            rn.git_ops.GitError("worktree add failed")))
    submit(r, sha, "j1", ["true"])
    r.admit()
    body = json.loads((r.queue.root / "failed" / "j1.json").read_text())
    assert "checkout failed" in body["error"]


def test_a_reap_failure_files_a_bug_and_still_crashes_the_tick(env, filed,
                                                               monkeypatch):
    """Reporting must not change what supervisor sees. A broken reaper still
    takes the runner down and gets restarted; it just leaves evidence now."""
    r, sha = env
    _enable(r)
    monkeypatch.setattr(rn, "reap",
                        lambda *a, **k: (_ for _ in ()).throw(
                            RuntimeError("reaper exploded")))
    with pytest.raises(RuntimeError, match="reaper exploded"):
        r.tick()
    assert [c["phase"] for c in filed] == ["reap"]


def test_a_collect_failure_files_under_collect_not_execute(env, filed,
                                                           monkeypatch):
    """`tick` wraps `collect` and `_launch` reports StartFailed, and both
    said "execute" once -- so the issue title read the same for a job that
    would not start and for the pass that finishes jobs which did. The
    signature told them apart by frame names; the human reading the title
    could not."""
    r, sha = env
    _enable(r)
    monkeypatch.setattr(r, "collect",
                        lambda: (_ for _ in ()).throw(
                            RuntimeError("collect exploded")))
    with pytest.raises(RuntimeError, match="collect exploded"):
        r.tick()
    assert [c["phase"] for c in filed] == ["collect"]


def test_a_preflight_failure_files_only_the_inner_phase(env, filed,
                                                        monkeypatch):
    """`_take_card` already reports and re-raises a preflight exception
    before it reaches `admit`, which `tick` also wraps in `_phase`. Without
    the reported-tag, the same exception would be filed again there under a
    different phase string -- two issues, one crash. The inner, more
    specific site must win; the outer one must see it is already done."""
    r, sha = env
    _enable(r)
    monkeypatch.setattr(rn, "preflight",
                        lambda: (_ for _ in ()).throw(
                            RuntimeError("preflight exploded")))
    submit(r, sha, "j1", ["true"], lane="gpu")
    with pytest.raises(RuntimeError, match="preflight exploded"):
        r.tick()
    assert [c["phase"] for c in filed] == ["preflight"]


def test_a_missing_artifact_reaches_the_filer_as_caller_error(env, filed):
    """Caller fault. file_bug is still called -- it owns the decision, and
    the runner must not pre-judge -- but it is handed a CallerError, which
    Task 7 turns into 'caller-fault' and no issue."""
    from gpuqueue.bugreport import CallerError
    r, sha = env
    _enable(r)
    submit(r, sha, "j1", ["true"], artifacts=["runs/never.json"])
    drain(r)
    assert [type(c["exc"]) for c in filed] == [CallerError]
    assert filed[0]["phase"] == "artifacts"


def test_an_ordinary_job_failure_files_nothing(env, filed):
    """exit N, timeout and OOM never reach the filer at all: they are not
    exceptions and gpuq's code did not raise."""
    r, sha = env
    _enable(r)
    submit(r, sha, "j1", ["sh", "-c", "echo bad >&2; exit 4"])
    drain(r)
    assert filed == []


def test_a_timeout_files_nothing(env, filed):
    r, sha = env
    _enable(r)
    submit(r, sha, "j1", ["sleep", "30"], timeout_s=1)
    drain(r, limit=60)
    assert filed == []


def test_an_unstartable_command_reaches_the_filer_as_start_failed(env, filed):
    """The filer classifies it out by errno; the runner does not pre-judge."""
    from gpuqueue.executor import StartFailed
    r, sha = env
    _enable(r)
    submit(r, sha, "j1", ["definitely-not-a-real-binary"])
    drain(r)
    assert [type(c["exc"]) for c in filed] == [StartFailed]
    assert filed[0]["phase"] == "execute"


def test_the_report_carries_the_queue_counts(env, filed, monkeypatch):
    r, sha = env
    _enable(r)
    monkeypatch.setattr(r, "_prepare_workdir",
                        lambda spec, project: (_ for _ in ()).throw(
                            rn.git_ops.GitError("nope")))
    submit(r, sha, "j1", ["true"])
    r.admit()
    assert set(filed[0]["counts"]) == {"pending", "running", "done", "failed"}


def test_a_broken_filer_never_breaks_the_queue(env, monkeypatch):
    """The whole point. gh missing, token wrong, GitHub down -- jobs run."""
    r, sha = env
    _enable(r)
    monkeypatch.setattr(rn.bugfiler, "file_bug",
                        lambda *a, **k: (_ for _ in ()).throw(
                            rn.bugfiler.GhError("gh is not installed")))
    submit(r, sha, "j1", ["true"], artifacts=["runs/never.json"])
    drain(r)
    assert (r.queue.root / "failed" / "j1.json").exists()


def test_filing_is_skipped_entirely_when_autofix_is_off(env, filed):
    r, sha = env  # autofix left at its default, disabled
    submit(r, sha, "j1", ["true"], artifacts=["runs/never.json"])
    drain(r)
    assert filed == []


def test_a_cooldown_bounds_gh_calls_when_every_pending_job_fails_the_same_way(
        env, filed, monkeypatch):
    """admit() attempts every pending job in one tick, and a job that never
    reaches self.active (a failed _launch) never shrinks capacity, so a
    broken git_ops with many queued jobs would otherwise call file_bug --
    up to three `gh` subprocesses at 30s each -- once per job in the same
    tick. That would stall poll_job for the whole window and leave running
    jobs' deadlines unenforced. The in-process per-signature cooldown must
    keep one admit() call down to a single file_bug call for one bug."""
    r, sha = env
    _enable(r)
    monkeypatch.setattr(r, "_prepare_workdir",
                        lambda spec, project: (_ for _ in ()).throw(
                            rn.git_ops.GitError("worktree add failed")))
    for i in range(5):
        submit(r, sha, f"j{i}", ["true"])
    r.admit()
    assert len(filed) == 1


def test_the_cooldown_expires_and_reports_again(env, filed, monkeypatch):
    r, sha = env
    _enable(r)
    r.cfg.autofix.report_cooldown_s = 0
    monkeypatch.setattr(r, "_prepare_workdir",
                        lambda spec, project: (_ for _ in ()).throw(
                            rn.git_ops.GitError("worktree add failed")))
    for i in range(3):
        submit(r, sha, f"j{i}", ["true"])
    r.admit()
    assert len(filed) == 3


def test_an_unpushed_commit_is_a_caller_error_not_a_checkout_bug(env, filed):
    """'I submitted at a commit I forgot to push' is the likeliest caller
    mistake in this system. Unchecked, it would reach `add_worktree`, fail
    exactly like a genuine git_ops fault, and file a gpuq bug -- burning one
    of three daily dispatches on a Claude run that can only close it. The
    classifier must see a CallerError, not a raw GitError."""
    from gpuqueue.bugreport import CallerError
    r, sha = env
    _enable(r)
    unpushed = "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"
    submit(r, unpushed, "j1", ["true"])
    r.admit()
    body = json.loads((r.queue.root / "failed" / "j1.json").read_text())
    assert "push it first" in body["error"]
    assert unpushed in body["error"]
    assert [type(c["exc"]) for c in filed] == [CallerError]
    assert filed[0]["phase"] == "checkout"


def test_settle_finishes_the_job_before_filing_an_artifact_bug(env, monkeypatch):
    """`_report_bug` can block for ~250s on `gh`. The job is already out of
    `self.active` by the time `_settle` runs, so if that call happened
    before `queue.finish`, a crash in the filing window would leave a
    completed job stranded in running/ with a stale pid -- and the reaper
    would requeue and re-run work that already finished. Filing must happen
    after the job is recorded finished, matching both `_launch` sites."""
    r, sha = env
    _enable(r)
    order = []
    real_finish = r.queue.finish

    def spy_finish(spec, ok):
        order.append("finish")
        return real_finish(spec, ok=ok)

    def spy_report(exc, phase, spec=None):
        order.append("report_bug")

    monkeypatch.setattr(r.queue, "finish", spy_finish)
    monkeypatch.setattr(r, "_report_bug", spy_report)
    submit(r, sha, "j1", ["true"], artifacts=["runs/never.json"])
    drain(r)
    assert order == ["finish", "report_bug"]


def test_a_throttled_bug_still_carries_the_auto_label(env, monkeypatch):
    """A `label:gpuq-auto` triage query must find a throttled bug too -- it
    is the same structural evidence as a dispatched one, only the budget
    held the run back. bugfiler.file_bug owns the label choice; this
    confirms the runner's call into it produces a labels list containing
    both, via the real file_bug (not the `filed` fixture's stub)."""
    from gpuqueue import bugfiler
    from gpuqueue.config import AutofixConfig

    r, sha = env
    r.cfg.autofix = AutofixConfig(enabled=True, repo="you/gpuq",
                                  max_dispatches_per_day=0,
                                  state_file=r.queue.root / "autofix.json")
    seen_labels = []

    def fake_gh(cfg, args, stdin=None):
        if args[:2] == ["issue", "list"] or args[:2] == ["pr", "list"]:
            return "[]"
        if args[:2] == ["issue", "create"]:
            seen_labels.append([args[i + 1] for i, a in enumerate(args)
                                if a == "--label"])
            return "https://github.com/you/gpuq/issues/1\n"
        return ""

    monkeypatch.setattr(bugfiler, "_gh", fake_gh)
    monkeypatch.setattr(r, "_prepare_workdir",
                        lambda spec, project: (_ for _ in ()).throw(
                            rn.git_ops.GitError("worktree add failed")))
    submit(r, sha, "j1", ["true"])
    r.admit()
    assert seen_labels == [[bugfiler.AUTO_LABEL, bugfiler.THROTTLED_LABEL]]


# --- pinning a gpu job to the card it was allocated --------------------------
# The claim serializes the card; the pin makes the allocation binding on the
# job. Without it a consumer that refuses to guess a device (the pattern
# `resolve_device(..., strict=True)` implements) cannot run under the queue at
# all, because nothing in the environment says which card it was given.

def _pin_seen_by(r, sha, tmp_path, lane):
    out = tmp_path / f"pin_{lane}.txt"
    submit(r, sha, f"pin_{lane}",
           ["sh", "-c", f"printf '%s' \"${{CUDA_VISIBLE_DEVICES-unset}}\" > {out}"],
           lane=lane)
    drain(r)
    return out.read_text()


def test_gpu_job_is_pinned_to_the_claimed_card(env, tmp_path, monkeypatch):
    r, sha = env
    monkeypatch.setattr(rn, "cuda_visible_value", lambda index=0: "GPU-abc123")
    assert _pin_seen_by(r, sha, tmp_path, "gpu") == "GPU-abc123"


def test_cpu_job_is_not_pinned(env, tmp_path, monkeypatch):
    # A cpu-lane job never took the card, so it must not be handed one.
    r, sha = env
    monkeypatch.setattr(rn, "cuda_visible_value", lambda index=0: "GPU-abc123")
    assert _pin_seen_by(r, sha, tmp_path, "cpu") == "unset"


def test_gpu_job_runs_unpinned_when_no_uuid_is_available(env, tmp_path,
                                                         monkeypatch):
    # Degraded but not broken: a box whose driver reports no uuid still runs
    # jobs, exactly as it did before pinning existed.
    r, sha = env
    monkeypatch.setattr(rn, "cuda_visible_value", lambda index=0: None)
    assert _pin_seen_by(r, sha, tmp_path, "gpu") == "unset"


def test_the_job_may_override_the_pin(env, tmp_path, monkeypatch):
    # The pin is a default, not a cage: a job that names its own mapping wins,
    # which is what makes the existing `export CUDA_VISIBLE_DEVICES=...` driver
    # scripts keep working unchanged.
    r, sha = env
    monkeypatch.setattr(rn, "cuda_visible_value", lambda index=0: "GPU-abc123")
    out = tmp_path / "override.txt"
    submit(r, sha, "override",
           ["sh", "-c", f"export CUDA_VISIBLE_DEVICES=7; printf '%s' \"$CUDA_VISIBLE_DEVICES\" > {out}"],
           lane="gpu")
    drain(r)
    assert out.read_text() == "7"
