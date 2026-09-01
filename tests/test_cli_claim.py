import json
import pytest
from gpuqueue import cli_claim
from gpuqueue.claim import ClaimBusy
from gpuqueue.preflight import PreflightFailed
from gpuqueue.gpuid import GpuIdError
from pathlib import Path
from gpuqueue import claim

KEY = "4b8f2c1a-0000-0000-0000-000000000001"

@pytest.fixture(autouse=True)
def fake_gpu(tmp_path, monkeypatch):
    monkeypatch.setenv("GPU_CLAIM_DIR", str(tmp_path))
    monkeypatch.setattr(cli_claim, "gpu_key", lambda index=0: KEY)
    monkeypatch.setattr(cli_claim, "preflight",
                        lambda allow=None, directory=None, scope=None: None)
    # Otherwise every test here shells out to nvidia-smi for the capacity,
    # and answers differently on a box that has a card than on one that
    # does not.
    monkeypatch.setattr(cli_claim, "default_usable_mb", lambda index=0: 7676)

def test_runs_command_and_returns_its_exit_code():
    assert cli_claim.main(["--", "sh", "-c", "exit 0"]) == 0
    assert cli_claim.main(["--", "sh", "-c", "exit 3"]) == 3

def test_claim_released_after_command(tmp_path):
    cli_claim.main(["--", "true"])
    assert list(tmp_path.glob("*.lock.d/*.json")) == []

def test_busy_exits_75(monkeypatch, capsys):
    def busy(**kw):
        raise ClaimBusy("held by pid 999")
    monkeypatch.setattr(cli_claim, "gpu_claim", busy)
    assert cli_claim.main(["--", "true"]) == 75
    assert "999" in capsys.readouterr().err


def test_gpu_claim_passes_the_declaration_through(tmp_path, monkeypatch):
    seen = {}
    from contextlib import contextmanager

    @contextmanager
    def fake_claim(**kw):
        seen.update(kw)
        yield None

    monkeypatch.setattr(cli_claim, "gpu_claim", fake_claim)
    monkeypatch.setattr(cli_claim, "gpu_key", lambda index=0: "k")
    monkeypatch.setattr(cli_claim, "preflight", lambda **kw: None)
    cli_claim.main(["--vram-mb", "512", "--", "true"])
    assert seen["vram_mb"] == 512


def test_gpu_claim_without_a_declaration_takes_the_whole_card(tmp_path, monkeypatch):
    seen = {}
    from contextlib import contextmanager

    @contextmanager
    def fake_claim(**kw):
        seen.update(kw)
        yield None

    monkeypatch.setattr(cli_claim, "gpu_claim", fake_claim)
    monkeypatch.setattr(cli_claim, "gpu_key", lambda index=0: "k")
    monkeypatch.setattr(cli_claim, "preflight", lambda **kw: None)
    cli_claim.main(["--", "true"])
    assert seen["vram_mb"] is None

def test_preflight_failure_exits_69(monkeypatch, capsys):
    def fail(allow=None, directory=None, scope=None):
        raise PreflightFailed("pid 4321 train.py")
    monkeypatch.setattr(cli_claim, "preflight", fail)
    assert cli_claim.main(["--", "true"]) == 69
    assert "4321" in capsys.readouterr().err

def test_no_preflight_flag_skips_it(monkeypatch):
    def fail(allow=None, directory=None, scope=None):
        raise PreflightFailed("should not be called")
    monkeypatch.setattr(cli_claim, "preflight", fail)
    assert cli_claim.main(["--no-preflight", "--", "true"]) == 0

def test_no_gpu_exits_69(monkeypatch, capsys):
    def boom(index=0):
        raise GpuIdError("no CUDA device")
    monkeypatch.setattr(cli_claim, "gpu_key", boom)
    assert cli_claim.main(["--", "true"]) == 69
    assert "no CUDA device" in capsys.readouterr().err

def test_an_impossible_declaration_is_unavailable_not_tempfail(monkeypatch,
                                                               capsys):
    """75 (EX_TEMPFAIL) tells a wrapper "this may work later". There is no
    later in which a claim bigger than the whole card fits, and a wrapper
    that believes 75 would poll forever -- the same silent hang as --wait."""
    # 7676 usable, from the fake_gpu fixture.
    assert cli_claim.main(["--vram-mb", "99999", "--", "true"]) == 69
    assert "never be admitted" in capsys.readouterr().err


def test_status_prints_claims_as_json(tmp_path, capsys):
    (tmp_path / "x.lock.json").write_text(json.dumps(
        {"pid": 1, "owner": "me", "cmd": ["t"], "started_at": "2026-08-05T00:00:00Z"}))
    assert cli_claim.main(["--status"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["owner"] == "me"

def test_reap_removes_dead_claims(tmp_path, capsys):
    (tmp_path / "x.lock.json").write_text(json.dumps(
        {"pid": 4000000, "owner": "ghost", "cmd": ["t"],
         "started_at": "2026-08-05T00:00:00Z"}))
    assert cli_claim.main(["--reap"]) == 0
    assert not (tmp_path / "x.lock.json").exists()

def test_missing_command_exits_2(capsys):
    assert cli_claim.main([]) == 2


# --- pinning the child to the claimed card -----------------------------------
# Same reasoning as the runner: holding the lock says which card is yours, and
# the pin is what makes the child process able to act on that.

def test_child_is_pinned_to_the_claimed_card(tmp_path, monkeypatch):
    monkeypatch.setattr(cli_claim, "cuda_visible_value",
                        lambda index=0: f"GPU-{KEY}")
    out = tmp_path / "pin.txt"
    cli_claim.main(["--", "sh", "-c",
                    f"printf '%s' \"${{CUDA_VISIBLE_DEVICES-unset}}\" > {out}"])
    assert out.read_text() == f"GPU-{KEY}"


def test_pin_follows_the_requested_gpu_index(tmp_path, monkeypatch):
    seen = {}

    def fake(index=0):
        seen["index"] = index
        return "GPU-second-card"

    monkeypatch.setattr(cli_claim, "cuda_visible_value", fake)
    out = tmp_path / "pin.txt"
    cli_claim.main(["--gpu-index", "1", "--", "sh", "-c",
                    f"printf '%s' \"$CUDA_VISIBLE_DEVICES\" > {out}"])
    assert seen["index"] == 1
    assert out.read_text() == "GPU-second-card"


def test_an_explicit_caller_setting_is_not_clobbered(tmp_path, monkeypatch):
    # Unlike the runner, whose job can override the pin from inside its own
    # command, gpu-claim's caller can only express intent through the
    # environment it invokes us with. Treat that as deliberate.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    monkeypatch.setattr(cli_claim, "cuda_visible_value",
                        lambda index=0: f"GPU-{KEY}")
    out = tmp_path / "pin.txt"
    cli_claim.main(["--", "sh", "-c",
                    f"printf '%s' \"$CUDA_VISIBLE_DEVICES\" > {out}"])
    assert out.read_text() == "7"


def test_runs_unpinned_when_no_uuid_is_available(tmp_path, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(cli_claim, "cuda_visible_value", lambda index=0: None)
    out = tmp_path / "pin.txt"
    assert cli_claim.main(["--", "sh", "-c",
                           f"printf '%s' \"${{CUDA_VISIBLE_DEVICES-unset}}\" > {out}"]) == 0
    assert out.read_text() == "unset"


def test_an_empty_caller_setting_is_treated_as_unset(tmp_path, monkeypatch):
    # `export CUDA_VISIBLE_DEVICES=${SOMETHING}` with SOMETHING unset leaves
    # the variable present but empty. Read as a deliberate setting it means
    # "see no cards at all", so the child would hold the card and be unable
    # to use it. Nobody expresses that on purpose while claiming a GPU.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(cli_claim, "cuda_visible_value",
                        lambda index=0: f"GPU-{KEY}")
    out = tmp_path / "pin.txt"
    cli_claim.main(["--", "sh", "-c",
                    f"printf '%s' \"$CUDA_VISIBLE_DEVICES\" > {out}"])
    assert out.read_text() == f"GPU-{KEY}"


def test_an_empty_caller_setting_is_removed_when_there_is_nothing_to_pin(
        tmp_path, monkeypatch):
    # Degraded must mean "as it was before pinning existed". Leaving the empty
    # value in place would instead hand the child a card it cannot see.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(cli_claim, "cuda_visible_value", lambda index=0: None)
    out = tmp_path / "pin.txt"
    cli_claim.main(["--", "sh", "-c",
                    f"printf '%s' \"${{CUDA_VISIBLE_DEVICES-unset}}\" > {out}"])
    assert out.read_text() == "unset"


def test_a_caller_setting_naming_another_card_is_honoured_but_warned_about(
        tmp_path, monkeypatch, capsys):
    # The caller still wins -- their environment is the only way they can
    # express intent. But this is the exact shape of the collision the pin
    # exists to prevent: the lock is taken on the claimed card while the
    # command runs somewhere else, so it must not happen silently.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(cli_claim, "cuda_visible_value",
                        lambda index=0: f"GPU-{KEY}")
    out = tmp_path / "pin.txt"
    cli_claim.main(["--", "sh", "-c",
                    f"printf '%s' \"$CUDA_VISIBLE_DEVICES\" > {out}"])
    assert out.read_text() == "0"
    err = capsys.readouterr().err
    assert "CUDA_VISIBLE_DEVICES" in err
    assert "0" in err and f"GPU-{KEY}" in err


def test_a_caller_setting_naming_the_claimed_card_is_not_warned_about(
        tmp_path, monkeypatch, capsys):
    # Agreement is the common case for a wrapper that pins deliberately;
    # warning about it would train people to ignore the warning.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", f"GPU-{KEY}")
    monkeypatch.setattr(cli_claim, "cuda_visible_value",
                        lambda index=0: f"GPU-{KEY}")
    cli_claim.main(["--", "true"])
    assert "CUDA_VISIBLE_DEVICES" not in capsys.readouterr().err


def test_a_nonpositive_declaration_is_a_usage_error(tmp_path, capsys):
    """A typo'd `--vram-mb -5000` is a declaration that *subtracts* from
    the ledger's accounted total, so the next claimant is admitted past the
    end of the card. `gpuq submit` refuses it via `JobSpec.validate`; this
    path admitted it. Exit 2 rather than 69/75: nothing about the card is
    wrong, the command line is."""
    for bad in ("0", "-5000"):
        assert cli_claim.main(["--vram-mb", bad, "--", "true"]) == 2
        assert "--vram-mb" in capsys.readouterr().err
    assert list(tmp_path.glob("*.lock.d/*.json")) == []


def test_the_capacity_is_sized_for_the_card_being_claimed(monkeypatch):
    """`--gpu-index 1` keys the ledger on card 1 and pins the child to it,
    so the capacity it is admitted against has to be card 1's too."""
    seen = {}
    from contextlib import contextmanager

    @contextmanager
    def fake_claim(**kw):
        seen.update(kw)
        yield None

    monkeypatch.setattr(cli_claim, "default_usable_mb",
                        lambda index=0: {0: 24052, 1: 7676}[index])
    monkeypatch.setattr(cli_claim, "gpu_claim", fake_claim)
    cli_claim.main(["--gpu-index", "1", "--vram-mb", "4000", "--", "true"])
    assert seen["usable_mb"] == 7676


# --- Warning when the runner reads somewhere else (issue #19) ------------
#
# `gpu-claim` runs in an interactive shell; the runner is a supervisor
# unit. A shell never inherits a unit's environment, so the two resolve
# `$GPU_CLAIM_DIR` differently and neither one says so. The symptom is a
# SIGKILL with an empty stderr, and `--status` reporting the claim as
# healthy the whole time, because it reads the same directory the claim
# was written to.

from pathlib import Path
from gpuqueue import claim as _claim


def _runner_reads(tmp_path, monkeypatch, directory):
    cfg = tmp_path / "gpuq.toml"
    cfg.write_text(f'[queue]\nroot = "/q"\nclaim_dir = "{directory}"\n')
    monkeypatch.setenv("GPUQ_CONFIG", str(cfg))
    return cfg


def test_warns_when_the_runner_reads_another_directory(tmp_path, monkeypatch,
                                                       capsys):
    _runner_reads(tmp_path, monkeypatch, "/workspace/lock/gpu")

    cli_claim.main(["--", "true"])

    err = capsys.readouterr().err
    assert str(tmp_path) in err and "/workspace/lock/gpu" in err, err
    assert "GPU_CLAIM_DIR=/workspace/lock/gpu" in err, err


def test_the_warning_says_the_claim_is_not_counted(tmp_path, monkeypatch,
                                                   capsys):
    """The consequence that survives the exemption fix: the runner's
    ledger does not know this claim exists, so `gpu_max_jobs` and the VRAM
    accounting can admit a job straight on top of it."""
    _runner_reads(tmp_path, monkeypatch, "/workspace/lock/gpu")
    cli_claim.main(["--", "true"])
    assert "on top of it" in capsys.readouterr().err


def test_quiet_when_the_directories_agree(tmp_path, monkeypatch, capsys):
    _runner_reads(tmp_path, monkeypatch, str(tmp_path))
    cli_claim.main(["--", "true"])
    assert "warning" not in capsys.readouterr().err


def test_quiet_when_no_config_declares_a_directory(tmp_path, monkeypatch,
                                                   capsys):
    """The daemon then reads its own `$GPU_CLAIM_DIR`, which this process
    cannot see. A warning on a guess would fire on every correctly
    configured box, and one that is usually wrong is one people skip."""
    cfg = tmp_path / "gpuq.toml"
    cfg.write_text('[queue]\nroot = "/q"\n')
    monkeypatch.setenv("GPUQ_CONFIG", str(cfg))
    cli_claim.main(["--", "true"])
    assert "warning" not in capsys.readouterr().err


def test_the_warning_names_sigkill_only_for_a_third_directory(
        tmp_path, monkeypatch, capsys):
    """The reaper exempts its own `$GPU_CLAIM_DIR` *and* the default, so a
    claim that landed on the default is covered even though it diverges.
    Claiming otherwise would send an operator chasing a kill that cannot
    happen."""
    _runner_reads(tmp_path, monkeypatch, "/workspace/lock/gpu")

    monkeypatch.delenv("GPU_CLAIM_DIR", raising=False)   # -> DEFAULT_CLAIM_DIR
    cli_claim.main(["--", "true"])
    on_default = capsys.readouterr().err
    assert "warning" in on_default, on_default
    assert "SIGKILL" not in on_default, on_default

    monkeypatch.setenv("GPU_CLAIM_DIR", str(tmp_path / "third"))
    cli_claim.main(["--", "true"])
    on_third = capsys.readouterr().err
    assert "SIGKILL" in on_third, on_third


def test_status_warns_before_reporting_a_claim_the_reaper_cannot_see(
        tmp_path, monkeypatch, capsys):
    """`gpu-claim --status` reads the same directory the claim went to, so
    the operator's own health check confirms a claim the reaper cannot
    see. That is named in issue #19 as part of what made this hard."""
    _runner_reads(tmp_path, monkeypatch, "/workspace/lock/gpu")

    assert cli_claim.main(["--status"]) == 0

    out = capsys.readouterr()
    assert "/workspace/lock/gpu" in out.err, out.err
    assert out.out.strip().startswith("["), out.out


def test_reap_warns_too(tmp_path, monkeypatch, capsys):
    _runner_reads(tmp_path, monkeypatch, "/workspace/lock/gpu")
    assert cli_claim.main(["--reap"]) == 0
    assert "/workspace/lock/gpu" in capsys.readouterr().err


def _dead_claim(directory, owner="ghost", pid=4000000):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    p = directory / f"{owner}.lock.json"
    p.write_text(json.dumps({"pid": pid, "owner": owner, "cmd": ["t"],
                             "started_at": "2026-08-05T00:00:00Z"}))
    return p


def test_reap_sweeps_every_directory_a_claim_could_be_in(
        tmp_path, monkeypatch, capsys):
    """`--reap` is what an operator runs when a card looks held by nothing,
    and the record they are chasing is as likely to be under the default
    directory as under their own `$GPU_CLAIM_DIR` -- an interactive shell
    and a supervisor unit disagree about that variable, which is the whole
    of issue #19. Sweeping only one of them leaves the other's dead records
    exempting reused pids with nothing to clear them (issue #21).
    """
    default = tmp_path / "default"
    monkeypatch.setattr(claim, "DEFAULT_CLAIM_DIR", str(default))
    mine = _dead_claim(tmp_path, "ghost-mine")
    theirs = _dead_claim(default, "ghost-default")

    assert cli_claim.main(["--reap"]) == 0

    assert not mine.exists() and not theirs.exists()
    err = capsys.readouterr().err
    assert "ghost-mine" in err and "ghost-default" in err


def test_reap_names_a_record_it_may_not_remove(tmp_path, monkeypatch, capsys):
    """The sweep now reaches `/var/lock/gpu`, where another user's record
    is not this process's to unlink. Silence would report a card as freed
    while the record still holds it."""
    from gpuqueue import ledger as lg
    stuck = _dead_claim(tmp_path, "someone-else")
    monkeypatch.setattr(lg, "remove", _refuse_remove)

    assert cli_claim.main(["--reap"]) == 0

    err = capsys.readouterr().err
    assert str(stuck) in err, err
    assert "could not" in err.lower(), err
    assert "released stale claim" not in err, err


def _refuse_remove(rec):
    raise PermissionError(13, "Operation not permitted")


def test_reap_sweeps_the_directory_the_runner_is_configured_to_read(
        tmp_path, monkeypatch, capsys):
    """The third directory, and on the deployed box the important one.

    `docs/deploying.md` describes exactly this shape: the daemon reads
    `[queue].claim_dir` out of `$GPUQ_CONFIG`, while an interactive shell
    -- which never inherits the unit's `$GPU_CLAIM_DIR` -- lands on the
    default. Nearly every record on such a box is therefore under the
    runner's directory, and `all_claim_dirs()` cannot name it: it resolves
    the environment of *this* process. A `--reap` built from that list
    alone opens neither the directory holding the card nor says so; it
    prints nothing, which reads as "nothing was stale".

    The config file is the one thing both sides can read, and
    `_warn_if_the_runner_reads_elsewhere` already reads it on every
    invocation -- so the directory is in hand, and leaving it out of the
    sweep is the same drift between two lists that issue #21 was.
    """
    runners = tmp_path / "runnerdir"
    _runner_reads(tmp_path, monkeypatch, str(runners))
    ghost = _dead_claim(runners, "ghost-runner")

    assert cli_claim.main(["--reap"]) == 0

    assert not ghost.exists(), "the record holding the card was never read"
    assert "ghost-runner" in capsys.readouterr().err


def test_reap_says_why_it_could_not_remove_a_record(tmp_path, monkeypatch,
                                                    capsys):
    """`OSError` out of the unlink is not only "another user's file".

    A read-only remount of `/var/lock`, a stale NFS handle and an
    exhausted inode table all reach the same branch, and the errno is the
    only thing that tells them apart. Naming a cause the kernel did not
    give sends the one operator already mid-incident to go find an owner
    who is not the problem.
    """
    import errno
    from gpuqueue import ledger as lg
    stuck = _dead_claim(tmp_path, "mine")

    def _readonly(rec):
        raise OSError(errno.EROFS, "Read-only file system")

    monkeypatch.setattr(lg, "remove", _readonly)

    assert cli_claim.main(["--reap"]) == 0

    err = capsys.readouterr().err
    assert str(stuck) in err, err
    assert "Read-only file system" in err, err
    assert "another user" not in err, err


DOCKER_SCOPE = "/system.slice/docker-43faa0ee4d16.scope"


def test_scope_pid_is_passed_into_the_claim(tmp_path, monkeypatch):
    from contextlib import contextmanager
    seen = {}

    @contextmanager
    def fake_claim(**kw):
        seen.update(kw)
        yield None

    monkeypatch.setattr(cli_claim, "gpu_claim", fake_claim)
    monkeypatch.setattr(cli_claim.cgroups, "cgroup_of",
                        lambda pid, proc_root="/proc": DOCKER_SCOPE)
    assert cli_claim.main(["--scope-pid", "2818873", "--", "true"]) == 0
    assert seen["scope_pid"] == 2818873
    assert seen["scope_cgroup"] == DOCKER_SCOPE


def test_scope_pid_prints_the_resolved_scope(monkeypatch, capsys):
    # The operator's sanity check that they named a container and not the
    # box. Without it a wrong --scope-pid looks exactly like a right one.
    monkeypatch.setattr(cli_claim.cgroups, "cgroup_of",
                        lambda pid, proc_root="/proc": DOCKER_SCOPE)
    monkeypatch.setattr(cli_claim.cgroups, "scope_process_count",
                        lambda scope, proc_root="/proc": 3)
    cli_claim.main(["--scope-pid", "2818873", "--", "true"])
    err = capsys.readouterr().err
    assert DOCKER_SCOPE in err
    # The whole phrase: DOCKER_SCOPE already contains a "3", so a bare
    # `"3" in err` passes with the count deleted entirely.
    assert "3 live processes" in err


def test_scope_pid_naming_the_whole_box_is_refused(monkeypatch, capsys):
    monkeypatch.setattr(cli_claim.cgroups, "cgroup_of",
                        lambda pid, proc_root="/proc": "/")
    assert cli_claim.main(["--scope-pid", "1", "--", "true"]) == 2
    assert "whole box" in capsys.readouterr().err


def test_scope_pid_naming_a_login_session_is_refused(monkeypatch, capsys):
    monkeypatch.setattr(
        cli_claim.cgroups, "cgroup_of",
        lambda pid, proc_root="/proc":
            "/user.slice/user-0.slice/session-1848.scope")
    assert cli_claim.main(["--scope-pid", "2838576", "--", "true"]) == 2
    assert "login session" in capsys.readouterr().err


def test_scope_pid_that_is_not_running_is_refused(monkeypatch, capsys):
    monkeypatch.setattr(cli_claim.cgroups, "cgroup_of",
                        lambda pid, proc_root="/proc": None)
    monkeypatch.setattr(cli_claim, "pid_alive", lambda pid: False)
    assert cli_claim.main(["--scope-pid", "999999", "--", "true"]) == 2
    assert "not a running process" in capsys.readouterr().err


def test_scope_pid_on_a_cgroup_v1_box_is_refused(monkeypatch, capsys):
    # A live pid with no unified path is a v1 box, and the operator's
    # next move is nothing like "fix the pid".
    monkeypatch.setattr(cli_claim.cgroups, "cgroup_of",
                        lambda pid, proc_root="/proc": None)
    monkeypatch.setattr(cli_claim, "pid_alive", lambda pid: True)
    assert cli_claim.main(["--scope-pid", "2818873", "--", "true"]) == 2
    assert "cgroup v2" in capsys.readouterr().err
