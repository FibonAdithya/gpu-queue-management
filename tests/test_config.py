import textwrap
from pathlib import Path

import pytest
from gpuqueue.config import (load_config, vram_policy, max_holders,
                             claim_dir_setting, ConfigError)
from gpuqueue.ledger import DEFAULT_RESERVE_MB

TOML = """
[queue]
root = "/workspace/queue"
cpu_slots = 4

[project.myproject]
remote   = "git@github.com:you/myproject.git"
checkout = "/workspace/checkouts/myproject"
venv     = "/workspace/checkouts/myproject/.venv"
commit_artifacts = true
"""

def _write(tmp_path, text):
    p = tmp_path / "gpuq.toml"
    p.write_text(text)
    return p

def write_and_load(tmp_path, body):
    # Indented triple-quoted TOML in the test body needs dedenting before
    # it's valid at column 0; _write() alone doesn't do that.
    return load_config(_write(tmp_path, textwrap.dedent(body)))

def test_loads_queue_settings(tmp_path):
    cfg = load_config(_write(tmp_path, TOML))
    assert str(cfg.queue_root) == "/workspace/queue"
    assert cfg.cpu_slots == 4

def test_loads_project(tmp_path):
    proj = load_config(_write(tmp_path, TOML)).projects["myproject"]
    assert proj.name == "myproject"
    assert proj.commit_artifacts is True
    assert str(proj.venv).endswith(".venv")

def test_defaults_applied(tmp_path):
    cfg = load_config(_write(tmp_path, '[queue]\nroot = "/q"\n'))
    assert cfg.cpu_slots == 4
    assert cfg.poll_interval_s == 2.0
    assert cfg.kill_orphan_cuda is True
    assert cfg.projects == {}

def test_missing_root_rejected(tmp_path):
    with pytest.raises(ConfigError, match="root"):
        load_config(_write(tmp_path, "[queue]\n"))

def test_project_without_checkout_rejected(tmp_path):
    bad = '[queue]\nroot="/q"\n[project.p]\nremote="git@x:y.git"\n'
    with pytest.raises(ConfigError, match="checkout"):
        load_config(_write(tmp_path, bad))

def test_zero_cpu_slots_rejected(tmp_path):
    with pytest.raises(ConfigError, match="cpu_slots"):
        load_config(_write(tmp_path, '[queue]\nroot="/q"\ncpu_slots=0\n'))

def test_missing_file_raises_config_error(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.toml")

def test_example_config_in_repo_is_loadable():
    from pathlib import Path
    load_config(Path(__file__).resolve().parents[1] / "gpuq.example.toml")


SPLIT = """
[queue]
root = "/workspace/queue"

[project.myproject]
remote   = "git@github.com:you/myproject.git"
checkout = "/workspace/checkouts/myproject"
commit_artifacts = true
results_remote   = "git@github.com:you/myproject-results.git"
results_checkout = "/workspace/checkouts/myproject-results"
results_branch   = "main"
"""

def test_loads_a_results_repo(tmp_path):
    proj = load_config(_write(tmp_path, SPLIT)).projects["myproject"]
    assert proj.results_remote.endswith("myproject-results.git")
    assert str(proj.results_checkout).endswith("myproject-results")
    assert proj.results_branch == "main"

def test_results_defaults_to_none(tmp_path):
    proj = load_config(_write(tmp_path, TOML)).projects["myproject"]
    assert proj.results_remote is None and proj.results_checkout is None

def test_half_a_results_repo_is_rejected(tmp_path):
    """One without the other has nowhere to publish."""
    bad = SPLIT.replace('results_checkout = "/workspace/checkouts/myproject-results"\n', "")
    with pytest.raises(ConfigError, match="results_remote and results_checkout"):
        load_config(_write(tmp_path, bad))


AUTOFIX = TOML + """
[autofix]
enabled = true
repo = "you/gpu-queue-management"
"""

def test_autofix_is_off_unless_declared(tmp_path):
    cfg = load_config(_write(tmp_path, TOML))
    assert cfg.autofix.enabled is False
    assert cfg.autofix.repo is None

def test_autofix_loads(tmp_path):
    af = load_config(_write(tmp_path, AUTOFIX)).autofix
    assert af.enabled is True
    assert af.repo == "you/gpu-queue-management"

def test_autofix_defaults(tmp_path):
    af = load_config(_write(tmp_path, AUTOFIX)).autofix
    assert af.max_dispatches_per_day == 3
    assert af.closed_lookback_days == 30
    assert af.token_env == "GPUQ_GITHUB_TOKEN"

def test_autofix_state_file_defaults_under_the_queue_root(tmp_path):
    af = load_config(_write(tmp_path, AUTOFIX)).autofix
    assert str(af.state_file) == "/workspace/queue/autofix.json"

def test_enabled_autofix_without_a_repo_is_rejected(tmp_path):
    text = TOML + '\n[autofix]\nenabled = true\n'
    with pytest.raises(ConfigError, match="repo"):
        load_config(_write(tmp_path, text))

def test_a_repo_must_look_like_owner_slash_name(tmp_path):
    text = TOML + '\n[autofix]\nenabled = true\nrepo = "https://github.com/a/b"\n'
    with pytest.raises(ConfigError, match="owner/name"):
        load_config(_write(tmp_path, text))


def test_capacity_defaults(tmp_path):
    cfg = write_and_load(tmp_path, """
        [queue]
        root = "/q"
    """)
    assert cfg.gpu_vram_mb is None      # discovered from nvidia-smi
    assert cfg.gpu_vram_reserve_mb == 512
    assert cfg.gpu_max_jobs == 2
    assert cfg.enforce_vram is True


def test_capacity_keys_are_read(tmp_path):
    cfg = write_and_load(tmp_path, """
        [queue]
        root = "/q"
        gpu_vram_mb = 8188
        gpu_vram_reserve_mb = 1024
        gpu_max_jobs = 4
        enforce_vram = false
    """)
    assert (cfg.gpu_vram_mb, cfg.gpu_vram_reserve_mb) == (8188, 1024)
    assert cfg.gpu_max_jobs == 4
    assert cfg.enforce_vram is False


def test_gpu_max_jobs_must_be_at_least_one(tmp_path):
    with pytest.raises(ConfigError, match="gpu_max_jobs"):
        write_and_load(tmp_path, '[queue]\nroot = "/q"\ngpu_max_jobs = 0\n')


def test_reserve_may_not_swallow_the_whole_card(tmp_path):
    """A reserve at or above capacity admits nothing and would leave every
    GPU job pending forever, which is worse than refusing to start."""
    with pytest.raises(ConfigError, match="gpu_vram_reserve_mb"):
        write_and_load(tmp_path, """
            [queue]
            root = "/q"
            gpu_vram_mb = 1024
            gpu_vram_reserve_mb = 1024
        """)


def test_a_quoted_false_switches_the_watchdog_off_not_on(tmp_path):
    """`bool("false")` is True. Read that way, an operator who wrote the
    off switch gets a watchdog that kills jobs."""
    cfg = write_and_load(tmp_path, """
        [queue]
        root = "/q"
        enforce_vram = "false"
        kill_orphan_cuda = "off"
    """)
    assert cfg.enforce_vram is False
    assert cfg.kill_orphan_cuda is False


def test_quoted_true_spellings_still_mean_true(tmp_path):
    cfg = write_and_load(tmp_path, """
        [queue]
        root = "/q"
        enforce_vram = "yes"
        kill_orphan_cuda = "1"
    """)
    assert cfg.enforce_vram is True
    assert cfg.kill_orphan_cuda is True


def test_an_unreadable_bool_is_an_error_not_a_guess(tmp_path):
    """Loud at startup beats a job killed hours later by a setting its
    author believed said otherwise."""
    with pytest.raises(ConfigError, match=r"\[queue\].enforce_vram"):
        write_and_load(tmp_path, """
            [queue]
            root = "/q"
            enforce_vram = "sometimes"
        """)


def test_an_integer_bool_is_still_read_as_one(tmp_path):
    """`kill_orphan_cuda = 1` is valid TOML that the old `bool(...)` path
    read as True. Rejecting it now would refuse to start the daemon after
    an upgrade, with no change on the operator's side -- a stricter reader
    is worth an unreadable *string*, not a config that already worked."""
    cfg = write_and_load(tmp_path, """
        [queue]
        root = "/q"
        enforce_vram = 0
        kill_orphan_cuda = 1
    """)
    assert cfg.enforce_vram is False
    assert cfg.kill_orphan_cuda is True


def test_an_integer_that_is_not_a_bool_is_still_an_error(tmp_path):
    """Widening to int is for the two values that spell a bool. `2` is as
    unreadable as `"sometimes"`."""
    with pytest.raises(ConfigError, match=r"\[queue\].enforce_vram"):
        write_and_load(tmp_path, """
            [queue]
            root = "/q"
            enforce_vram = 2
        """)


# --- vram_policy: the capacity keys, for participants that are not the runner


def test_vram_policy_reads_the_capacity_keys(tmp_path):
    p = _write(tmp_path, textwrap.dedent("""
        [queue]
        root = "/q"
        gpu_vram_mb = 4096
        gpu_vram_reserve_mb = 256
    """))
    assert vram_policy(p) == (4096, 256)


def test_vram_policy_defaults_when_the_keys_are_absent(tmp_path):
    assert vram_policy(_write(tmp_path, '[queue]\nroot = "/q"\n')) == (
        None, DEFAULT_RESERVE_MB)


def test_vram_policy_does_not_need_a_loadable_config(tmp_path):
    """Deliberately not `load_config`. A caller that only wants the card's
    size must not be refused because `[queue].root` is missing or a project
    is half-declared -- it would fall back to a capacity the runner does not
    share, which is the divergence this exists to close."""
    p = _write(tmp_path, textwrap.dedent("""
        [queue]
        gpu_vram_mb = 4096

        [project.p]
        remote = "git@github.com:you/p.git"
    """))
    with pytest.raises(ConfigError):
        load_config(p)
    assert vram_policy(p) == (4096, DEFAULT_RESERVE_MB)


def test_vram_policy_falls_back_when_there_is_no_config(tmp_path):
    """The standalone `gpu-claim` path on a box with no runner deployed:
    exactly what it did before there was a config to read."""
    assert vram_policy(tmp_path / "absent.toml") == (None, DEFAULT_RESERVE_MB)
    (tmp_path / "junk.toml").write_text("this is not toml [[[")
    assert vram_policy(tmp_path / "junk.toml") == (None, DEFAULT_RESERVE_MB)


# --- max_holders: the latency cap, for participants that are not the runner

def test_max_holders_reads_the_key(tmp_path):
    p = _write(tmp_path, '[queue]\nroot = "/q"\ngpu_max_jobs = 4\n')
    assert max_holders(p) == 4


def test_max_holders_defaults_when_the_key_is_absent(tmp_path):
    assert max_holders(_write(tmp_path, '[queue]\nroot = "/q"\n')) == 2


def test_max_holders_does_not_need_a_loadable_config(tmp_path):
    """Same posture as vram_policy: a box with no runner deployed must
    still be able to run gpu-claim."""
    assert max_holders(tmp_path / "absent.toml") == 2


def test_max_holders_falls_back_on_a_nonsense_value(tmp_path):
    p = _write(tmp_path, '[queue]\nroot = "/q"\ngpu_max_jobs = 0\n')
    assert max_holders(p) == 2


# --- claim_dir_setting: what directory the deployed runner reads ---------
#
# `gpu-claim` runs in an interactive shell and the runner is a supervisor
# unit; the shell cannot see the unit's environment. The config file is
# the one thing both can read, so it is how a hand-run claim finds out it
# is writing somewhere the daemon is not looking (issue #19).

def test_claim_dir_setting_reads_the_queue_table(tmp_path):
    p = _write(tmp_path, '[queue]\nroot = "/q"\nclaim_dir = "/workspace/lock/gpu"\n')
    assert claim_dir_setting(p) == Path("/workspace/lock/gpu")


def test_claim_dir_setting_is_none_when_the_key_is_absent(tmp_path):
    """Not a divergence. The daemon then reads its own `$GPU_CLAIM_DIR`,
    which this process cannot see, so there is nothing to compare against
    and guessing would warn every user on a correctly configured box."""
    assert claim_dir_setting(_write(tmp_path, '[queue]\nroot = "/q"\n')) is None


def test_claim_dir_setting_is_none_when_there_is_no_config(tmp_path):
    assert claim_dir_setting(tmp_path / "absent.toml") is None


def test_claim_dir_setting_does_not_need_a_loadable_config(tmp_path):
    """The same posture `vram_policy` and `max_holders` take: a caller that
    only wants one key must not be refused because `[queue].root` is
    missing or a project is half-declared."""
    p = _write(tmp_path, '[queue]\nclaim_dir = "/workspace/lock/gpu"\n'
                         '[project.p]\nremote = "x"\n')
    with pytest.raises(ConfigError):
        load_config(p)
    assert claim_dir_setting(p) == Path("/workspace/lock/gpu")


def test_claim_dir_setting_is_none_for_a_malformed_config(tmp_path):
    assert claim_dir_setting(_write(tmp_path, "[queue\nroot =")) is None


def test_claim_dir_setting_ignores_an_empty_value(tmp_path):
    """`claim_dir = ""` is not a directory; `load_config` reads it as
    unset (`Path(claim_dir) if claim_dir else None`), and a reader that
    disagreed would warn about a divergence the runner does not have."""
    p = _write(tmp_path, '[queue]\nroot = "/q"\nclaim_dir = ""\n')
    assert claim_dir_setting(p) is None


def test_claim_dir_setting_defaults_to_the_deployed_config(tmp_path,
                                                           monkeypatch):
    p = _write(tmp_path, '[queue]\nroot = "/q"\nclaim_dir = "/somewhere"\n')
    monkeypatch.setenv("GPUQ_CONFIG", str(p))
    assert claim_dir_setting() == Path("/somewhere")
