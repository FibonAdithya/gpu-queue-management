import textwrap

import pytest
from gpuqueue.config import load_config, ConfigError

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
