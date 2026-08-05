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
