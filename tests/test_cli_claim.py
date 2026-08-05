import json
import pytest
from gpuqueue import cli_claim
from gpuqueue.claim import ClaimBusy
from gpuqueue.preflight import PreflightFailed
from gpuqueue.gpuid import GpuIdError

KEY = "4b8f2c1a-0000-0000-0000-000000000001"

@pytest.fixture(autouse=True)
def fake_gpu(tmp_path, monkeypatch):
    monkeypatch.setenv("GPU_CLAIM_DIR", str(tmp_path))
    monkeypatch.setattr(cli_claim, "gpu_key", lambda index=0: KEY)
    monkeypatch.setattr(cli_claim, "preflight", lambda allow=None: None)

def test_runs_command_and_returns_its_exit_code():
    assert cli_claim.main(["--", "sh", "-c", "exit 0"]) == 0
    assert cli_claim.main(["--", "sh", "-c", "exit 3"]) == 3

def test_claim_released_after_command(tmp_path):
    cli_claim.main(["--", "true"])
    assert list(tmp_path.glob("*.lock.json")) == []

def test_busy_exits_75(monkeypatch, capsys):
    def busy(**kw):
        raise ClaimBusy("held by pid 999")
    monkeypatch.setattr(cli_claim, "gpu_claim", busy)
    assert cli_claim.main(["--", "true"]) == 75
    assert "999" in capsys.readouterr().err

def test_preflight_failure_exits_69(monkeypatch, capsys):
    def fail(allow=None):
        raise PreflightFailed("pid 4321 train.py")
    monkeypatch.setattr(cli_claim, "preflight", fail)
    assert cli_claim.main(["--", "true"]) == 69
    assert "4321" in capsys.readouterr().err

def test_no_preflight_flag_skips_it(monkeypatch):
    def fail(allow=None):
        raise PreflightFailed("should not be called")
    monkeypatch.setattr(cli_claim, "preflight", fail)
    assert cli_claim.main(["--no-preflight", "--", "true"]) == 0

def test_no_gpu_exits_69(monkeypatch, capsys):
    def boom(index=0):
        raise GpuIdError("no CUDA device")
    monkeypatch.setattr(cli_claim, "gpu_key", boom)
    assert cli_claim.main(["--", "true"]) == 69
    assert "no CUDA device" in capsys.readouterr().err

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
