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
    monkeypatch.setattr(cli_claim, "preflight", lambda allow=None, directory=None: None)

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
    monkeypatch.setattr(cli_claim, "preflight", lambda: None)
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
    monkeypatch.setattr(cli_claim, "preflight", lambda: None)
    cli_claim.main(["--", "true"])
    assert seen["vram_mb"] is None

def test_preflight_failure_exits_69(monkeypatch, capsys):
    def fail(allow=None, directory=None):
        raise PreflightFailed("pid 4321 train.py")
    monkeypatch.setattr(cli_claim, "preflight", fail)
    assert cli_claim.main(["--", "true"]) == 69
    assert "4321" in capsys.readouterr().err

def test_no_preflight_flag_skips_it(monkeypatch):
    def fail(allow=None, directory=None):
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
    monkeypatch.setattr("gpuqueue.claim.total_vram_mb", lambda: 8188)
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
