"""The runner entry point, and the one thing it has to say at startup."""
import logging

import pytest

from gpuqueue import cli_runner


class _StubRunner:
    """`main` only needs something with tick/stop; the Runner itself has
    its own tests and would want a real queue root and a card."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.ticks = 0

    def tick(self):
        self.ticks += 1

    def stop(self):
        pass


@pytest.fixture(autouse=True)
def stub_runner(monkeypatch):
    monkeypatch.setattr(cli_runner, "Runner", _StubRunner)


def _config(tmp_path, name="gpuq.toml"):
    p = tmp_path / name
    p.write_text(f'[queue]\nroot = "{tmp_path / "q"}"\n')
    return p


def test_warns_when_gpu_claim_will_read_a_different_config(
        tmp_path, monkeypatch, caplog):
    """`--config` moves the runner's capacity keys without moving
    `gpu-claim`'s. The two share one `<key>.lock.d`, so they then admit
    against different totals into the same ledger -- the double-booking
    `vram_policy` exists to close, reopened by a flag."""
    monkeypatch.delenv("GPUQ_CONFIG", raising=False)
    p = _config(tmp_path, "alt.toml")
    with caplog.at_level(logging.WARNING):
        assert cli_runner.main(["--config", str(p), "--once"]) == 0
    assert "GPUQ_CONFIG" in caplog.text
    assert str(p) in caplog.text


def test_no_warning_when_the_environment_already_agrees(
        tmp_path, monkeypatch, caplog):
    p = _config(tmp_path)
    monkeypatch.setenv("GPUQ_CONFIG", str(p))
    with caplog.at_level(logging.WARNING):
        assert cli_runner.main(["--config", str(p), "--once"]) == 0
    assert "GPUQ_CONFIG" not in caplog.text


def test_no_warning_without_the_flag(tmp_path, monkeypatch, caplog):
    p = _config(tmp_path)
    monkeypatch.setenv("GPUQ_CONFIG", str(p))
    with caplog.at_level(logging.WARNING):
        assert cli_runner.main(["--once"]) == 0
    assert "GPUQ_CONFIG" not in caplog.text


def test_a_bad_config_still_exits_two(tmp_path, monkeypatch):
    monkeypatch.delenv("GPUQ_CONFIG", raising=False)
    p = tmp_path / "bad.toml"
    p.write_text("[queue]\n")          # no root
    assert cli_runner.main(["--config", str(p), "--once"]) == 2
