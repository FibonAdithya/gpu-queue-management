import os
import pytest
from gpuqueue import preflight as pf
from gpuqueue.preflight import PreflightFailed, foreign_processes, compute_apps

SMI_ROWS = "1234, 512 MiB, python\n5678, [N/A], jupyter\n"

def test_compute_apps_parses_rows(monkeypatch):
    monkeypatch.setattr(pf, "_run", lambda argv: SMI_ROWS)
    apps = compute_apps()
    assert apps[0] == {"pid": 1234, "used_mb": 512, "name": "python"}
    assert apps[1]["used_mb"] is None

def test_compute_apps_empty_output_means_none_running(monkeypatch):
    monkeypatch.setattr(pf, "_run", lambda argv: "\n")
    assert compute_apps() == []

def test_compute_apps_not_supported_means_cannot_see(monkeypatch):
    monkeypatch.setattr(pf, "_run", lambda argv: "[Not Supported]\n")
    assert compute_apps() is None

def test_compute_apps_missing_smi_means_cannot_see(monkeypatch):
    def boom(argv):
        raise FileNotFoundError()
    monkeypatch.setattr(pf, "_run", boom)
    assert compute_apps() is None

def test_foreign_excludes_allowed_pids(monkeypatch):
    monkeypatch.setattr(pf, "compute_apps",
                        lambda: [{"pid": 1234, "used_mb": 1, "name": "python"}])
    assert foreign_processes(allow={1234}) == []

def test_foreign_excludes_own_pids(monkeypatch):
    monkeypatch.setattr(pf, "compute_apps",
                        lambda: [{"pid": os.getpid(), "used_mb": 1, "name": "py"}])
    assert foreign_processes() == []

def test_foreign_reports_stranger(monkeypatch):
    monkeypatch.setattr(pf, "compute_apps",
                        lambda: [{"pid": 4321, "used_mb": 900, "name": "train.py"}])
    assert [p["pid"] for p in foreign_processes()] == [4321]

def test_preflight_raises_naming_pid_and_command(monkeypatch):
    monkeypatch.setattr(pf, "compute_apps",
                        lambda: [{"pid": 4321, "used_mb": 900, "name": "train.py"}])
    with pytest.raises(PreflightFailed) as e:
        pf.preflight()
    assert "4321" in str(e.value) and "train.py" in str(e.value)

def test_preflight_queries_the_card_once(monkeypatch):
    """Two queries let the visibility check and the contention check
    disagree about what they saw."""
    calls = []
    def counted():
        calls.append(1)
        return []
    monkeypatch.setattr(pf, "compute_apps", counted)
    pf.preflight()
    assert len(calls) == 1

def test_preflight_passes_when_cannot_see(monkeypatch, capsys):
    """Unprivileged containers often cannot enumerate compute apps. Warn,
    do not block — refusing to run on every box that hides the list makes
    the tool useless exactly where it is needed."""
    monkeypatch.setattr(pf, "compute_apps", lambda: None)
    pf.preflight()
    assert "cannot enumerate" in capsys.readouterr().err

def test_preflight_passes_when_clear(monkeypatch):
    monkeypatch.setattr(pf, "compute_apps", lambda: [])
    pf.preflight()

from gpuqueue import ledger as lg

KEY = "4b8f2c1a-0000-0000-0000-000000000001"


def _record(tmp_path, pid, usage_pid, vram_mb=512):
    rec = lg.Record(path=lg.ledger_dir(KEY, tmp_path) / f"{pid}.aaa.json",
                    pid=pid, usage_pid=usage_pid, vram_mb=vram_mb,
                    owner="co-tenant", cmd=["python", "t.py"],
                    started_at="2026-08-10T00:00:00Z", key=KEY)
    lg.write_record(rec)
    return rec


def test_a_ledgered_co_tenant_is_not_contention(tmp_path, monkeypatch):
    """Sharing is the point. A declared holder's process must not read as
    an intruder, or no second job can ever start."""
    _record(tmp_path, os.getpid(), os.getpid())
    monkeypatch.setattr(pf, "compute_apps",
                        lambda: [{"pid": 5150, "used_mb": 400, "name": "co.py"}])
    # descendants(os.getpid()) must stay empty so this process's own
    # exempt set can't accidentally cover 5150 and let the test pass for
    # the wrong reason.
    monkeypatch.setattr(pf, "descendants", lambda pid: set())
    # ledger.attribute() walks its *own* `descendants` (imported directly
    # in ledger.py), never preflight's. Guarding on the record's
    # usage_pid is what actually puts 5150 in the co-tenant's tree, so
    # ledger attribution -- not the own-process exemption -- is what
    # preflight relies on here.
    monkeypatch.setattr(lg, "descendants",
                        lambda pid: {5150} if pid == os.getpid() else set())
    pf.preflight(directory=tmp_path)


def test_an_unledgered_process_still_refuses(tmp_path, monkeypatch):
    monkeypatch.setattr(pf, "compute_apps",
                        lambda: [{"pid": 4321, "used_mb": 900, "name": "train.py"}])
    monkeypatch.setattr(pf, "descendants", lambda pid: set())
    with pytest.raises(PreflightFailed) as e:
        pf.preflight(directory=tmp_path)
    assert "4321" in str(e.value) and "train.py" in str(e.value)


def test_a_dead_holders_record_does_not_shelter_anyone(tmp_path, monkeypatch):
    _record(tmp_path, 4000000, 4000000)
    monkeypatch.setattr(pf, "compute_apps",
                        lambda: [{"pid": 4321, "used_mb": 900, "name": "t.py"}])
    # descendants(os.getpid()) must stay empty so this process's own
    # exempt set can't accidentally cover 4321 and let the test pass for
    # the wrong reason.
    monkeypatch.setattr(pf, "descendants", lambda pid: set())
    # ledger.attribute() walks its *own* `descendants` (imported directly
    # in ledger.py), never preflight's -- patching pf.descendants cannot
    # reach it. This is what actually puts 4321 in the dead holder's
    # tree, so liveness filtering has something real to prove: without
    # it, the dead record would legitimately own 4321 and preflight
    # would stay silent.
    monkeypatch.setattr(lg, "descendants",
                        lambda pid: {4321} if pid == 4000000 else set())
    with pytest.raises(PreflightFailed):
        pf.preflight(directory=tmp_path)
