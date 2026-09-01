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


# --- The union stops at own_pids (issue #19) -----------------------------
#
# `own_pids` reads every directory a claim could be in, because it is the
# last exemption before a SIGKILL and over-exempting is the safe way to be
# wrong there. These two are the opposite trade: they decide whether a run
# is allowed to *start*, where over-exempting fails open and puts a second
# trainer onto a card someone else is already using. A record under
# `DEFAULT_CLAIM_DIR` must not vouch for a process on a card this
# directory knows nothing about.
#
# Asserted on which ledger is consulted, because that is precisely the
# quantity issue #19 is about, and because the alternative -- a live,
# non-descendant pid whose tree `ledger.attribute` would charge the
# process to -- needs a detached helper for what is a one-line invariant.

from pathlib import Path as _Path
from gpuqueue import ledger as _lg


def _ledgers_read(monkeypatch):
    """Every directory handed to `ledger.all_records`, in order. Wraps the
    real function rather than replacing it, so the call still does its
    work and the assertion is about a real argument."""
    seen = []
    real = _lg.all_records

    def spy(directory):
        seen.append(_Path(directory))
        return real(directory)

    monkeypatch.setattr(pf.ledger, "all_records", spy)
    return seen


def test_preflight_consults_one_ledger(tmp_path, monkeypatch):
    ours = tmp_path / "ours"
    ours.mkdir()
    monkeypatch.setenv("GPU_CLAIM_DIR", str(ours))
    monkeypatch.setattr(pf, "compute_apps", lambda: [])
    seen = _ledgers_read(monkeypatch)

    pf.preflight()

    assert seen == [ours], (
        "preflight decides whether a run may start; reading the other "
        "claim directory as well would let a record there vouch for a "
        "process on a card this one knows nothing about")


def test_unledgered_processes_consults_one_ledger(tmp_path, monkeypatch):
    ours = tmp_path / "ours"
    ours.mkdir()
    monkeypatch.setenv("GPU_CLAIM_DIR", str(ours))
    monkeypatch.setattr(pf, "compute_apps",
                        lambda: [{"pid": 999_002, "used_mb": 9, "name": "t"}])
    seen = _ledgers_read(monkeypatch)

    assert [a["pid"] for a in pf.unledgered_processes()] == [999_002]
    assert seen == [ours]


def test_preflight_does_not_refuse_a_process_in_the_prospective_scope(
        tmp_path, monkeypatch):
    # preflight runs BEFORE the claim exists, so there is no record to
    # attribute the container's in-flight CUDA to. Without the
    # prospective scope, claiming a busy container is refused and
    # claiming an idle one races the next request -- the feature fails
    # exactly when it is needed.
    scope = "/system.slice/docker-43faa0ee.scope"
    monkeypatch.setattr(
        pf, "compute_apps",
        lambda: [{"pid": 2791919, "used_mb": 900, "name": "tig-runtime"}])
    monkeypatch.setattr(pf.cgroups, "in_scope",
                        lambda pid, s, proc_root="/proc": s == scope)
    with pytest.raises(pf.PreflightFailed):
        pf.preflight(directory=tmp_path)
    pf.preflight(directory=tmp_path, scope=scope)  # must not raise
