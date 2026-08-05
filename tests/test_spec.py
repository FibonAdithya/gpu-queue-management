import pytest
from gpuqueue.spec import JobSpec, SpecError, utcnow_iso

def _minimal(**over):
    d = {
        "id": "myproject-v0-train-01",
        "lane": "gpu",
        "project": "myproject",
        "commit": "a1b2c3d",
        "branch": "experiment/v0",
        "cmd": ["python", "-m", "src.train", "--config", "c.yaml"],
        "artifacts": ["runs/v0/summary.json"],
        "timeout_s": 21600,
        "attempts": 0,
        "dedupe_key": "myproject:v0:a1b2c3d",
    }
    d.update(over)
    return d

def test_round_trip_preserves_fields():
    spec = JobSpec.from_dict(_minimal())
    assert JobSpec.from_dict(spec.to_dict()) == spec

def test_defaults_are_filled():
    d = _minimal()
    del d["attempts"]
    spec = JobSpec.from_dict(d)
    assert spec.attempts == 0
    assert spec.pid is None
    assert spec.submitted_at.endswith("Z")

def test_unknown_lane_rejected():
    with pytest.raises(SpecError, match="lane"):
        JobSpec.from_dict(_minimal(lane="tpu")).validate()

def test_empty_cmd_rejected():
    with pytest.raises(SpecError, match="cmd"):
        JobSpec.from_dict(_minimal(cmd=[])).validate()

def test_nonpositive_timeout_rejected():
    with pytest.raises(SpecError, match="timeout_s"):
        JobSpec.from_dict(_minimal(timeout_s=0)).validate()

def test_id_with_path_separator_rejected():
    with pytest.raises(SpecError, match="id"):
        JobSpec.from_dict(_minimal(id="../escape")).validate()

def test_absolute_artifact_path_rejected():
    with pytest.raises(SpecError, match="artifact"):
        JobSpec.from_dict(_minimal(artifacts=["/etc/passwd"])).validate()

def test_utcnow_iso_format():
    assert utcnow_iso().endswith("Z")
    assert "T" in utcnow_iso()
