import pytest
from gpuqueue import gpuid
from gpuqueue.gpuid import normalize_gpu_uuid, gpu_key, lock_filename, GpuIdError

HEX = "4b8f2c1a-0000-0000-0000-000000000001"

def test_smi_and_torch_spellings_collapse_to_one_key():
    assert normalize_gpu_uuid(f"GPU-{HEX}") == normalize_gpu_uuid(HEX)

def test_normalize_is_lowercase():
    assert normalize_gpu_uuid(f"GPU-{HEX.upper()}") == HEX

def test_normalize_strips_mig_prefix_and_whitespace():
    assert normalize_gpu_uuid(f"  MIG-{HEX}\n") == HEX

def test_normalize_rejects_empty():
    with pytest.raises(GpuIdError):
        normalize_gpu_uuid("   ")

def test_gpu_key_normalizes_the_smi_spelling(monkeypatch):
    monkeypatch.setattr(gpuid, "gpu_uuid_from_nvidia_smi", lambda i=0: f"GPU-{HEX}")
    assert gpu_key() == HEX

def test_gpu_key_falls_back_to_name_index(monkeypatch):
    monkeypatch.setattr(gpuid, "gpu_uuid_from_nvidia_smi", lambda i=0: None)
    monkeypatch.setattr(gpuid, "gpu_name_index_fallback",
                        lambda i=0: "NVIDIA GeForce RTX 4060-0")
    assert gpu_key() == "nvidia geforce rtx 4060-0"

def test_gpu_key_raises_when_no_source_works(monkeypatch):
    monkeypatch.setattr(gpuid, "gpu_uuid_from_nvidia_smi", lambda i=0: None)
    monkeypatch.setattr(gpuid, "gpu_name_index_fallback", lambda i=0: None)
    with pytest.raises(GpuIdError):
        gpu_key()

def test_module_never_imports_torch():
    """A JAX project must be able to use this lock, and the runner daemon
    must not initialize a CUDA context just to read an identifier."""
    import inspect
    assert "import torch" not in inspect.getsource(gpuid)

def test_lock_filename_is_a_safe_basename():
    name = lock_filename(HEX)
    assert "/" not in name and name.endswith(".lock")

def test_lock_filename_sanitizes_fallback_keys():
    assert "/" not in lock_filename("NVIDIA GeForce RTX 4060/weird-0")

def test_nvidia_smi_parses_first_line(monkeypatch):
    monkeypatch.setattr(gpuid, "_run",
                        lambda argv: f"GPU-{HEX}\nGPU-other\n")
    assert gpuid.gpu_uuid_from_nvidia_smi(0) == f"GPU-{HEX}"

def test_nvidia_smi_returns_none_when_missing(monkeypatch):
    def boom(argv):
        raise FileNotFoundError("nvidia-smi")
    monkeypatch.setattr(gpuid, "_run", boom)
    assert gpuid.gpu_uuid_from_nvidia_smi(0) is None


# --- pinning a job to the claimed card ---------------------------------------
# `gpu_key` is deliberately normalized (lowercased, prefix stripped) because it
# names a lock file. CUDA_VISIBLE_DEVICES is a different consumer: the driver
# matches the uuid as the driver spells it, so the pin value must not be put
# through that normalization.

def test_cuda_visible_value_keeps_the_driver_spelling(monkeypatch):
    monkeypatch.setattr(gpuid, "gpu_uuid_from_nvidia_smi",
                        lambda i=0: f"GPU-{HEX.upper()}")
    assert gpuid.cuda_visible_value() == f"GPU-{HEX.upper()}"


def test_cuda_visible_value_is_none_without_a_uuid(monkeypatch):
    # The name-index fallback is a usable *lock key* but not a thing the CUDA
    # driver can resolve, so pinning must decline rather than emit garbage.
    monkeypatch.setattr(gpuid, "gpu_uuid_from_nvidia_smi", lambda i=0: None)
    monkeypatch.setattr(gpuid, "gpu_name_index_fallback",
                        lambda i=0: "NVIDIA GeForce RTX 4060-0")
    assert gpuid.cuda_visible_value() is None


def test_cuda_visible_value_is_none_when_smi_is_missing(monkeypatch):
    def boom(argv):
        raise FileNotFoundError("nvidia-smi")
    monkeypatch.setattr(gpuid, "_run", boom)
    assert gpuid.cuda_visible_value() is None
