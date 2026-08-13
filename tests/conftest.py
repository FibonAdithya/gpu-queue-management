"""Suite-wide test environment.

`CUDA_VISIBLE_DEVICES` is cleared for every test because the box this
project runs on is one where people export it -- that is the whole premise
of the pinning feature. Inherited into a test, it silently changes what is
under test: `_child_env` returns early on the caller-wins path instead of
pinning, and a child that should be unpinned inherits a value rather than
seeing nothing. Both read as a product bug when they are only a dirty
environment.

A test that cares about an inherited value sets it explicitly; this fixture
runs first, so `monkeypatch.setenv` in a test body still wins.
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_inherited_cuda_visible_devices(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)


@pytest.fixture(autouse=True)
def _no_deployed_config(tmp_path, monkeypatch):
    """`claim.default_usable_mb` reads the runner's config so that the two
    participants in one ledger size the card the same way. That makes the
    deployed `/workspace/gpuq.toml` -- present on exactly the box this is
    developed on -- an input to the suite, where a declared `gpu_vram_mb`
    would silently change what capacity tests admit against.

    A test that wants a config writes one and points GPUQ_CONFIG at it.
    """
    monkeypatch.setenv("GPUQ_CONFIG", str(tmp_path / "no-such-gpuq.toml"))


@pytest.fixture(autouse=True)
def _no_deployed_claims(tmp_path, monkeypatch):
    """The same hazard as the config above, one directory over.

    A call that omits `directory=` falls back to `claim_dir()`, which
    reads GPU_CLAIM_DIR and otherwise lands on the live `/var/lock/gpu`.
    The preflight tests take that path, so on the box this is developed on
    a real record whose process tree happens to cover a hard-coded test pid
    silently inverts the assertion.

    A test that wants records writes them under its own tmp_path and
    passes `directory=`, or sets this variable itself.
    """
    monkeypatch.setenv("GPU_CLAIM_DIR", str(tmp_path / "claims-isolated"))
