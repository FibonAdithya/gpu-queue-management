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
