"""Runs the suite's shell tests, so that they run at all.

`tests/test_bootstrap.sh` was written as a real test and then collected by
nothing: pytest gathers `test_*.py`, and `ci.yml` runs pytest and nothing
else. Seventy-six assertions had never executed in CI. A test that cannot
fail is worse than no test, because the coverage is counted and the
regression still ships -- so the fix belongs here rather than in a README
line telling people to run it by hand.

Each `tests/test_*.sh` becomes one pytest case. The shell script owns its
own assertions and reports them; this only asserts the exit status and
surfaces the output when it is non-zero.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).parent

# The environment the shell tests must NOT inherit.
#
# `conftest.py` sets GPUQ_CONFIG and GPU_CLAIM_DIR on every test to keep the
# deployed `/workspace` out of the Python suite. Those same variables are
# *inputs* to the scripts under test here: `bootstrap.sh` reads
# `GPU_CLAIM_DIR` as the default claim directory, so inheriting the
# fixture's value sends the claim dir somewhere the script's own
# `--prefix` never reaches, and `test_bootstrap.sh`'s "creates the claim
# dir" check fails against a correct bootstrap.sh.
#
# Unset rather than overridden: the scripts derive their own defaults, and
# any value chosen here would be this harness deciding something the
# script is supposed to decide.
_SCRUBBED = ("GPUQ_PREFIX", "QUEUE_ROOT", "GPU_CLAIM_DIR", "GPUQ_CONFIG",
             "GPUQ_SKILLS_DIR", "SUPERVISOR_CONF_DIR", "PYTHON")


def _shell_tests() -> list[Path]:
    return sorted(TESTS_DIR.glob("test_*.sh"))


def test_shell_tests_are_discovered():
    """The harness is pointed at something.

    Without this, deleting every `.sh` -- or renaming the directory this
    globs -- turns the parametrised case below into zero cases, which
    pytest reports as success. That is the exact failure this file exists
    to fix, so it must not be reintroducible here.
    """
    found = _shell_tests()
    assert found, f"no test_*.sh found in {TESTS_DIR}"


@pytest.mark.parametrize("script", _shell_tests(), ids=lambda p: p.name)
def test_shell_script_passes(script: Path):
    env = {k: v for k, v in os.environ.items() if k not in _SCRUBBED}
    proc = subprocess.run(
        ["bash", str(script)],
        capture_output=True, text=True, env=env, timeout=600,
        cwd=str(TESTS_DIR.parent),
    )
    if proc.returncode != 0:
        pytest.fail(
            f"{script.name} exited {proc.returncode}\n"
            f"--- stdout ---\n{proc.stdout}\n"
            f"--- stderr ---\n{proc.stderr}"
        )
