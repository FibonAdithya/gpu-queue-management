import os
import subprocess
import sys
import time

from gpuqueue.procs import pid_alive, descendants


def test_pid_alive_true_for_self():
    assert pid_alive(os.getpid()) is True


def test_pid_alive_false_for_impossible_pid():
    assert pid_alive(4000000) is False


def test_pid_alive_false_for_zero_and_negative():
    assert pid_alive(0) is False
    assert pid_alive(-1) is False


def test_descendants_finds_a_grandchild():
    """One ps call per node, so a grandchild only shows up if the walk
    actually recurses."""
    child = subprocess.Popen(
        [sys.executable, "-c",
         "import subprocess,sys,time;"
         "subprocess.Popen(['sleep','30']);print('up',flush=True);"
         "time.sleep(30)"],
        stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "up"
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            kids = descendants(os.getpid())
            if len(kids) >= 2:
                break
            time.sleep(0.05)
        assert child.pid in kids
        assert len(kids) >= 2, "did not recurse past the direct child"
    finally:
        child.kill()
        child.wait()


def test_descendants_of_a_leaf_is_empty():
    assert descendants(4000000) == set()
