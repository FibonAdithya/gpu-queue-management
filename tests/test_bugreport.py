import errno
import pytest
from gpuqueue.bugreport import CallerError, is_gpuq_fault, PHASES
from gpuqueue.executor import StartFailed
from gpuqueue.git_ops import GitError


def _raised(exc):
    """Give the exception a real traceback, as classify sees it in anger."""
    try:
        raise exc
    except type(exc) as e:
        return e


def test_a_git_failure_is_gpuq_fault():
    assert is_gpuq_fault(_raised(GitError("git worktree add failed"))) is True


def test_an_unknown_exception_is_gpuq_fault():
    """Default to ours. A novel exception out of gpuq's stack is exactly the
    case this system exists to surface."""
    assert is_gpuq_fault(_raised(ValueError("something new"))) is True


def test_a_declared_artifact_never_produced_is_caller_fault():
    exc = _raised(CallerError("declared artifact not produced: runs/s.json"))
    assert is_gpuq_fault(exc) is False


def test_start_failed_on_a_missing_binary_is_caller_fault():
    exc = StartFailed("cannot execute 'nope': No such file or directory")
    exc.errno = errno.ENOENT
    assert is_gpuq_fault(_raised(exc)) is False


def test_start_failed_on_a_non_executable_file_is_caller_fault():
    exc = StartFailed("cannot execute './train.sh': Permission denied")
    exc.errno = errno.EACCES
    assert is_gpuq_fault(_raised(exc)) is False


def test_start_failed_for_any_other_reason_is_gpuq_fault():
    """StartFailed is genuinely ambiguous. Only 'the thing you asked to run
    does not exist' is the caller's; a bad cwd is ours."""
    exc = StartFailed("cannot execute 'python': Not a directory")
    exc.errno = errno.ENOTDIR
    assert is_gpuq_fault(_raised(exc)) is True


def test_start_failed_with_no_errno_is_gpuq_fault():
    assert is_gpuq_fault(_raised(StartFailed("cannot execute 'x'"))) is True


def test_the_designed_phases_exist():
    assert PHASES == ("preflight", "checkout", "execute", "artifacts",
                      "reap", "admit")


from pathlib import Path
from gpuqueue.bugreport import signature
from gpuqueue import git_ops


def _git_failure():
    """A real traceback with real gpuqueue frames in it."""
    try:
        git_ops.git(["rev-parse", "--verify", "nope"], cwd=Path("/"))
    except Exception as e:
        return e


def test_signature_is_short_hex():
    sig = signature(_git_failure(), "checkout")
    assert len(sig) == 12 and all(c in "0123456789abcdef" for c in sig)


def test_the_same_failure_signs_the_same():
    assert signature(_git_failure(), "checkout") == \
           signature(_git_failure(), "checkout")


def test_the_phase_changes_the_signature():
    exc = _git_failure()
    assert signature(exc, "checkout") != signature(exc, "artifacts")


def test_the_exception_type_changes_the_signature():
    a = _raised(ValueError("x"))
    b = _raised(TypeError("x"))
    assert signature(a, "reap") != signature(b, "reap")


def test_the_signature_is_frame_names_not_line_numbers():
    """Pinned exactly, because this is the property the whole dedup rests
    on: an unrelated edit above the raise must not refile an open bug.

    `git_ops.git` is the only gpuqueue frame in this traceback -- the test's
    own frame is outside the package and dropped -- so the payload is
    phase, exception type and that one name.
    """
    import hashlib
    expected = hashlib.sha256(b"checkout|GitError|git").hexdigest()[:12]
    assert signature(_git_failure(), "checkout") == expected


def test_frames_outside_gpuqueue_are_ignored():
    """A caller's stack must not be part of gpuq's fingerprint."""
    def not_gpuqueue():
        raise RuntimeError("boom")

    try:
        not_gpuqueue()
    except RuntimeError as e:
        outside = e
    try:
        raise RuntimeError("boom")
    except RuntimeError as e:
        bare = e
    assert signature(outside, "reap") == signature(bare, "reap")


def test_an_unknown_phase_is_rejected():
    with pytest.raises(ValueError, match="phase"):
        signature(_raised(RuntimeError("x")), "nonsense")
