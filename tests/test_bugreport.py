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
