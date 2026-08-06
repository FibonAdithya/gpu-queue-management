import errno
import hashlib
from pathlib import Path

import pytest
from gpuqueue.bugreport import (CallerError, is_gpuq_fault, PHASES,
                                signature, build_report, issue_title,
                                issue_body, bump_body)
from gpuqueue.executor import StartFailed
from gpuqueue.git_ops import GitError
from gpuqueue import git_ops
from gpuqueue.spec import JobSpec


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


def _spec():
    return JobSpec(id="j1", lane="gpu", project="p", commit="deadbeef",
                   branch="main", cmd=["python", "train.py"])


def _report():
    return build_report(_git_failure(), "checkout", spec=_spec(),
                        queue_counts={"pending": 3, "running": 1,
                                      "done": 40, "failed": 2},
                        gpuq_commit="96f0b57",
                        occurred_at="2026-08-05T12:00:00Z")


def test_the_title_names_the_phase_and_the_exception():
    title = issue_title(_report())
    assert "checkout" in title and "GitError" in title


def test_the_title_carries_the_signature_so_a_human_can_match_it():
    r = _report()
    assert r.sig in issue_title(r)


def test_the_body_carries_the_signature_line_verbatim():
    """bugfiler greps for this exact string; gh search alone is fuzzy."""
    r = _report()
    assert f"sig: {r.sig}" in issue_body(r)


def test_the_body_carries_the_traceback():
    assert "Traceback (most recent call last)" in issue_body(_report())


def test_the_body_carries_the_jobspec_as_json():
    body = issue_body(_report())
    assert '"id": "j1"' in body and '"project": "p"' in body


def test_the_body_carries_queue_state_and_the_gpuq_commit():
    body = issue_body(_report())
    assert "pending 3" in body and "96f0b57" in body


def test_the_body_starts_at_one_occurrence():
    body = issue_body(_report())
    assert "occurrences: 1" in body
    assert "first seen: 2026-08-05T12:00:00Z" in body
    assert "last seen: 2026-08-05T12:00:00Z" in body


def test_the_body_says_no_agent_wrote_it():
    """The auto path carries no prose from any agent, and the reader of the
    issue -- human or model -- must be able to tell."""
    assert "no agent" in issue_body(_report()).lower()


def test_a_report_with_no_job_still_renders():
    """Reaper and admit failures have no one job to blame."""
    r = build_report(_git_failure(), "reap", queue_counts={},
                     occurred_at="2026-08-05T12:00:00Z")
    assert "no job" in issue_body(r).lower()


def test_bump_increments_the_count_and_moves_last_seen():
    body = issue_body(_report())
    bumped = bump_body(body, "2026-08-06T09:00:00Z")
    assert "occurrences: 2" in bumped
    assert "first seen: 2026-08-05T12:00:00Z" in bumped
    assert "last seen: 2026-08-06T09:00:00Z" in bumped


def test_bump_is_repeatable():
    body = issue_body(_report())
    for _ in range(3):
        body = bump_body(body, "2026-08-06T09:00:00Z")
    assert "occurrences: 4" in body


def test_bump_leaves_a_body_it_does_not_recognise_alone():
    """An owner may have rewritten the body by hand. Never mangle it."""
    assert bump_body("hand written", "2026-08-06T09:00:00Z") == "hand written"


def test_every_module_in_the_package_is_recognised_as_a_gpuqueue_frame():
    """`_is_gpuqueue_frame` recognises a package frame by checking that
    `Path(filename).parent.name == "gpuqueue"` -- true today because the
    package is flat, but silently false for any file a future subpackage
    puts one directory deeper. That failure mode is quiet: frames just drop
    out of the signature and the fingerprint gets weaker, not an error
    anyone would notice. This test is cheap insurance, not a guarantee the
    check is future-proof -- it exists so the day someone adds a
    subpackage, the suite says so instead of the fingerprint degrading
    unnoticed."""
    import gpuqueue
    from gpuqueue.bugreport import _is_gpuqueue_frame

    pkg_dir = Path(gpuqueue.__file__).parent
    modules = list(pkg_dir.glob("**/*.py"))
    assert modules, "sanity check: the glob must actually find the package"
    for module in modules:
        assert _is_gpuqueue_frame(str(module)), (
            f"{module} is not recognised as a gpuqueue frame -- "
            "_is_gpuqueue_frame needs updating for the package layout")
