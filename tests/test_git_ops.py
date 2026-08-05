import subprocess
import pytest
from pathlib import Path
from gpuqueue.git_ops import (git, ensure_checkout, add_worktree,
                              remove_worktree, commit_artifacts, GitError)
from gpuqueue.config import ProjectConfig

def _init_origin(tmp_path) -> Path:
    origin = tmp_path / "origin"
    origin.mkdir()
    git(["init", "-q", "-b", "main"], cwd=origin)
    git(["config", "user.email", "t@t"], cwd=origin)
    git(["config", "user.name", "t"], cwd=origin)
    (origin / "a.txt").write_text("one\n")
    git(["add", "a.txt"], cwd=origin)
    git(["commit", "-qm", "first"], cwd=origin)
    return origin

@pytest.fixture
def project(tmp_path):
    origin = _init_origin(tmp_path)
    return ProjectConfig(name="p", remote=str(origin),
                         checkout=tmp_path / "checkout", commit_artifacts=True)

def test_ensure_checkout_clones(project):
    path = ensure_checkout(project)
    assert (path / "a.txt").exists()

def test_ensure_checkout_is_idempotent(project):
    ensure_checkout(project)
    assert (ensure_checkout(project) / "a.txt").exists()

def test_add_worktree_at_pinned_commit(project, tmp_path):
    checkout = ensure_checkout(project)
    sha = git(["rev-parse", "HEAD"], cwd=checkout).strip()
    (checkout / "b.txt").write_text("two\n")
    git(["add", "b.txt"], cwd=checkout)
    git(["-c", "user.email=t@t", "-c", "user.name=t",
         "commit", "-qm", "second"], cwd=checkout)

    wt = add_worktree(checkout, tmp_path / "work" / "j1", sha)
    assert (wt / "a.txt").exists()
    assert not (wt / "b.txt").exists()   # pinned to the older commit

def test_two_worktrees_at_different_commits_coexist(project, tmp_path):
    """The reason worktrees exist here at all: concurrent CPU jobs pinned
    to different commits cannot share one working tree."""
    checkout = ensure_checkout(project)
    old = git(["rev-parse", "HEAD"], cwd=checkout).strip()
    (checkout / "b.txt").write_text("two\n")
    git(["add", "b.txt"], cwd=checkout)
    git(["-c", "user.email=t@t", "-c", "user.name=t",
         "commit", "-qm", "second"], cwd=checkout)
    new = git(["rev-parse", "HEAD"], cwd=checkout).strip()

    w1 = add_worktree(checkout, tmp_path / "w" / "j1", old)
    w2 = add_worktree(checkout, tmp_path / "w" / "j2", new)
    assert not (w1 / "b.txt").exists()
    assert (w2 / "b.txt").exists()

def test_remove_worktree_cleans_up(project, tmp_path):
    checkout = ensure_checkout(project)
    sha = git(["rev-parse", "HEAD"], cwd=checkout).strip()
    wt = add_worktree(checkout, tmp_path / "w" / "j1", sha)
    remove_worktree(checkout, wt)
    assert not wt.exists()
    assert "j1" not in git(["worktree", "list"], cwd=checkout)

def test_unknown_commit_raises_git_error(project, tmp_path):
    checkout = ensure_checkout(project)
    with pytest.raises(GitError):
        add_worktree(checkout, tmp_path / "w" / "j1", "deadbee")

def test_commit_artifacts_creates_commit(project, tmp_path):
    checkout = ensure_checkout(project)
    src = tmp_path / "summary.json"
    src.write_text('{"loss": 1}')
    sha = commit_artifacts(project, "main", [src], ["runs/v0/summary.json"],
                           "artifacts: j1")
    assert sha
    assert (checkout / "runs" / "v0" / "summary.json").exists()
    assert "artifacts: j1" in git(["log", "-1", "--pretty=%s"], cwd=checkout)

def test_commit_artifacts_noop_when_unchanged(project, tmp_path):
    checkout = ensure_checkout(project)
    src = tmp_path / "s.json"
    src.write_text("{}")
    commit_artifacts(project, "main", [src], ["runs/s.json"], "first")
    assert commit_artifacts(project, "main", [src], ["runs/s.json"],
                            "again") is None

def test_git_error_includes_stderr(tmp_path):
    with pytest.raises(GitError, match="not a git repository"):
        git(["status"], cwd=tmp_path)
