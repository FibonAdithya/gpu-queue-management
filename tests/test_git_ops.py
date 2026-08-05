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


def _results_origin(tmp_path) -> Path:
    """A separate repo standing in for the writable results remote."""
    o = tmp_path / "results-origin"
    o.mkdir()
    git(["init", "-q", "-b", "main"], cwd=o)
    git(["config", "user.email", "r@r"], cwd=o)
    git(["config", "user.name", "r"], cwd=o)
    git(["config", "receive.denyCurrentBranch", "ignore"], cwd=o)
    (o / "README.md").write_text("results\n")
    git(["add", "-A"], cwd=o)
    git(["commit", "-qm", "init results"], cwd=o)
    return o

@pytest.fixture
def split(tmp_path, project):
    """Code repo read-only in spirit; results repo is the writable one."""
    ro = _results_origin(tmp_path)
    project.results_remote = str(ro)
    project.results_checkout = tmp_path / "results-checkout"
    return project, ro

def test_artifacts_go_to_the_results_repo_not_the_code_checkout(split, tmp_path):
    project, _ = split
    code = ensure_checkout(project)
    src = tmp_path / "summary.json"; src.write_text('{"loss": 1}')
    sha = commit_artifacts(project, "main", [src], ["runs/summary.json"],
                           "artifacts: j1", job_id="j1")
    assert sha
    assert not (code / "runs" / "summary.json").exists()   # code repo untouched
    assert (Path(project.results_checkout) / "p" / "j1" / "runs" / "summary.json").exists()

def test_results_are_namespaced_so_runs_do_not_overwrite(split, tmp_path):
    """A results repo aggregates runs; without namespacing each one silently
    replaces the last and 'results survive' is true only of the newest."""
    project, _ = split
    a = tmp_path / "a.json"; a.write_text('{"run": 1}')
    b = tmp_path / "b.json"; b.write_text('{"run": 2}')
    commit_artifacts(project, "main", [a], ["runs/summary.json"], "j1", job_id="j1")
    commit_artifacts(project, "main", [b], ["runs/summary.json"], "j2", job_id="j2")
    r = Path(project.results_checkout)
    assert (r / "p" / "j1" / "runs" / "summary.json").read_text() == '{"run": 1}'
    assert (r / "p" / "j2" / "runs" / "summary.json").read_text() == '{"run": 2}'

def test_results_are_pushed_so_they_survive_the_box(split, tmp_path):
    project, results_origin = split
    src = tmp_path / "s.json"; src.write_text("{}")
    commit_artifacts(project, "main", [src], ["runs/s.json"], "j1", job_id="j1")
    listed = git(["show", "--name-only", "--format=", "main"], cwd=results_origin)
    assert "p/j1/runs/s.json" in listed

def test_the_code_repo_is_never_pushed_to_in_split_mode(split, tmp_path):
    """The whole point: the box needs no write access to the code repo."""
    project, _ = split
    project.push = False
    code_origin_head = git(["rev-parse", "HEAD"], cwd=Path(project.remote)).strip()
    src = tmp_path / "s.json"; src.write_text("{}")
    commit_artifacts(project, "main", [src], ["runs/s.json"], "j1", job_id="j1")
    assert git(["rev-parse", "HEAD"], cwd=Path(project.remote)).strip() == code_origin_head

def test_without_a_results_repo_behaviour_is_unchanged(project, tmp_path):
    code = ensure_checkout(project)
    src = tmp_path / "s.json"; src.write_text("{}")
    commit_artifacts(project, "main", [src], ["runs/s.json"], "j1", job_id="j1")
    assert (code / "runs" / "s.json").exists()
