"""All git. Called only from Runner.tick().

Jobs write artifacts into their own worktree; the loop moves them into the
checkout and commits between polls. The runner has one thread, so there is
one caller here and repository mutation is serialized by construction.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .config import ProjectConfig

_IDENTITY = ["-c", "user.email=gpuq@localhost", "-c", "user.name=gpuq-runner"]


class GitError(RuntimeError):
    pass


def git(args: list[str], cwd: Path | None = None, check: bool = True) -> str:
    proc = subprocess.run(["git", *args], cwd=cwd, text=True,
                          capture_output=True)
    if check and proc.returncode != 0:
        raise GitError(f"git {' '.join(args)} failed ({proc.returncode}): "
                       f"{proc.stderr.strip()}")
    return proc.stdout


def ensure_checkout(project: ProjectConfig) -> Path:
    path = Path(project.checkout)
    if (path / ".git").exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    git(["clone", project.remote, str(path)])
    return path


def add_worktree(checkout: Path, dest: Path, commit: str) -> Path:
    dest = Path(dest)
    if dest.exists():
        remove_worktree(checkout, dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    git(["worktree", "add", "--detach", str(dest), commit], cwd=checkout)
    return dest


def remove_worktree(checkout: Path, dest: Path) -> None:
    git(["worktree", "remove", "--force", str(dest)], cwd=checkout, check=False)
    if Path(dest).exists():
        shutil.rmtree(dest, ignore_errors=True)
    git(["worktree", "prune"], cwd=checkout, check=False)


def ensure_results_checkout(project: ProjectConfig) -> Path | None:
    """Clone the results repository, if this project publishes to one."""
    if not (project.results_remote and project.results_checkout):
        return None
    path = Path(project.results_checkout)
    if not (path / ".git").exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        git(["clone", project.results_remote, str(path)])
    return path


def commit_artifacts(project: ProjectConfig, branch: str,
                     files: list[Path], rel_paths: list[str],
                     message: str, job_id: str | None = None) -> str | None:
    """Publish a job's declared artifacts. Returns the new sha, or None.

    Two arrangements. By default artifacts are committed into the project's
    own checkout, and pushed only if `push` is set.

    If the project declares a results repository, they go there instead. That
    is what lets a shared box hold a read-only key for your code and a write
    key that reaches nothing but results — anyone who can queue a job on the
    box can push with it, so the smaller that blast radius, the better.

    In a results repository the paths are namespaced `<project>/<job>/<path>`.
    A results repo aggregates many runs and many projects; committing at the
    bare declared path means each run silently overwrites the last, and
    "results survive the box" would only be true of the most recent one.
    """
    results = ensure_results_checkout(project)
    if results is not None:
        checkout, branch, push = results, project.results_branch, True
        prefix = Path(project.name) / (job_id or "unknown")
        rel_paths = [str(prefix / r) for r in rel_paths]
    else:
        checkout, push = ensure_checkout(project), project.push

    git(["checkout", "-q", branch], cwd=checkout, check=False)
    for src, rel in zip(files, rel_paths):
        dst = checkout / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if Path(src).is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
        git(["add", "--", rel], cwd=checkout)
    if not git(["status", "--porcelain"], cwd=checkout).strip():
        return None
    git([*_IDENTITY, "commit", "-qm", message], cwd=checkout)
    sha = git(["rev-parse", "HEAD"], cwd=checkout).strip()
    if push:
        git(["push", "origin", f"HEAD:{branch}"], cwd=checkout, check=False)
    return sha
