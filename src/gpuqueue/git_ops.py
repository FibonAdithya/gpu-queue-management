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


def commit_artifacts(project: ProjectConfig, branch: str,
                     files: list[Path], rel_paths: list[str],
                     message: str) -> str | None:
    checkout = ensure_checkout(project)
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
    if project.push:
        git(["push", "origin", f"HEAD:{branch}"], cwd=checkout, check=False)
    return sha
