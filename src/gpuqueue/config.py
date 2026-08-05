"""Runner configuration. Every project the runner serves is declared here."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import tomllib  # stdlib from 3.11, which is why 3.11 is the floor


class ConfigError(ValueError):
    """The configuration file is missing or malformed."""


@dataclass
class ProjectConfig:
    name: str
    remote: str
    checkout: Path
    venv: Path | None = None
    commit_artifacts: bool = False
    push: bool = False


@dataclass
class RunnerConfig:
    queue_root: Path
    cpu_slots: int = 4
    poll_interval_s: float = 2.0
    claim_dir: Path | None = None
    kill_orphan_cuda: bool = True
    projects: dict[str, ProjectConfig] = field(default_factory=dict)


def default_config_path() -> Path:
    return Path(os.environ.get("GPUQ_CONFIG", "/workspace/gpuq.toml"))


def load_config(path: Path) -> RunnerConfig:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config not found: {path}")
    data = tomllib.loads(path.read_text())

    queue = data.get("queue") or {}
    root = queue.get("root")
    if not root:
        raise ConfigError("[queue].root is required")
    cpu_slots = int(queue.get("cpu_slots", 4))
    if cpu_slots < 1:
        raise ConfigError("[queue].cpu_slots must be >= 1")

    projects: dict[str, ProjectConfig] = {}
    for name, p in (data.get("project") or {}).items():
        if not p.get("checkout"):
            raise ConfigError(f"[project.{name}].checkout is required")
        if not p.get("remote"):
            raise ConfigError(f"[project.{name}].remote is required")
        projects[name] = ProjectConfig(
            name=name,
            remote=p["remote"],
            checkout=Path(p["checkout"]),
            venv=Path(p["venv"]) if p.get("venv") else None,
            commit_artifacts=bool(p.get("commit_artifacts", False)),
            push=bool(p.get("push", False)),
        )

    claim_dir = queue.get("claim_dir")
    return RunnerConfig(
        queue_root=Path(root),
        cpu_slots=cpu_slots,
        poll_interval_s=float(queue.get("poll_interval_s", 2.0)),
        claim_dir=Path(claim_dir) if claim_dir else None,
        kill_orphan_cuda=bool(queue.get("kill_orphan_cuda", True)),
        projects=projects,
    )
