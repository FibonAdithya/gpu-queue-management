"""Runner configuration. Every project the runner serves is declared here."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import tomllib  # stdlib from 3.11, which is why 3.11 is the floor

from .ledger import DEFAULT_RESERVE_MB


class ConfigError(ValueError):
    """The configuration file is missing or malformed."""


# One source of truth with the standalone gpu-claim path, which reads the
# same key through `max_holders` and has no config to default from.
DEFAULT_MAX_HOLDERS = 2


@dataclass
class ProjectConfig:
    name: str
    remote: str
    checkout: Path
    venv: Path | None = None
    commit_artifacts: bool = False
    push: bool = False
    # Optional split: publish artifacts to a *different* repository from the
    # one the code is checked out from. This is what lets the box hold a
    # read-only key for your code and a write key only for results.
    results_remote: str | None = None
    results_checkout: Path | None = None
    results_branch: str = "main"


@dataclass
class AutofixConfig:
    """Where gpuq files bugs against itself, and how hard it is allowed to try.

    Off by default. Turning it on means the box holds a GitHub token, so it
    is never something a config inherits by accident.
    """
    enabled: bool = False
    repo: str | None = None            # owner/name
    # Which environment variable holds the fine-grained PAT. Named rather
    # than GH_TOKEN so an issues-only key is not inherited by every other
    # tool on the box that speaks to GitHub.
    token_env: str = "GPUQ_GITHUB_TOKEN"
    max_dispatches_per_day: int = 3
    closed_lookback_days: int = 30
    state_file: Path | None = None     # defaults to <queue_root>/autofix.json
    # Per-signature cooldown, in-process, before `Runner._report_bug` will
    # call `file_bug` again for the same bug. `admit()` can attempt every
    # pending job in one tick, and a failed `_launch` never enters
    # `self.active`, so a broken `git_ops` on a box with 20 queued jobs
    # would otherwise call `file_bug` -- up to three `gh` subprocesses at
    # 30s each -- once per pending job, stalling `admit` for tens of
    # minutes with `poll_job` never running in between. 900s (15 minutes)
    # bounds that to a handful of `gh` round trips per bug per tick cycle,
    # while still refreshing the issue's occurrence count and comment at a
    # human-relevant cadence if the bug persists.
    report_cooldown_s: float = 900.0


@dataclass
class RunnerConfig:
    queue_root: Path
    cpu_slots: int = 4
    # None means "ask the card". An explicit value is for boxes where the
    # query is unavailable or reports something the driver will not
    # actually hand out.
    gpu_vram_mb: int | None = None
    # One source of truth with the standalone gpu-claim path, which needs
    # the same number and has no config to read it from.
    gpu_vram_reserve_mb: int = DEFAULT_RESERVE_MB
    # A latency budget, not a safety one. VRAM accounting alone would admit
    # sixteen 500 MiB jobs onto an 8 GB card, all time-slicing, each slower
    # than it would have been queued -- and with independent submitters
    # that cost lands on a stranger. Two is what has been measured (15% to
    # 62% utilization); raise it on a box that has measured more.
    #
    # Enforced in `ledger.acquire`, which counts every holder of the card,
    # not just in `Runner._capacity`, which counts only this runner's own
    # jobs. A budget the hand-run `gpu-claim` path walked past was not
    # bounding the contention it was written to bound.
    gpu_max_jobs: int = DEFAULT_MAX_HOLDERS
    enforce_vram: bool = True
    poll_interval_s: float = 2.0
    claim_dir: Path | None = None
    kill_orphan_cuda: bool = True
    orphan_cuda_interval_s: float = 60.0
    projects: dict[str, ProjectConfig] = field(default_factory=dict)
    autofix: AutofixConfig = field(default_factory=AutofixConfig)


_TRUE = {"true", "yes", "on", "1"}
_FALSE = {"false", "no", "off", "0"}


def _as_bool(value, key: str, default: bool) -> bool:
    """A TOML bool, or a spelling of one -- never `bool(value)`.

    `enforce_vram = "false"` (quoted, which TOML accepts as a string) went
    through `bool("false")` and came out True, switching the watchdog *on*
    for an operator who wrote it to turn the killing off, and who was told
    by gpuq.example.toml that it was an off switch. Anything that is not
    obviously one or the other is a ConfigError: a config this file cannot
    read is worth a loud failure at startup, where guessing is worth a
    dead job hours later.

    `0` and `1` are accepted because `bool(value)` accepted them: a
    stricter reader is worth an unreadable string, and not worth refusing
    to start the daemon on a config that already worked before an upgrade.
    Any other int is as unreadable as "sometimes".
    """
    if value is None:
        return default
    if isinstance(value, bool):     # before int: bool *is* an int
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        s = value.strip().lower()
        if s in _TRUE:
            return True
        if s in _FALSE:
            return False
    raise ConfigError(f"{key} must be true or false, got {value!r}")


def vram_policy(path: Path | None = None) -> tuple[int | None, int]:
    """`(gpu_vram_mb, gpu_vram_reserve_mb)`, for callers that are not the runner.

    A standalone `gpu-claim` shares one `<key>.lock.d` with the runner but
    has no config of its own, so it sized the card from nvidia-smi while
    the runner sized it from `[queue].gpu_vram_mb`. An operator who follows
    docs/deploying.md and declares a card smaller than nvidia-smi reports
    -- the documented fix for a driver that will not hand out what the
    query claims -- then has two participants admitting against different
    totals into the same ledger, which is the double-booking the ledger
    exists to prevent.

    Deliberately not `load_config`: a caller that only wants the card's
    size must not be refused because `[queue].root` is missing or a project
    is half-declared. It would fall back to a capacity the runner does not
    share, which is the divergence this closes. For the same reason an
    unreadable or absent file is the defaults rather than an error -- that
    is a box with no runner deployed, and exactly what this path did before
    there was a config to read.
    """
    p = Path(path) if path else default_config_path()
    try:
        queue = tomllib.loads(p.read_text()).get("queue") or {}
        total = queue.get("gpu_vram_mb")
        total = int(total) if total is not None else None
        reserve = int(queue.get("gpu_vram_reserve_mb", DEFAULT_RESERVE_MB))
    except Exception:
        return None, DEFAULT_RESERVE_MB
    # load_config rejects these outright, but this reader runs in a process
    # that has no runner to refuse to start; a nonsense reserve falls back
    # rather than propagating a negative capacity into ledger.fits.
    if reserve < 0 or (total is not None and reserve >= total):
        return total, DEFAULT_RESERVE_MB
    return total, reserve


def max_holders(path: Path | None = None) -> int:
    """`[queue].gpu_max_jobs`, for callers that are not the runner.

    The cap is a property of the card, not of the runner: it bounds how
    many processes time-slice on it, and a hand-run `gpu-claim` occupies a
    slot exactly as a queued job does. Enforced in `ledger.acquire` rather
    than in `Runner._capacity` alone for that reason, which means this
    reader is what a standalone claim admits against.

    Defensive in the same three ways `vram_policy` is, and for the same
    reasons: not `load_config`, an unreadable file is the default rather
    than an error, and a value `load_config` would reject falls back
    instead of propagating. A zero or negative cap here would refuse every
    claim on the box, including the runner's -- silently, with no daemon
    around to fail loudly at startup.
    """
    p = Path(path) if path else default_config_path()
    try:
        queue = tomllib.loads(p.read_text()).get("queue") or {}
        n = int(queue.get("gpu_max_jobs", DEFAULT_MAX_HOLDERS))
    except Exception:
        return DEFAULT_MAX_HOLDERS
    return n if n >= 1 else DEFAULT_MAX_HOLDERS


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

    gpu_vram_mb = queue.get("gpu_vram_mb")
    gpu_vram_mb = int(gpu_vram_mb) if gpu_vram_mb is not None else None
    gpu_vram_reserve_mb = int(queue.get("gpu_vram_reserve_mb",
                                        DEFAULT_RESERVE_MB))
    gpu_max_jobs = int(queue.get("gpu_max_jobs", DEFAULT_MAX_HOLDERS))
    if gpu_max_jobs < 1:
        raise ConfigError("[queue].gpu_max_jobs must be >= 1")
    if gpu_vram_reserve_mb < 0:
        raise ConfigError("[queue].gpu_vram_reserve_mb must be >= 0")
    if gpu_vram_mb is not None and gpu_vram_reserve_mb >= gpu_vram_mb:
        raise ConfigError(
            f"[queue].gpu_vram_reserve_mb ({gpu_vram_reserve_mb}) must be "
            f"less than gpu_vram_mb ({gpu_vram_mb}); a reserve that "
            "swallows the card admits nothing and queues GPU jobs forever")

    projects: dict[str, ProjectConfig] = {}
    for name, p in (data.get("project") or {}).items():
        if not p.get("checkout"):
            raise ConfigError(f"[project.{name}].checkout is required")
        if not p.get("remote"):
            raise ConfigError(f"[project.{name}].remote is required")
        r_remote, r_checkout = p.get("results_remote"), p.get("results_checkout")
        if bool(r_remote) != bool(r_checkout):
            raise ConfigError(
                f"[project.{name}]: results_remote and results_checkout must be "
                "set together; one without the other has nowhere to publish")
        projects[name] = ProjectConfig(
            name=name,
            remote=p["remote"],
            checkout=Path(p["checkout"]),
            venv=Path(p["venv"]) if p.get("venv") else None,
            commit_artifacts=bool(p.get("commit_artifacts", False)),
            push=bool(p.get("push", False)),
            results_remote=r_remote,
            results_checkout=Path(r_checkout) if r_checkout else None,
            results_branch=p.get("results_branch", "main"),
        )

    claim_dir = queue.get("claim_dir")

    a = data.get("autofix") or {}
    repo = a.get("repo")
    enabled = bool(a.get("enabled", False))
    if enabled and not repo:
        raise ConfigError("[autofix].repo is required when enabled")
    if repo and (repo.count("/") != 1 or repo.startswith(("http", "git@"))):
        raise ConfigError(f"[autofix].repo must be owner/name, got {repo!r}")
    state_file = a.get("state_file")
    autofix = AutofixConfig(
        enabled=enabled,
        repo=repo,
        token_env=a.get("token_env", "GPUQ_GITHUB_TOKEN"),
        max_dispatches_per_day=int(a.get("max_dispatches_per_day", 3)),
        closed_lookback_days=int(a.get("closed_lookback_days", 30)),
        state_file=Path(state_file) if state_file
                   else Path(root) / "autofix.json",
        report_cooldown_s=float(a.get("report_cooldown_s", 900.0)),
    )

    return RunnerConfig(
        queue_root=Path(root),
        cpu_slots=cpu_slots,
        gpu_vram_mb=gpu_vram_mb,
        gpu_vram_reserve_mb=gpu_vram_reserve_mb,
        gpu_max_jobs=gpu_max_jobs,
        # The two keys that decide whether gpuq kills something. Every
        # other bool here still goes through bool(), where a quoted
        # "false" means a feature quietly stays on rather than a job
        # being killed -- worth fixing, not worth widening this change.
        enforce_vram=_as_bool(queue.get("enforce_vram"),
                              "[queue].enforce_vram", True),
        poll_interval_s=float(queue.get("poll_interval_s", 2.0)),
        claim_dir=Path(claim_dir) if claim_dir else None,
        kill_orphan_cuda=_as_bool(queue.get("kill_orphan_cuda"),
                                  "[queue].kill_orphan_cuda", True),
        orphan_cuda_interval_s=float(queue.get("orphan_cuda_interval_s", 60.0)),
        projects=projects,
        autofix=autofix,
    )


def claim_dir_setting(path: Path | None = None) -> Path | None:
    """`[queue].claim_dir`, for a process that is not the runner.

    A hand-run `gpu-claim` and the daemon share one card, and each writes
    into whatever `$GPU_CLAIM_DIR` resolves to in *its own* process. The
    daemon's comes from a supervisor unit; an interactive shell never
    inherits one. The config file is the only thing both of them can read,
    so it is how the interactive side finds out it is claiming into a
    directory the runner is not looking at -- which cost 48% of one
    session's runs (issue #19).

    None means "no configured directory to disagree with", not "they
    agree". With the key unset the daemon reads its own environment, which
    this process cannot see; warning on a guess would fire on every
    correctly configured box, and a warning that is usually wrong is one
    operators learn to skip past.

    An empty string is None for the same reason `load_config` treats it as
    unset (`Path(claim_dir) if claim_dir else None`): a reader that
    disagreed with the runner about what the file says would report a
    divergence the runner does not have.

    Defensive in the three ways `vram_policy` and `max_holders` are, and
    for the same reasons: not `load_config`, so a caller wanting one key
    is not refused because `[queue].root` is missing; an unreadable or
    absent file is None rather than an error, because this runs in a
    process with no daemon to fail loudly at startup.
    """
    p = Path(path) if path else default_config_path()
    try:
        queue = tomllib.loads(p.read_text()).get("queue") or {}
        value = queue.get("claim_dir")
    except Exception:
        return None
    return Path(value) if value else None
