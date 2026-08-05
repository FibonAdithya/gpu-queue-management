"""The queue: a directory tree whose transitions are atomic renames."""
from __future__ import annotations

import fcntl
import json
import os
from contextlib import contextmanager
from pathlib import Path

from .spec import JobSpec

STATES = ("pending", "running", "done", "failed")
_ACTIVE = ("pending", "running")


class QueueRoot:
    def __init__(self, root: Path):
        self.root = Path(root)

    # --- layout -------------------------------------------------------
    def ensure_dirs(self) -> None:
        for d in (*STATES, "logs", "work"):
            (self.root / d).mkdir(parents=True, exist_ok=True)

    def path_for(self, state: str, job_id: str) -> Path:
        if state not in STATES:
            raise ValueError(f"unknown state {state!r}")
        return self.root / state / f"{job_id}.json"

    def log_paths(self, job_id: str) -> tuple[Path, Path]:
        return (self.root / "logs" / f"{job_id}.out",
                self.root / "logs" / f"{job_id}.err")

    def work_dir(self, job_id: str) -> Path:
        return self.root / "work" / job_id

    # --- io -----------------------------------------------------------
    def _write(self, path: Path, spec: JobSpec) -> None:
        """Write via a temp file in the same directory, then rename.

        A reader must never see a half-written spec; rename(2) is the only
        way to publish one atomically.
        """
        tmp = path.with_suffix(".json.part")
        tmp.write_text(json.dumps(spec.to_dict(), indent=2) + "\n")
        os.rename(tmp, path)

    def _read(self, path: Path) -> JobSpec:
        return JobSpec.from_dict(json.loads(path.read_text()))

    @contextmanager
    def _submit_lock(self):
        """Serialize dedupe-check-then-write across concurrent submitters."""
        self.root.mkdir(parents=True, exist_ok=True)
        lock = self.root / ".submit.lock"
        fd = os.open(lock, os.O_CREAT | os.O_RDWR, 0o666)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            os.close(fd)

    # --- transitions --------------------------------------------------
    def submit(self, spec: JobSpec) -> str:
        spec.validate()
        with self._submit_lock():
            if spec.dedupe_key:
                for state in _ACTIVE:
                    for other in self.list_state(state):
                        if other.dedupe_key == spec.dedupe_key:
                            return other.id
            dest = self.path_for("pending", spec.id)
            if any(self.path_for(s, spec.id).exists() for s in STATES):
                raise FileExistsError(f"job id already exists: {spec.id}")
            self._write(dest, spec)
            return spec.id

    def claim(self, job_id: str) -> JobSpec | None:
        """pending -> running. None if another thread got there first."""
        src = self.path_for("pending", job_id)
        dst = self.path_for("running", job_id)
        try:
            os.rename(src, dst)
        except FileNotFoundError:
            return None
        return self._read(dst)

    def update(self, spec: JobSpec, state: str = "running") -> None:
        """Rewrite a spec where it already is. The runner records a pid this
        way, so the reaper can tell a live job from an abandoned one."""
        self._write(self.path_for(state, spec.id), spec)

    def finish(self, spec: JobSpec, ok: bool) -> None:
        state = "done" if ok else "failed"
        self._write(self.path_for("running", spec.id), spec)
        os.rename(self.path_for("running", spec.id),
                  self.path_for(state, spec.id))

    def requeue(self, spec: JobSpec) -> None:
        spec.attempts += 1
        spec.pid = None
        self._write(self.path_for("running", spec.id), spec)
        os.rename(self.path_for("running", spec.id),
                  self.path_for("pending", spec.id))

    def cancel(self, job_id: str) -> bool:
        """Cancel a pending job. Running jobs are the runner's to stop."""
        src = self.path_for("pending", job_id)
        try:
            spec = self._read(src)
        except FileNotFoundError:
            return False
        spec.error = "cancelled"
        self._write(src, spec)
        try:
            os.rename(src, self.path_for("failed", job_id))
        except FileNotFoundError:
            return False
        return True

    # --- queries ------------------------------------------------------
    def list_state(self, state: str) -> list[JobSpec]:
        d = self.root / state
        if not d.is_dir():
            return []
        out = []
        for p in sorted(d.glob("*.json")):
            try:
                out.append(self._read(p))
            except Exception:
                continue  # a hand-repairable queue tolerates one bad file
        return out

    def find(self, job_id: str) -> tuple[str, JobSpec] | None:
        for state in STATES:
            p = self.path_for(state, job_id)
            if p.exists():
                try:
                    return state, self._read(p)
                except Exception:
                    return None
        return None
