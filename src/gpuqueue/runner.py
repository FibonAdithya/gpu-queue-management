"""The sole launcher of queued work on this box.

One thread, one loop. `tick()` reaps, collects what has finished, and admits
what the lanes allow; `run_forever` is a sleep around it. Concurrency is
`Popen`, not threads — several jobs run at once because they are separate
processes, and the runner supervises them by polling.

That is why there is no lock in this file. A thread per job would put a
worker and this loop on the same queue files, the same lane counters and the
same JobSpec objects; single-threaded, none of that state is shared, and
every git call is trivially serialized because there is only one caller.
"""
from __future__ import annotations

import logging
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from . import git_ops
from .claim import gpu_claim, ClaimBusy
from .config import RunnerConfig, ProjectConfig
from .executor import (start_job, poll_job, kill_job, JobResult, RunningJob,
                       StartFailed)
from .gpuid import gpu_key, GpuIdError
from .preflight import preflight, PreflightFailed
from .queue import QueueRoot
from .reaper import reap
from .spec import JobSpec

log = logging.getLogger("gpuqueue.runner")


@dataclass
class Active:
    running: RunningJob
    project: ProjectConfig
    workdir: Path
    claim_cm: object | None = None  # entered gpu_claim, released on settle


class Runner:
    def __init__(self, cfg: RunnerConfig):
        self.cfg = cfg
        self.queue = QueueRoot(cfg.queue_root)
        self.queue.ensure_dirs()
        self.active: dict[str, Active] = {}
        self._stopping = False
        self._reap_due = True  # on start, then after every completion

    # --- lifecycle ----------------------------------------------------
    def run_forever(self) -> None:
        while not self._stopping:
            self.tick()
            if self._stopping:
                break
            time.sleep(self.cfg.poll_interval_s)
        self.shutdown()

    def stop(self) -> None:
        """Signal-handler safe: sets a flag, does no work."""
        self._stopping = True

    def tick(self) -> None:
        if self._reap_due:
            reap(self.queue, self.cfg, active_ids=set(self.active))
            self._reap_due = False
        self.collect()
        self.admit()

    def shutdown(self) -> None:
        """Kill what is running and leave it in running/ with no pid.

        Deciding the fate of an interrupted job is the reaper's business, and
        it already has the rule: requeue once, then fail. Duplicating that
        here would give a job two different retry policies depending on how
        the runner died.
        """
        for job_id, active in list(self.active.items()):
            self.active.pop(job_id, None)
            # Each step is guarded separately. The queue is meant to be
            # repairable with mv, so a spec file may have moved out from
            # under us — and one job's problem must never strand another
            # job's hold on the card.
            try:
                kill_job(active.running)
            except Exception as e:
                log.warning("shutdown: could not kill %s: %s", job_id, e)
            try:
                self._release_card(active)
            except Exception as e:
                log.error("shutdown: could not release the card after %s: %s",
                          job_id, e)
            try:
                self._remove_worktree(active)
                spec = active.running.spec
                spec.pid = None
                self.queue.update(spec)
            except Exception as e:
                log.warning("shutdown: could not record %s: %s", job_id, e)
            log.info("stopped %s for shutdown; left for the reaper", job_id)

    # --- admission ----------------------------------------------------
    def _capacity(self, lane: str) -> int:
        limit = self.cfg.cpu_slots if lane == "cpu" else 1
        in_lane = sum(1 for a in self.active.values()
                      if a.running.spec.lane == lane)
        return limit - in_lane

    def admit(self) -> list[str]:
        if self._stopping:
            return []
        pending = sorted(self.queue.list_state("pending"),
                         key=lambda s: (s.submitted_at, s.id))
        started = []
        for spec in pending:
            project = self.cfg.projects.get(spec.project)
            if project is None:
                self._fail_pending(spec, f"unknown project {spec.project!r}; "
                                         "declare it in the runner config")
                continue
            if self._capacity(spec.lane) <= 0:
                continue

            # Take the card before the rename: a job that never reached
            # running/ needs no unwinding if the card is busy.
            claim_cm = None
            if spec.lane == "gpu":
                claim_cm = self._take_card(spec)
                if claim_cm is None:
                    continue

            claimed = self.queue.claim(spec.id)
            if claimed is None:  # cancelled, or another process won the rename
                self._exit_claim(claim_cm)
                continue

            if self._launch(claimed, project, claim_cm):
                started.append(claimed.id)
        return started

    def _take_card(self, spec: JobSpec):
        """Enter a gpu_claim by hand so it can be held across ticks.

        Non-blocking on purpose: `wait=True` in a single-threaded loop would
        stall the CPU lane behind whoever holds the card.
        """
        try:
            preflight()
        except PreflightFailed as e:
            log.warning("%s waiting: %s", spec.id, e)
            return None
        try:
            key = gpu_key()
        except GpuIdError as e:
            # No card will appear on a box that has none; do not queue forever.
            self._fail_pending(spec, f"no usable GPU: {e}")
            return None
        cm = gpu_claim(key=key, owner=f"gpuq:{spec.id}", cmd=spec.cmd,
                       wait=False, directory=self.cfg.claim_dir)
        try:
            cm.__enter__()
        except ClaimBusy as e:
            log.info("%s waiting: %s", spec.id, e)
            return None
        return cm

    def _launch(self, spec: JobSpec, project: ProjectConfig, claim_cm) -> bool:
        try:
            workdir = self._prepare_workdir(spec, project)  # git, on the loop
        except Exception as e:
            self._exit_claim(claim_cm)
            self._fail_running(spec, f"checkout failed: {e}")
            return False
        out_log, err_log = self.queue.log_paths(spec.id)
        try:
            running = start_job(
                spec, workdir, out_log, err_log, project=project,
                extra_env={"GPUQ_JOB_ID": spec.id,
                           "GPUQ_QUEUE_ROOT": str(self.queue.root)})
        except StartFailed as e:
            self._exit_claim(claim_cm)
            self._remove_worktree_at(project, workdir)
            self._fail_running(spec, str(e))
            return False

        spec.pid = running.pid
        # Record the owner too: if this runner dies, a live job with a dead
        # runner is an orphan nothing is supervising, and `gpuq list` says so.
        spec.runner_pid = os.getpid()
        self.queue.update(spec)  # the reaper reads this to tell live from dead
        self.active[spec.id] = Active(running=running, project=project,
                                      workdir=workdir, claim_cm=claim_cm)
        log.info("started %s (%s lane, pid %d)", spec.id, spec.lane, running.pid)
        return True

    def _prepare_workdir(self, spec: JobSpec, project: ProjectConfig) -> Path:
        checkout = git_ops.ensure_checkout(project)
        git_ops.git(["fetch", "--quiet", "origin"], cwd=checkout, check=False)
        return git_ops.add_worktree(checkout, self.queue.work_dir(spec.id),
                                    spec.commit)

    # --- collection ---------------------------------------------------
    def collect(self) -> list[str]:
        finished = []
        for job_id, active in list(self.active.items()):
            result = poll_job(active.running)
            if result is None:
                continue
            self.active.pop(job_id, None)
            self._settle(active, result)
            finished.append(job_id)
        if finished:
            self._reap_due = True  # design: reap between jobs
        return finished

    def _settle(self, active: Active, result: JobResult) -> None:
        spec = active.running.spec
        # Free the card before the git work: artifact commits can take a
        # while and nothing about them needs the GPU.
        self._release_card(active)

        spec.pid = None
        spec.exit_code = result.exit_code
        ok = result.exit_code == 0 and not result.timed_out
        if ok:
            try:
                self._collect_artifacts(spec, active.project, active.workdir)
            except Exception as e:
                ok = False
                spec.error = str(e)
        if not ok and spec.error is None:
            spec.error = self._describe_failure(spec, result)

        self._remove_worktree(active)
        self.queue.finish(spec, ok=ok)
        log.info("%s %s", spec.id, "done" if ok else f"failed: {spec.error}")

    def _collect_artifacts(self, spec: JobSpec, project: ProjectConfig,
                           workdir: Path) -> None:
        if not spec.artifacts:
            return
        srcs, rels = [], []
        for rel in spec.artifacts:
            src = workdir / rel
            if not src.exists():
                raise RuntimeError(f"declared artifact not produced: {rel}")
            srcs.append(src)
            rels.append(rel)
        if project.commit_artifacts:
            git_ops.commit_artifacts(project, spec.branch, srcs, rels,
                                     f"artifacts: {spec.id}", job_id=spec.id)

    def _describe_failure(self, spec: JobSpec, result: JobResult) -> str:
        if result.timed_out:
            return (f"timeout after {spec.timeout_s}s; killed. A hung job "
                    "is a bug, not a transient — not retried.")
        if result.oom:
            return ("CUDA out of memory — a configuration error, not a "
                    f"transient; not retried.\n{result.stderr_tail}")
        return f"exit {result.exit_code}\n{result.stderr_tail}"

    # --- teardown helpers ---------------------------------------------
    def _release_card(self, active: Active) -> None:
        self._exit_claim(active.claim_cm)
        active.claim_cm = None

    @staticmethod
    def _exit_claim(claim_cm) -> None:
        if claim_cm is not None:
            claim_cm.__exit__(None, None, None)

    def _remove_worktree(self, active: Active) -> None:
        self._remove_worktree_at(active.project, active.workdir)

    @staticmethod
    def _remove_worktree_at(project: ProjectConfig, workdir: Path) -> None:
        try:
            git_ops.remove_worktree(Path(project.checkout), workdir)
        except Exception:
            shutil.rmtree(workdir, ignore_errors=True)

    # --- failure helpers ----------------------------------------------
    def _fail_pending(self, spec: JobSpec, message: str) -> None:
        claimed = self.queue.claim(spec.id)
        if claimed:
            self._fail_running(claimed, message)

    def _fail_running(self, spec: JobSpec, message: str) -> None:
        spec.error = message
        spec.exit_code = -1
        spec.pid = None
        self.queue.finish(spec, ok=False)
        log.warning("%s failed: %s", spec.id, message)
