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

from . import bugfiler
from . import git_ops
from .bugreport import CallerError, signature
from .claim import gpu_claim, ClaimBusy
from .config import RunnerConfig, ProjectConfig
from .executor import (start_job, poll_job, kill_job, JobResult, RunningJob,
                       StartFailed)
from .gpuid import gpu_key, GpuIdError
from .preflight import preflight, PreflightFailed
from .queue import QueueRoot, STATES
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
        # None means "never swept", so the first tick does a full one.
        self._last_cuda_sweep: float | None = None
        # signature -> time.monotonic() of the last report. See the
        # cooldown check in _report_bug for why this exists.
        self._report_cooldowns: dict[str, float] = {}

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
        self._phase("reap", self._reap)
        self._phase("execute", self.collect)
        self._phase("admit", self.admit)

    def _phase(self, phase: str, fn) -> None:
        """Report an unhandled exception, then let it out.

        Re-raised deliberately: a runner whose reaper is broken should still
        die and be restarted by supervisor, exactly as before. Reporting is
        additive, and a filer that changed crash semantics would be a second
        bug shipped alongside the first.

        Some phases nest: `_take_card` already reports and re-raises a
        preflight exception before it reaches `admit`, which is itself
        wrapped in `_phase("admit", ...)`. Without the tag check below that
        same exception would be filed a second time here, under a different
        phase string -- and because `signature()` hashes phase into the
        payload, that is a second, undeduped issue for one crash. The
        `__gpuq_reported__` tag set by `_report_bug` is how the inner,
        more specific site keeps the outer one from re-filing.
        """
        try:
            fn()
        except Exception as e:
            if not getattr(e, "__gpuq_reported__", False):
                self._report_bug(e, phase)
            raise

    def _report_bug(self, exc: BaseException, phase: str,
                    spec: JobSpec | None = None) -> None:
        """File a bug against gpuq, and never let that break the queue.

        Every path into bugfiler goes through here. A box with no `gh`, no
        token or no network must run jobs exactly as it does without
        autofix; losing a bug report is an acceptable outcome, losing a job
        is not.

        Filing a new bug is slow in a way worth knowing about: a first
        occurrence runs a git call, three dedup lookups, up to four label
        creations and an issue creation, each a subprocess capped at
        `bugfiler.GH_TIMEOUT_S`. On a network-partitioned box that is
        minutes, not seconds -- on the single thread that also polls every
        running job. Two things bound it, and neither is arithmetic that
        survives a refactor: caller faults (`CallerError`, errno-classified
        `StartFailed`) return before any `gh` subprocess runs, and the
        per-signature cooldown below means a persisting bug pays that cost
        once rather than once per occurrence.
        """
        # Tag first, and regardless of what filing does below: the tag means
        # "this exception already had its one chance to be reported", not
        # "filing succeeded". A caller further up the stack (see `_phase`)
        # must not re-file it just because this attempt failed or was a
        # no-op. Guarded because an exception type with __slots__ would
        # otherwise turn a cosmetic dedup problem into a crash inside the
        # one function that must never raise.
        try:
            exc.__gpuq_reported__ = True
        except Exception:
            pass
        if not self.cfg.autofix.enabled:
            return
        try:
            # Computed before any I/O, and inside this try: signature()
            # raises ValueError on a phase outside PHASES, and this
            # function must never raise. `admit()` can reach here once per
            # pending job in a single tick with none of them entering
            # `self.active` (a failed `_launch` never does), so without a
            # cooldown a broken git_ops with 20 queued jobs means 20 calls
            # into file_bug -- up to three `gh` subprocesses at 30s each --
            # before admit() returns. That stalls `poll_job` for the same
            # window, so a running job's deadline goes unenforced and a
            # timed-out job is never killed: worse than a crash, since
            # supervisor restarts a crashed runner in seconds.
            #
            # Suppressed occurrences lose their issue comment and
            # occurrence bump -- that is a deliberate, correct trade.
            # Recurrence counting on the issue is a nice-to-have; not
            # stalling the reaper is not. Do not "fix" this back.
            #
            # The stamp is written before file_bug, so a failed attempt
            # (GitHub down, token expired) also spends the cooldown and
            # this bug goes unfiled for the next window. Deliberate: the
            # thing being bounded is how often the loop can block on `gh`,
            # and a call that fails slowly blocks it exactly as hard as one
            # that succeeds. A persisting bug refiles at the next window;
            # one that stopped happening was never worth the stall.
            sig = signature(exc, phase)
            now = time.monotonic()
            last = self._report_cooldowns.get(sig)
            if last is not None and now - last < self.cfg.autofix.report_cooldown_s:
                return
            self._report_cooldowns[sig] = now
            counts = {s: len(self.queue.list_state(s)) for s in STATES}
            outcome = bugfiler.file_bug(self.cfg.autofix, exc, phase,
                                        spec=spec, queue_counts=counts)
            log.info("autofix (%s): %s", phase, outcome)
        except Exception as e:
            log.warning("autofix: could not file a bug for %s: %s", phase, e)

    def _reap(self) -> None:
        """Recover abandoned work every tick; sweep the card on a timer.

        The recovery path — requeueing a job whose runner died, releasing its
        claim, clearing debris — is three file operations, so it runs every
        time. Gating it on job completions meant an idle runner never reaped
        at all, which is precisely when there is something to recover.

        The orphaned-CUDA sweep is the expensive part: a nvidia-smi subprocess
        plus a walk of the process tree. It is a safety net rather than a
        recovery path, so it runs at most once per orphan_cuda_interval_s and
        stays out of a loop that also gates job admission.
        """
        now = time.monotonic()
        sweep = (self._last_cuda_sweep is None
                 or now - self._last_cuda_sweep >= self.cfg.orphan_cuda_interval_s)
        reap(self.queue, self.cfg, active_ids=set(self.active),
             include_orphan_cuda=sweep)
        if sweep:
            self._last_cuda_sweep = now

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
        except Exception as e:
            self._report_bug(e, "preflight", spec)
            raise
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
            # Card release first: one job's problem must never strand
            # another job's hold on the card, and `_report_bug` never
            # raises but that is not a property this cleanup should have to
            # depend on to be unconditional.
            self._exit_claim(claim_cm)
            self._fail_running(spec, f"checkout failed: {e}")
            self._report_bug(e, "checkout", spec)
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
            self._report_bug(e, "execute", spec)
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
        # "I submitted at a commit I forgot to push" is the likeliest caller
        # mistake in this whole system. Left unchecked, `add_worktree` below
        # fails on it just the same as it would on a genuine gpuq/git fault,
        # and `_launch` reports both as phase `checkout` -- filing a gpuq bug
        # (and burning one of three daily dispatches) for a Claude run that
        # can only close it. Catch it here, before that path, with a message
        # that tells the submitter what to do instead of a raw GitError.
        try:
            git_ops.git(["cat-file", "-e", f"{spec.commit}^{{commit}}"],
                       cwd=checkout)
        except git_ops.GitError:
            raise CallerError(
                f"commit {spec.commit} is not in {project.remote}; "
                "push it first") from None
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
        return finished

    def _settle(self, active: Active, result: JobResult) -> None:
        spec = active.running.spec
        # Free the card before the git work: artifact commits can take a
        # while and nothing about them needs the GPU.
        self._release_card(active)

        spec.pid = None
        spec.exit_code = result.exit_code
        ok = result.exit_code == 0 and not result.timed_out
        # Filing is deferred until after queue.finish (below), even though
        # the exception is caught right here. `_report_bug` can block for
        # ~250s on `gh`, and by this point the job is already out of
        # `self.active` -- a crash in that window would leave a finished
        # job sitting in running/ with a stale pid, and the reaper would
        # requeue and re-run a job that already completed. Matches the
        # ordering already used at both `_launch` call sites.
        artifact_exc = None
        if ok:
            try:
                self._collect_artifacts(spec, active.project, active.workdir)
            except Exception as e:
                artifact_exc = e
                ok = False
                spec.error = str(e)
        if not ok and spec.error is None:
            spec.error = self._describe_failure(spec, result)

        self._remove_worktree(active)
        self.queue.finish(spec, ok=ok)
        if artifact_exc is not None:
            self._report_bug(artifact_exc, "artifacts", spec)
        log.info("%s %s", spec.id, "done" if ok else f"failed: {spec.error}")

    def _collect_artifacts(self, spec: JobSpec, project: ProjectConfig,
                           workdir: Path) -> None:
        if not spec.artifacts:
            return
        srcs, rels = [], []
        for rel in spec.artifacts:
            src = workdir / rel
            if not src.exists():
                # Not a gpuq bug: the job was asked for this file and did
                # not produce it. Typed so the classifier does not read the
                # gpuq traceback around it as gpuq's fault.
                raise CallerError(f"declared artifact not produced: {rel}")
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
