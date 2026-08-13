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
from . import ledger
from .bugreport import CallerError, signature
from .claim import gpu_claim, ClaimBusy
from .config import RunnerConfig, ProjectConfig
from .executor import (start_job, poll_job, kill_job, JobResult, RunningJob,
                       StartFailed)
from .gpuid import gpu_key, cuda_visible_value, total_vram_mb, GpuIdError
from .preflight import preflight, PreflightFailed
from .queue import QueueRoot, STATES
from .reaper import reap, MAX_ATTEMPTS
from .spec import JobSpec

log = logging.getLogger("gpuqueue.runner")


@dataclass
class Active:
    running: RunningJob
    project: ProjectConfig
    workdir: Path
    claim_cm: object | None = None       # entered gpu_claim, released on settle
    claim_record: object | None = None   # its ledger.Record, for usage_pid
    started_mono: float = 0.0            # for the watchdog's victim retry


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
        self._usable_mb_cache: int | None = None
        self._usable_asked = False
        # The card's identity, cached for the same reason its total is: a
        # second nvidia-smi subprocess for an answer that cannot change
        # without a reboot. See _card_key_cached.
        self._card_key: str | None = None
        # Logged once, not once per admit: see _usable_mb.
        self._usable_query_warned = False
        # str(record path) -> consecutive sweeps over its declaration. The
        # full path, never the bare filename: `reaper.check_vram` explains
        # why one is not unique across `<key>.lock.d` directories.
        self._vram_strikes: dict[str, int] = {}
        # job id -> the conviction that killed it, so _describe_failure can
        # say "declared 512 MiB, using 3070" instead of "exit -9"
        self._convicted: dict[str, dict] = {}
        self._last_conviction: float | None = None

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
        self._phase("collect", self.collect)
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
        result = reap(self.queue, self.cfg, active_ids=set(self.active),
                      include_orphan_cuda=sweep,
                      vram_strikes=self._vram_strikes)
        for c in result.get("convicted", []):
            self._last_conviction = time.monotonic()
            log.warning("killed %s: declared %s MiB, using %s MiB",
                        c["owner"], c["declared"], c["used"])
            if c["owner"].startswith("gpuq:"):
                self._convicted[c["owner"][len("gpuq:"):]] = c
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
    def _usable_mb(self) -> int | None:
        """Cached, but only once there is an answer worth caching.

        A card's total does not change, so a query that succeeds is
        cached -- otherwise this is an nvidia-smi subprocess on every
        admit, on the loop that also polls every running job. A query
        that *fails* is not cached: a subprocess hiccup, a timeout while
        the box is under load, or brief driver contention is not exotic
        on a box whose whole purpose is GPU jobs, and latching that as
        "unqueryable forever" would silently degrade the GPU lane to
        exclusive-only admission for the rest of the daemon's life, with
        no way back short of a restart. Returning None uncached here
        means the next admit tries again.
        """
        if self._usable_asked:
            return self._usable_mb_cache
        total = self.cfg.gpu_vram_mb
        if total is None:
            total = total_vram_mb()
            if total is None:
                if not self._usable_query_warned:
                    log.warning("could not query the card's VRAM total; "
                               "GPU lane admits exclusive-only until this "
                               "succeeds")
                    self._usable_query_warned = True
                return None
        usable = total - self.cfg.gpu_vram_reserve_mb
        # config.load_config rejects a reserve >= gpu_vram_mb, but only on
        # that explicit path -- it has no card to check against. On the
        # default "ask the card" path a card that genuinely reports less
        # than the reserve is possible, and a negative number here would
        # flow straight into ledger.fits/exceeds_capacity, where `want_mb
        # > usable_mb` is always true and nothing is ever admitted --
        # silently, with no error to explain why. This is a real, static
        # answer (the query above succeeded), not a transient failure, so
        # it is cached like any other successful query rather than
        # re-querying every admit for the same result.
        if usable <= 0:
            log.warning("gpu_vram_reserve_mb (%s) leaves no usable VRAM "
                       "out of %s MiB reported by the card; GPU lane "
                       "admits exclusive-only", self.cfg.gpu_vram_reserve_mb,
                       total)
            usable = None
        self._usable_asked = True
        self._usable_mb_cache = usable
        return usable

    def _capacity(self, lane: str) -> int:
        limit = (self.cfg.cpu_slots if lane == "cpu"
                 else self.cfg.gpu_max_jobs)
        in_lane = sum(1 for a in self.active.values()
                      if a.running.spec.lane == lane)
        return limit - in_lane

    def admit(self) -> list[str]:
        if self._stopping:
            return []
        pending = sorted(self.queue.list_state("pending"),
                         key=lambda s: (s.submitted_at, s.id))
        started = []
        # A MutexTimeout cannot resolve inside one pass: the thing holding
        # the ledger mutex that long is a pre-ledger `gpu-claim`, which
        # holds it for its whole training run. Paying `MUTEX_WAIT_S` again
        # for the next pending GPU job buys nothing and costs 10s each --
        # five queued jobs stall `admit` for the best part of a minute, and
        # `collect` does not run in that window, so a hung job outlives its
        # `timeout_s`. That is the stall `_report_bug` calls worse than a
        # crash. One timeout ends this pass's GPU admissions; the next tick
        # tries again. Not fixed by lowering `ledger.MUTEX_WAIT_S`, which
        # the interactive `gpu-claim` path legitimately needs.
        mutex_blocked = False
        # Asked once, before the loop, because the answer is the same for
        # every pending GPU job and costs two nvidia-smi subprocesses plus
        # a recursive `ps` walk per ledger record to get. See `_ready_card`.
        card_key = card_error = None
        if any(s.lane == "gpu" for s in pending) and self._capacity("gpu") > 0:
            card_key, card_error = self._ready_card()
            if isinstance(card_error, PreflightFailed):
                # One line per pass, not one per pending job: the same
                # reasoning `mutex_blocked` applies below to a card-wide
                # condition nobody in this pass can get past.
                log.warning("GPU admissions deferred this pass: %s", card_error)
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
            claim_cm = claim_record = None
            if spec.lane == "gpu":
                if mutex_blocked or isinstance(card_error, PreflightFailed):
                    continue
                if card_error is not None:      # GpuIdError
                    # Card-wide like the above, but permanent: no card will
                    # appear on a box that has none, so deferring the pass
                    # would queue these forever.
                    self._fail_pending(spec, f"no usable GPU: {card_error}")
                    continue
                try:
                    taken = self._take_card(spec, card_key)
                except ledger.MutexTimeout as e:
                    # Logged here rather than per job, so one wedged holder
                    # writes one line per pass instead of one per pending
                    # job.
                    mutex_blocked = True
                    log.warning("GPU admissions deferred this pass: %s", e)
                    continue
                if taken is None:
                    continue
                claim_cm, claim_record = taken

            claimed = self.queue.claim(spec.id)
            if claimed is None:  # cancelled, or another process won the rename
                self._exit_claim(claim_cm)
                continue

            if self._launch(claimed, project, claim_cm, claim_record):
                started.append(claimed.id)
        return started

    def _ready_card(self) -> tuple[str | None, Exception | None]:
        """Is the card usable, and which card is it -- asked once per pass.

        Both halves answer the same for every pending GPU job, and both are
        expensive: `preflight` is an nvidia-smi subprocess with a 15s
        timeout plus a recursive `ps` walk per ledger record, and `gpu_key`
        is a second nvidia-smi. Per job that used to cost nothing, because
        `_capacity("gpu")` returned 0 the moment one GPU job ran and the
        loop skipped every other pending GPU job without asking. A lane
        that admits `gpu_max_jobs` still has capacity while the card is
        VRAM-full, so all of them now reach `_take_card`: 20 queued jobs at
        `poll_interval_s = 2.0` is ~40 nvidia-smi invocations every two
        seconds, and `collect` cannot run while `admit` does, so a hung job
        outlives its `timeout_s`. That is the stall `_report_bug` calls
        worse than a crash.

        Returns `(key, None)` when the card can be handed out, else
        `(None, error)`: `PreflightFailed` to defer the pass,
        `GpuIdError` to fail the jobs rather than queue them forever.

        What is deliberately *not* hoisted is the per-job half -- whether
        this declaration fits in the room that is left. Short-circuiting
        the pass on a full card would mean a 500 MiB job never slots in
        beside the 6 GB one running, because a 7 GB job queued ahead of it
        did not fit, which is the head-of-line blocking that per-job VRAM
        accounting exists to remove.
        """
        try:
            # The runner's configured claim dir, not $GPU_CLAIM_DIR. These
            # were already two different answers to "where are the
            # claims?"; now that preflight decides contention by reading
            # them, disagreeing means a co-tenant reads as an intruder.
            preflight(directory=self.cfg.claim_dir)
        except PreflightFailed as e:
            return None, e
        except Exception as e:
            self._report_bug(e, "preflight")
            raise
        try:
            return self._card_key_cached(), None
        except GpuIdError as e:
            return None, e

    def _card_key_cached(self) -> str:
        """`gpu_key()`, asked once per daemon rather than once per pass.

        Hoisting `_ready_card` out of the per-job loop bounded the cost per
        *job*, but introduced a steady-state one that did not exist before:
        a lane admitting `gpu_max_jobs` still has capacity while the card is
        VRAM-full, so a pending job that does not fit leaves `_capacity`
        positive and this whole function runs every `poll_interval_s`,
        indefinitely, on the thread that also enforces `timeout_s`.

        Only the key is cached. `preflight` above is the half that has to
        stay fresh -- it is measuring contention right now -- where the
        card's uuid cannot change without a reboot, which restarts this
        daemon anyway. Same trade `_usable_mb` makes for the card's total,
        and a failure is not cached for the same reason it is not there: a
        transient nvidia-smi hiccup must not latch this lane off for the
        life of the process.
        """
        if self._card_key is None:
            self._card_key = gpu_key()
        return self._card_key

    def _take_card(self, spec: JobSpec, key: str):
        """Enter a gpu_claim by hand so it can be held across ticks.

        `key` comes from `_ready_card`, which has already established that
        the card is usable; this is only the part that depends on `spec`.

        Never waits on *capacity*: `wait=True` in a single-threaded loop
        would stall the CPU lane behind whoever holds the card. A full card
        returns `None` and the job stays pending. It can still block for up
        to `ledger.MUTEX_WAIT_S` on the ledger mutex, which is a different
        thing -- a participant holds that only for a directory read and one
        rename, so the wait is short unless a pre-ledger `gpu-claim` is
        holding `flock` for its whole run. That case raises `MutexTimeout`,
        and `admit` stops trying for the rest of the pass.

        Returns `(claim_cm, record)` on success, or `None` when the job
        should stay pending or has just been failed outright. Raises
        `ledger.MutexTimeout`.
        """
        usable = self._usable_mb()
        if ledger.exceeds_capacity(spec.vram_mb, usable):
            # Permanent, so failing beats queueing forever -- the same call
            # this function already makes for a box with no GPU.
            self._fail_pending(
                spec, f"declared {spec.vram_mb} MiB but only {usable} MiB is "
                      "usable on this card; it can never be admitted")
            return None
        cm = gpu_claim(key=key, owner=f"gpuq:{spec.id}", cmd=spec.cmd,
                       wait=False, directory=self.cfg.claim_dir,
                       vram_mb=spec.vram_mb, usable_mb=usable,
                       own_usage=False,
                       # Passed rather than left to gpu_claim's default,
                       # which re-reads the config file: this runner has
                       # already loaded the key, and `_capacity` is only a
                       # cheap pre-filter -- the ledger is what enforces
                       # the cap against holders this process cannot see.
                       max_holders=self.cfg.gpu_max_jobs)
        try:
            record = cm.__enter__()
        except ledger.MutexTimeout:
            # Out to `admit`, which ends this pass's GPU admissions. Before
            # the general ClaimBusy clause below because MutexTimeout is a
            # subclass of it, and "the ledger is unreadable" is not "the
            # card is full".
            raise
        except ClaimBusy as e:
            log.info("%s waiting: %s", spec.id, e)
            return None
        return cm, record

    def _card_pin(self, spec: JobSpec) -> dict:
        """CUDA_VISIBLE_DEVICES for a gpu job, naming the card it was given.

        The claim already serializes the card. This makes the allocation
        binding on the job rather than advisory: a pinned process cannot see,
        and so cannot accidentally take, a card the queue did not hand it.

        It also makes the queue usable by consumers that refuse to guess a
        device. `resolve_device(..., strict=True)` in the wgan-synthetic
        project declines to resolve `device: auto` unless the process has been
        pinned, precisely so two agents cannot both silently land on cuda:0.
        Nothing was doing the pinning, so every `device: auto` config -- which
        is most of them -- failed to start under the queue.

        cpu-lane jobs are left alone. They never took the card, and handing
        one a pin would say they had.
        """
        if spec.lane != "gpu":
            return {}
        value = cuda_visible_value()
        if value is None:
            # Degraded, not broken: this is the no-uuid box the lock's
            # name-index fallback exists for. Jobs run exactly as they did
            # before pinning, so say so once rather than failing the job.
            log.warning("%s not pinned: no GPU uuid to pin to", spec.id)
            return {}
        return {"CUDA_VISIBLE_DEVICES": value}

    def _launch(self, spec: JobSpec, project: ProjectConfig, claim_cm,
               claim_record=None) -> bool:
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
        job_env = {"GPUQ_JOB_ID": spec.id,
                   "GPUQ_QUEUE_ROOT": str(self.queue.root)}
        job_env.update(self._card_pin(spec))
        try:
            running = start_job(
                spec, workdir, out_log, err_log, project=project,
                extra_env=job_env)
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
                                      workdir=workdir, claim_cm=claim_cm,
                                      claim_record=claim_record,
                                      started_mono=time.monotonic())
        if claim_record is not None:
            # The card was taken before this process existed. Charging the
            # record to it now is what lets the watchdog and the orphan
            # sweep tell this job's VRAM from a co-tenant's.
            ledger.set_usage_pid(claim_record, running.pid)
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
        if ok and spec.id in self._convicted:
            # A conviction outranks a clean exit. `_kill_tree` SIGTERMs the
            # holder's tree before it SIGKILLs it, so a trainer that
            # checkpoints on SIGTERM exits 0 -- and judged on the exit code
            # alone the job we just killed for over-using the card is filed
            # under done/. `_describe_failure` would never run, so the one
            # thing conviction exists to produce (the job dying while naming
            # its own declaration) is lost, and the `_convicted` entry leaks
            # to disqualify this id from the co-tenant retry for the life of
            # the runner.
            ok = False
        if not ok and self._hit_by_a_convicted_co_tenant(spec, active, result):
            self._remove_worktree(active)
            self.queue.requeue(spec)
            log.info("%s requeued: OOMed while a co-tenant was convicted of "
                     "exceeding its declaration", spec.id)
            return
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

    def _hit_by_a_convicted_co_tenant(self, spec: JobSpec, active: Active,
                                      result: JobResult) -> bool:
        """An OOM this job did not cause.

        `docs/design.md` says a CUDA OOM is a configuration error and is
        never retried blindly. That stays true, and sharing does not
        weaken it -- it only adds one case where the premise is false: the
        job OOMed while the watchdog convicted a *different* holder of
        over-using the card. That is a genuine transient, so it gets the
        single retry `attempts` already bounds, and every other OOM
        behaves exactly as before.

        Branch order mirrors `_describe_failure`: convicted, then
        timed_out, then oom -- so the two cannot drift apart. A job can
        print an OOM-looking line and then hang (e.g. in NCCL teardown);
        `result.timed_out` there is still true, and a hang is a bug, not
        a transient, no matter what a co-tenant did at the same moment.
        Checking timed_out before oom, same as _describe_failure does,
        keeps that job out of the retry path.
        """
        if spec.id in self._convicted:
            return False
        if result.timed_out:
            return False
        if not result.oom:
            return False
        if self._last_conviction is None:
            return False
        return (self._last_conviction > active.started_mono
                and spec.attempts < MAX_ATTEMPTS)

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
        guilty = self._convicted.pop(spec.id, None)
        if guilty:
            return (f"killed for exceeding its declaration: --vram-mb "
                    f"{guilty['declared']}, actually using {guilty['used']} "
                    "MiB. Declare what the job really needs, measured as "
                    "nvidia-smi reports it. Not retried.")
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
