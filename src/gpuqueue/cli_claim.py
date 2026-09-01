"""gpu-claim: hold the advisory lock for the duration of a command."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from . import claim as _claim, config
# `DEFAULT_CLAIM_DIR` is deliberately reached through the module rather
# than imported by name: `claim.claim_dir()` reads it at call time, so a
# name bound here at import would drift from the value actually in use --
# and drift between two readings of one claim directory is the whole of
# issue #19.
from .claim import (gpu_claim, ClaimBusy, CannotEverFit, sweep_stale,
                    list_claims, default_usable_mb, claim_dir,
                    all_claim_dirs)
from . import cgroups
from .claim import pid_alive
from .gpuid import gpu_key, cuda_visible_value, GpuIdError
from .preflight import preflight, PreflightFailed

EX_UNAVAILABLE = 69
EX_TEMPFAIL = 75


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gpu-claim",
        description="Run a command holding the advisory GPU lock.")
    p.add_argument("--wait", action="store_true",
                   help="block until the card is free instead of failing")
    p.add_argument("--no-preflight", action="store_true")
    p.add_argument("--owner", default=None)
    p.add_argument("--gpu-index", type=int, default=0)
    p.add_argument("--status", action="store_true", help="print live claims")
    p.add_argument("--reap", action="store_true", help="release dead claims")
    p.add_argument("--vram-mb", dest="vram_mb", type=int, default=None,
                   help="VRAM this command needs, in MiB as nvidia-smi "
                        "reports it (so including the ~250 MiB CUDA "
                        "context and the allocator's high-water mark, "
                        "not torch's max_memory_allocated). Omit to "
                        "take the whole card.")
    p.add_argument("--scope-pid", dest="scope_pid", type=int, default=None,
                   help="claim on behalf of the cgroup this pid belongs "
                        "to, for CUDA that runs in a container rather "
                        "than in this command's own process tree. Name "
                        "any pid inside the target, e.g. --scope-pid "
                        "$(docker inspect -f '{{.State.Pid}}' <container>).")
    p.add_argument("cmd", nargs=argparse.REMAINDER)
    return p


def _child_env(gpu_index: int) -> dict:
    """The command's environment, pinned to the card we just claimed.

    Holding the lock is what says the card is yours; the pin is what lets the
    command act on that without guessing. It matters for consumers that refuse
    to guess -- a trainer resolving `device: auto` under a strict policy has no
    way to know which card it was given unless something says so.

    An existing setting wins. The runner's jobs can override the pin from
    inside their own command, but gpu-claim's caller can only express intent
    through the environment they invoke us with, so treat that as deliberate.
    Unset when no uuid is available, which leaves behaviour exactly as it was.

    Two qualifications on "an existing setting wins":

    An *empty* setting does not count as one. `export CUDA_VISIBLE_DEVICES=
    ${SOMETHING}` with SOMETHING unset leaves the variable present and empty,
    which the driver reads as "no cards at all" -- a child that holds the card
    and cannot use it. That is a broken wrapper, not an intent.

    A setting naming a *different* card is still honoured, but not silently.
    The lock is then held on one card while the command runs on another, which
    is precisely the collision this pin exists to prevent; two claimants with
    different --gpu-index can both land on card 0 while holding distinct locks.
    We cannot tell that from a deliberate override, so we say so and continue.
    """
    env = dict(os.environ)
    value = cuda_visible_value(gpu_index)
    caller = env.get("CUDA_VISIBLE_DEVICES", "")
    if caller.strip():
        if value is not None and caller.strip() != value:
            print(f"gpu-claim: warning: CUDA_VISIBLE_DEVICES={caller} was "
                  f"already set and names a different card than the one "
                  f"claimed ({value}); honouring your setting, so the lock "
                  f"and the run may disagree", file=sys.stderr)
        return env
    if value is not None:
        env["CUDA_VISIBLE_DEVICES"] = value
    else:
        env.pop("CUDA_VISIBLE_DEVICES", None)
    return env


def _warn_if_the_runner_reads_elsewhere() -> None:
    """Say so when this claim is going somewhere the daemon is not looking.

    `cli_runner` already warns from the daemon's side when
    `[queue].claim_dir` and its own `$GPU_CLAIM_DIR` disagree. This is the
    other side of the same card, and it is the side that goes wrong
    silently: the daemon gets `GPU_CLAIM_DIR` from a supervisor unit and an
    interactive shell never inherits one, so the divergence needs no flag
    and no hand-edited config to appear. It cost 48% of one session's runs
    (issue #19), and the operator's own `--status` confirmed the claim as
    healthy throughout, because it reads the same directory the claim went
    to.

    Emitted before `--status` and `--reap` for exactly that reason.

    Two consequences, and the second is conditional. `preflight.own_pids`
    exempts a claim under the reaper's `$GPU_CLAIM_DIR` *or* the default,
    so landing on the default is covered even while diverging -- claiming
    a SIGKILL there would send someone chasing a kill that cannot happen.
    What survives in every case is that the runner's ledger does not count
    this claim at all, so `gpu_max_jobs` and the VRAM accounting will admit
    a job on top of it.

    A configured directory is the only thing worth comparing against. With
    the key unset the daemon reads its own environment, which this process
    cannot see; warning on that guess would fire on every correctly
    configured box.
    """
    theirs = config.claim_dir_setting()
    if theirs is None:
        return
    ours = claim_dir()
    if _same_dir(ours, theirs):
        return
    msg = (f"gpu-claim: warning: this claim goes in {ours}, but the runner "
           f"configured in {config.default_config_path()} reads {theirs}. "
           f"Nothing there counts this claim against the card, so the "
           f"runner can admit a job on top of it")
    default = Path(_claim.DEFAULT_CLAIM_DIR)
    if not _same_dir(ours, default):
        # Neither the runner's directory nor the default, so no exemption
        # the reaper can build covers this run.
        msg += (f"; and the orphan sweep exempts only the claim directories "
                f"the reaper's own process can see -- its $GPU_CLAIM_DIR "
                f"and {default} -- so it will SIGTERM this run and then "
                f"SIGKILL whatever survives the grace")
    print(f"{msg}. Export GPU_CLAIM_DIR={theirs} before claiming.",
          file=sys.stderr)


def _same_dir(a: Path, b: Path) -> bool:
    try:
        return a.resolve() == b.resolve()
    except OSError:
        return a == b


def _resolve_scope(scope_pid: int) -> str | None:
    """The cgroup `--scope-pid` names, or None after reporting why not.

    Resolution and refusal happen here, at claim time, rather than in the
    reaper an hour later: an over-broad scope does not fail, it silently
    disables orphan protection for the card, and the operator's only
    other signal would be a SIGKILL that never comes.
    """
    if scope_pid <= 0:
        print(f"gpu-claim: --scope-pid must be a pid, got {scope_pid}",
              file=sys.stderr)
        return None
    scope = cgroups.cgroup_of(scope_pid)
    if scope is None:
        # Three causes with three different next moves, and `cgroup_of`
        # collapses all of them to None on purpose -- it must never guess
        # a path. A pid that is gone is a typo or a race the operator
        # retries; an entry that could not be *read* is a `hidepid` mount
        # or another user's process, and `pid_alive` answers True on
        # EPERM so that case reaches here with /proc perfectly healthy;
        # only an entry we read with no `0::` line in it is a v1 box.
        # Reporting the middle one as the last named a kernel feature
        # that is working and sent the operator nowhere useful.
        if not pid_alive(scope_pid):
            print(f"gpu-claim: --scope-pid {scope_pid} is not a running "
                  f"process", file=sys.stderr)
            return None
        why = cgroups.read_error(scope_pid)
        if why is not None:
            print(f"gpu-claim: cannot read /proc/{scope_pid}/cgroup: "
                  f"{why}. A `hidepid` mount or a pid owned by another "
                  f"user does this; run gpu-claim as that process's owner",
                  file=sys.stderr)
        else:
            print(f"gpu-claim: pid {scope_pid} has no unified cgroup path; "
                  f"--scope-pid needs cgroup v2 and this box is not "
                  f"running it", file=sys.stderr)
        return None
    reason = cgroups.refuse_reason(scope)
    if reason is not None:
        print(f"gpu-claim: --scope-pid {scope_pid}: {reason}",
              file=sys.stderr)
        return None
    n = cgroups.scope_process_count(scope)
    where = "" if n is None else \
        f" ({n} live process{'' if n == 1 else 'es'})"
    print(f"gpu-claim: scope {scope}{where}", file=sys.stderr)
    return scope


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _warn_if_the_runner_reads_elsewhere()

    if args.status:
        print(json.dumps([body for _, body in list_claims()], indent=2))
        return 0
    if args.reap:
        # Every directory a claim could be in, not just this shell's. The
        # operator running `--reap` is chasing a card that looks held by
        # nothing, and the record doing the holding is as likely to be
        # under the default as under their own `$GPU_CLAIM_DIR` -- an
        # interactive shell and a supervisor unit systematically disagree
        # about that variable (issue #19), and the daemon's sweep now
        # covers both for the same reason (issue #21).
        #
        # `[queue].claim_dir` is the third, and on the deployed box the
        # one most records are under: the daemon reads it, and this
        # process's environment cannot name it. It is already in hand --
        # the warning above compares against it on every invocation --
        # and it is the same argument `reaper._swept_dirs` passes, so the
        # operator's sweep and the daemon's cover one set. Left out,
        # `--reap` opened neither the directory holding the card nor said
        # so, and printing nothing reads as "nothing was stale".
        released, stuck = sweep_stale(
            all_claim_dirs(config.claim_dir_setting()))
        for body in released:
            print(f"released stale claim: pid {body.get('pid')} "
                  f"{body.get('owner')}", file=sys.stderr)
        for body in stuck:
            # Named rather than swallowed: the card is still held by this
            # record, and reporting only what was freed would say it is
            # not.
            # The kernel's reason, not a guess at it. EACCES on another
            # user's record in a world-writable directory is the expected
            # one, but a read-only remount and a stale NFS handle reach
            # this same branch, and naming a cause the kernel did not give
            # sends an operator mid-incident to find an owner who is not
            # the problem.
            print(f"could not remove stale claim {body.get('path')}: "
                  f"pid {body.get('pid')} {body.get('owner')} -- "
                  f"{body.get('error')}", file=sys.stderr)
        return 0

    cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd
    if not cmd:
        print("gpu-claim: a command is required after --", file=sys.stderr)
        return 2
    if args.vram_mb is not None and args.vram_mb <= 0:
        # 2, not 69/75: nothing about the card is wrong, the command line
        # is. A typo'd minus sign here is not a claim that fails to be
        # admitted -- `ledger.fits` sums declarations, so a negative one
        # subtracts from the accounted total and lets the *next* claimant
        # be admitted past the end of the card.
        print(f"gpu-claim: --vram-mb must be a positive number of MiB, got "
              f"{args.vram_mb}; omit it to take the whole card",
              file=sys.stderr)
        return 2

    scope_cgroup = None
    if args.scope_pid is not None:
        scope_cgroup = _resolve_scope(args.scope_pid)
        if scope_cgroup is None:
            return 2

    try:
        key = gpu_key(args.gpu_index)
    except GpuIdError as e:
        print(f"gpu-claim: {e}", file=sys.stderr)
        return EX_UNAVAILABLE

    if not args.no_preflight:
        try:
            preflight(scope=scope_cgroup)
        except PreflightFailed as e:
            print(f"gpu-claim: {e}", file=sys.stderr)
            return EX_UNAVAILABLE

    try:
        # Sized here rather than left to gpu_claim's default, which asks
        # about card 0. We are the only one who knows which card --gpu-index
        # named, and it is the card the key and the pin already point at.
        with gpu_claim(key=key, owner=args.owner, cmd=cmd, wait=args.wait,
                       vram_mb=args.vram_mb,
                       usable_mb=default_usable_mb(args.gpu_index),
                       scope_pid=args.scope_pid,
                       scope_cgroup=scope_cgroup):
            return subprocess.run(cmd, env=_child_env(args.gpu_index)).returncode
    except CannotEverFit as e:
        # Ahead of ClaimBusy, which it subclasses: 75 means "try again
        # later" and there is no later in which this declaration fits.
        print(f"gpu-claim: {e}", file=sys.stderr)
        return EX_UNAVAILABLE
    except ClaimBusy as e:
        print(f"gpu-claim: {e}", file=sys.stderr)
        return EX_TEMPFAIL


if __name__ == "__main__":  # python -m gpuqueue.cli_claim
    raise SystemExit(main())
