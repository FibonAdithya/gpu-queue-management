"""gpuq: submit, list, inspect and cancel jobs."""
from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from . import bugfiler
from . import killlog
from .claim import job_orphaned
from .config import ConfigError, default_config_path, load_config
from .queue import QueueRoot, STATES
from .spec import JobSpec, SpecError

DEFAULT_QUEUE_ROOT = "/workspace/queue"
WAIT_TIMED_OUT = 124  # as timeout(1), so a shell can tell the cases apart


def generate_id(prefix: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{stamp}-{secrets.token_hex(3)}"


def _queue(args) -> QueueRoot:
    root = args.queue_root or os.environ.get("QUEUE_ROOT") or DEFAULT_QUEUE_ROOT
    q = QueueRoot(Path(root))
    q.ensure_dirs()
    return q


def _cmd_submit(args) -> int:
    q = _queue(args)
    spec = JobSpec(
        id=args.id or generate_id(args.project),
        lane=args.lane,
        project=args.project,
        commit=args.commit,
        branch=args.branch,
        cmd=args.cmd,
        artifacts=args.artifact,
        timeout_s=args.timeout_s,
        dedupe_key=args.dedupe_key,
        vram_mb=args.vram_mb,
    )
    try:
        job_id = q.submit(spec)
    except SpecError as e:
        print(f"gpuq: {e}", file=sys.stderr)
        return 2
    except FileExistsError as e:
        print(f"gpuq: {e}", file=sys.stderr)
        return 2
    print(job_id)
    if getattr(args, "wait", False):
        return wait_for(q, job_id, args.timeout, args.poll)
    return 0


def wait_for(q: QueueRoot, job_id: str, timeout: float | None,
             poll: float) -> int:
    """Block until the job is done or failed. Returns the exit code.

    Returns immediately for a job that has already finished — which is what
    lets a caller submit, go and do something else, and wait whenever it
    suits. Waiting late costs nothing.
    """
    deadline = None if timeout is None else time.monotonic() + timeout
    seen = False
    while True:
        found = q.find(job_id)
        if found is None:
            if not seen:
                print(f"gpuq: no such job: {job_id}", file=sys.stderr)
                return 2
            # Seen before, missing now: the reaper is moving it between
            # directories. Not a missing job.
        else:
            seen = True
            state, spec = found
            if state in ("done", "failed"):
                if state == "done":
                    print(f"done {job_id}")
                    return 0
                print(f"failed {job_id}: {spec.error or 'no error recorded'}",
                      file=sys.stderr)
                return 1
        if deadline is not None and time.monotonic() >= deadline:
            state = found[0] if found else "unknown"
            print(f"gpuq: {job_id} still {state} after {timeout}s; "
                  "the job is untouched, wait again when you like",
                  file=sys.stderr)
            return WAIT_TIMED_OUT
        time.sleep(poll)


def _cmd_wait(args) -> int:
    return wait_for(_queue(args), args.id, args.timeout, args.poll)


def _orphaned(state: str, spec: JobSpec) -> bool:
    """A running job whose runner has died. Nothing enforces its timeout and
    nothing will collect its result until the process exits on its own."""
    return state == "running" and job_orphaned(spec.pid, spec.runner_pid)


def _cmd_list(args) -> int:
    q = _queue(args)
    states = [args.state] if args.state else list(STATES)
    rows = [{"state": s, "orphaned": _orphaned(s, spec), **spec.to_dict()}
            for s in states for spec in q.list_state(s)]
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        for r in rows:
            flag = "  ORPHANED (runner gone; nothing is supervising it)" \
                if r["orphaned"] else ""
            print(f"{r['state']:<8} {r['lane']:<3} {r['id']:<40} "
                  f"{r['project']}{flag}")
    return 0


def _cmd_show(args) -> int:
    q = _queue(args)
    found = q.find(args.id)
    if not found:
        print(f"gpuq: no such job: {args.id}", file=sys.stderr)
        return 1
    state, spec = found
    out, err = q.log_paths(args.id)
    print(json.dumps({"state": state, "orphaned": _orphaned(state, spec),
                      **spec.to_dict(),
                      "stdout_log": str(out), "stderr_log": str(err)}, indent=2))
    return 0


def _cmd_cancel(args) -> int:
    q = _queue(args)
    if q.cancel(args.id):
        print(f"cancelled {args.id}")
        return 0
    print(f"gpuq: cannot cancel {args.id} (not pending)", file=sys.stderr)
    return 1


def _cmd_bug(args) -> int:
    """File a bug an agent noticed. Everything the runner cannot see:
    `gpuq submit` / `wait` / `show` failures, config errors, CLI gaps.

    This path carries free text and unreliable blame, so it dispatches
    nothing. The owner adds `fix-me`, and that same act is what deduplicates
    it -- there is no traceback here to sign.
    """
    try:
        cfg = load_config(args.config or default_config_path())
    except ConfigError as e:
        print(f"gpuq: {e}", file=sys.stderr)
        return 2
    if not cfg.autofix.enabled or not cfg.autofix.repo:
        print("gpuq: autofix is not configured on this box; "
              "set [autofix].enabled and [autofix].repo", file=sys.stderr)
        return 2

    body = args.body if args.body is not None else sys.stdin.read()
    if not body.strip():
        print("gpuq: a body is required — say what happened and what you "
              "expected", file=sys.stderr)
        return 2
    try:
        number = bugfiler.file_agent_report(cfg.autofix, args.title, body)
    except bugfiler.GhError as e:
        print(f"gpuq: could not file the report: {e}", file=sys.stderr)
        return 1
    print(f"filed #{number}" if number else "filed")
    return 0


def _cmd_kills(args) -> int:
    """What the orphan sweep killed, most recent last.

    Exists because #24's victim had no way to tell a queue kill from its
    own crash: SIGKILL writes no stderr, so the caller sees `exit -9` and
    an empty message. An agent that sees a signal death runs this.
    """
    q = _queue(args)
    # One read: the runner appends to this file concurrently and swaps
    # it atomically, so a second separate read for `total` could straddle
    # that swap and describe a truncation window that never existed.
    entries, total = killlog.read_with_total(q.root, limit=args.limit)
    if not entries:
        # "no kills recorded" is a fact about the queue, not about the
        # flag. Printed for a `--limit` that selects nothing from a queue
        # that HAS kills, it tells an agent chasing a signal death that
        # the queue killed nothing -- the same wrong answer, out of the
        # same file, that this subcommand exists to prevent. `total` comes
        # from the same snapshot as `entries`, so the two cannot disagree.
        if total:
            print(f"no kills shown: --limit {args.limit} selects none of "
                  f"the {total} recorded -- pass a positive --limit")
        else:
            print("no kills recorded")
        return 0
    if total > len(entries):
        # Say so rather than truncating quietly. An operator who sees
        # four kills and had five is chasing the wrong window.
        print(f"showing the most recent {len(entries)} of {total} "
              f"-- pass --limit {total} for all")
    for e in entries:
        # `is None`, not `or`: a measured 0 MiB is a real reading -- a
        # process that had just started, or one nvidia-smi sampled between
        # allocations -- and `?` claims the record is incomplete when it
        # is not.
        used = e.get("used_mb")
        print(f"{e.get('ts', '?')}  pid {e.get('pid')}  "
              f"{'?' if used is None else used} MiB  {e.get('name') or '?'}")
        print(f"    cgroup:  {e.get('cgroup') or '(none read)'}")
        print(f"    reason:  {e.get('reason')}")
        print(f"    ledgers: "
              f"{', '.join(e.get('ledgers_consulted') or ['(none)'])}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gpuq")
    p.add_argument("--queue-root", default=None)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("submit", help="queue a job")
    s.add_argument("--project", required=True)
    s.add_argument("--commit", required=True,
                   help="exact commit; a branch alone is not reproducible")
    s.add_argument("--branch", required=True)
    # No choices= here: one lane error message, from spec.validate().
    s.add_argument("--lane", default="cpu")
    s.add_argument("--id", default=None)
    s.add_argument("--artifact", action="append", default=[])
    s.add_argument("--timeout-s", dest="timeout_s", type=int, default=3600)
    s.add_argument("--dedupe-key", dest="dedupe_key", default=None)
    s.add_argument("--vram-mb", dest="vram_mb", type=int, default=None,
                   help="VRAM this job needs, in MiB as nvidia-smi reports "
                        "it (so including the ~250 MiB CUDA context and the "
                        "allocator's high-water mark, not torch's "
                        "max_memory_allocated). Omit to take the whole card.")
    s.add_argument("--wait", action="store_true",
                   help="block until the job finishes, then exit with its result")
    s.add_argument("--timeout", type=float, default=None,
                   help="how long to wait, not how long the job may run")
    s.add_argument("--poll", type=float, default=2.0)
    s.add_argument("cmd", nargs=argparse.REMAINDER)
    s.set_defaults(func=_cmd_submit)

    w = sub.add_parser("wait", help="block until a job finishes")
    w.add_argument("id")
    w.add_argument("--timeout", type=float, default=None,
                   help="how long YOU wait. The job is never cancelled by "
                        "this; its own limit is timeout_s in the spec.")
    w.add_argument("--poll", type=float, default=2.0)
    w.set_defaults(func=_cmd_wait)

    l = sub.add_parser("list")
    l.add_argument("--state", choices=list(STATES), default=None)
    l.add_argument("--json", action="store_true")
    l.set_defaults(func=_cmd_list)

    sh = sub.add_parser("show")
    sh.add_argument("id")
    sh.set_defaults(func=_cmd_show)

    c = sub.add_parser("cancel")
    c.add_argument("id")
    c.set_defaults(func=_cmd_cancel)

    b = sub.add_parser("bug", help="report a gpuq bug the runner cannot see")
    b.add_argument("title")
    b.add_argument("--body", default=None,
                   help="the report; read from stdin when omitted")
    b.add_argument("--config", default=None)
    b.set_defaults(func=_cmd_bug)

    k = sub.add_parser("kills",
                       help="what the orphan sweep killed and why")
    k.add_argument("--limit", type=int, default=20)
    k.set_defaults(func=_cmd_kills)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "cmd", None) is not None:
        if args.cmd and args.cmd[0] == "--":
            args.cmd = args.cmd[1:]
        if not args.cmd:
            print("gpuq: a command is required after --", file=sys.stderr)
            return 2
    return args.func(args)


if __name__ == "__main__":  # python -m gpuqueue.cli_gpuq
    raise SystemExit(main())
