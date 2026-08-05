"""gpuq: submit, list, inspect and cancel jobs."""
from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
from datetime import datetime, timezone
from pathlib import Path

from .queue import QueueRoot, STATES
from .spec import JobSpec, SpecError

DEFAULT_QUEUE_ROOT = "/workspace/queue"


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
    return 0


def _cmd_list(args) -> int:
    q = _queue(args)
    states = [args.state] if args.state else list(STATES)
    rows = [{"state": s, **spec.to_dict()}
            for s in states for spec in q.list_state(s)]
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        for r in rows:
            print(f"{r['state']:<8} {r['lane']:<3} {r['id']:<40} {r['project']}")
    return 0


def _cmd_show(args) -> int:
    q = _queue(args)
    found = q.find(args.id)
    if not found:
        print(f"gpuq: no such job: {args.id}", file=sys.stderr)
        return 1
    state, spec = found
    out, err = q.log_paths(args.id)
    print(json.dumps({"state": state, **spec.to_dict(),
                      "stdout_log": str(out), "stderr_log": str(err)}, indent=2))
    return 0


def _cmd_cancel(args) -> int:
    q = _queue(args)
    if q.cancel(args.id):
        print(f"cancelled {args.id}")
        return 0
    print(f"gpuq: cannot cancel {args.id} (not pending)", file=sys.stderr)
    return 1


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
    s.add_argument("cmd", nargs=argparse.REMAINDER)
    s.set_defaults(func=_cmd_submit)

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
