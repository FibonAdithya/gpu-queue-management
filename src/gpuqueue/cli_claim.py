"""gpu-claim: hold the advisory lock for the duration of a command."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

from .claim import gpu_claim, ClaimBusy, release_stale, list_claims
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
    """
    env = dict(os.environ)
    if "CUDA_VISIBLE_DEVICES" in env:
        return env
    value = cuda_visible_value(gpu_index)
    if value is not None:
        env["CUDA_VISIBLE_DEVICES"] = value
    return env


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.status:
        print(json.dumps([body for _, body in list_claims()], indent=2))
        return 0
    if args.reap:
        for body in release_stale():
            print(f"released stale claim: pid {body.get('pid')} "
                  f"{body.get('owner')}", file=sys.stderr)
        return 0

    cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd
    if not cmd:
        print("gpu-claim: a command is required after --", file=sys.stderr)
        return 2

    try:
        key = gpu_key(args.gpu_index)
    except GpuIdError as e:
        print(f"gpu-claim: {e}", file=sys.stderr)
        return EX_UNAVAILABLE

    if not args.no_preflight:
        try:
            preflight()
        except PreflightFailed as e:
            print(f"gpu-claim: {e}", file=sys.stderr)
            return EX_UNAVAILABLE

    try:
        with gpu_claim(key=key, owner=args.owner, cmd=cmd, wait=args.wait):
            return subprocess.run(cmd, env=_child_env(args.gpu_index)).returncode
    except ClaimBusy as e:
        print(f"gpu-claim: {e}", file=sys.stderr)
        return EX_TEMPFAIL


if __name__ == "__main__":  # python -m gpuqueue.cli_claim
    raise SystemExit(main())
