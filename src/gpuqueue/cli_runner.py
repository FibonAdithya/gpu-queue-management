"""gpuq-runner: the supervisor-managed daemon."""
from __future__ import annotations

import argparse
import logging
import signal
import sys
from pathlib import Path

from .config import load_config, default_config_path, ConfigError
from .runner import Runner


def _warn_if_gpu_claim_disagrees(path: Path) -> None:
    """`--config` moves the runner's policy without moving `gpu-claim`'s.

    The two share one `<key>.lock.d` and must size the card the same way,
    which is why `config.vram_policy` and `config.max_holders` exist. But
    those read `default_config_path()` -- `$GPUQ_CONFIG`, else
    `/workspace/gpuq.toml` -- because a standalone claim has no way to know
    what flags the daemon was started with. Point the runner somewhere else
    and the two admit against different totals into one ledger, which is
    the double-booking the ledger exists to prevent.

    A warning rather than a refusal: a box may deliberately run the daemon
    off a path no interactive `gpu-claim` will ever be run on, and failing
    to start would break a deployment that works today. Loud, because the
    symptom otherwise is an OOM hours later with nothing pointing here.
    """
    if path.resolve() == default_config_path().resolve():
        return
    logging.warning(
        "runner config is %s, but gpu-claim reads %s -- the two admit "
        "against the same card and will disagree about its capacity and "
        "its job limit. Export GPUQ_CONFIG=%s so both read one file.",
        path, default_config_path(), path)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="gpuq-runner")
    p.add_argument("--config", default=None)
    p.add_argument("--once", action="store_true",
                   help="run a single tick and exit (for debugging)")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(message)s")
    path = Path(args.config) if args.config else default_config_path()
    try:
        cfg = load_config(path)
    except ConfigError as e:
        print(f"gpuq-runner: {e}", file=sys.stderr)
        return 2
    _warn_if_gpu_claim_disagrees(path)

    runner = Runner(cfg)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: runner.stop())

    logging.info("runner started: queue=%s cpu_slots=%d projects=%s",
                 cfg.queue_root, cfg.cpu_slots, ", ".join(cfg.projects) or "none")
    if args.once:
        runner.tick()
        return 0
    runner.run_forever()
    logging.info("runner stopped")
    return 0


if __name__ == "__main__":  # python -m gpuqueue.cli_runner
    raise SystemExit(main())
