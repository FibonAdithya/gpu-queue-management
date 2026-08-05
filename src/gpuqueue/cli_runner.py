"""gpuq-runner: the supervisor-managed daemon."""
from __future__ import annotations

import argparse
import logging
import signal
import sys

from .config import load_config, default_config_path, ConfigError
from .runner import Runner


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="gpuq-runner")
    p.add_argument("--config", default=None)
    p.add_argument("--once", action="store_true",
                   help="run a single tick and exit (for debugging)")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(message)s")
    try:
        cfg = load_config(args.config or default_config_path())
    except ConfigError as e:
        print(f"gpuq-runner: {e}", file=sys.stderr)
        return 2

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
