"""gpuq-runner: the supervisor-managed daemon."""
from __future__ import annotations

import argparse
import logging
import signal
import sys
from pathlib import Path

from .claim import claim_dir
from .config import load_config, default_config_path, ConfigError
from .runner import Runner


def _warn_if_gpu_claim_disagrees(path: Path, cfg) -> None:
    """The runner and `gpu-claim` share one card; warn when they disagree.

    Two independent ways to disagree, and both end in double-booking:

    `--config` moves the runner's policy without moving `gpu-claim`'s. The
    two share one `<key>.lock.d` and must size the card the same way, which
    is why `config.vram_policy` and `config.max_holders` exist. But those
    read `default_config_path()` -- `$GPUQ_CONFIG`, else
    `/workspace/gpuq.toml` -- because a standalone claim has no way to know
    what flags the daemon was started with. Point the runner somewhere else
    and the two admit against different totals into one ledger.

    `[queue].claim_dir` is worse, because it does not even need a flag. The
    runner passes `cfg.claim_dir` to `gpu_claim` and `preflight`, while a
    bare `gpu-claim` reads `$GPU_CLAIM_DIR`. `bootstrap.sh` derives
    GPU_CLAIM_DIR from `$GPUQ_PREFIX` and templates it only into the
    supervisor unit, but `gpuq.example.toml` hardcodes its `claim_dir`. So
    a non-default prefix plus the example config copied verbatim gives one
    card *two* ledgers: each admits against a total the other's holders are
    missing from, and `gpu-claim`'s preflight reports the runner's live job
    as an unclaimed stray. `reaper.py` documents the same divergence from
    the other side.

    A warning rather than a refusal: a box may deliberately run the daemon
    off a path no interactive `gpu-claim` will ever be run on, and failing
    to start would break a deployment that works today. Loud, because the
    symptom otherwise is an OOM hours later with nothing pointing here.
    """
    if path.resolve() != default_config_path().resolve():
        logging.warning(
            "runner config is %s, but gpu-claim reads %s -- the two admit "
            "against the same card and will disagree about its capacity and "
            "its job limit. Export GPUQ_CONFIG=%s so both read one file.",
            path, default_config_path(), path)
    # None means the runner uses `claim_dir()` too, so there is nothing to
    # diverge from.
    if cfg.claim_dir is not None and \
            Path(cfg.claim_dir).resolve() != claim_dir().resolve():
        logging.warning(
            "runner claim_dir is %s, but gpu-claim reads %s -- one card "
            "with two ledgers. Each will admit against the other's holders "
            "as if the card were free. Export GPU_CLAIM_DIR=%s, or drop "
            "[queue].claim_dir so both read $GPU_CLAIM_DIR.",
            cfg.claim_dir, claim_dir(), cfg.claim_dir)


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
    _warn_if_gpu_claim_disagrees(path, cfg)

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
