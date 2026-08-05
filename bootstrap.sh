#!/usr/bin/env bash
# bootstrap.sh — take a bare box to a working runner, idempotently.
#
# Host identity lives in one variable: GPUQ_PREFIX. Rebuilding a destroyed
# box is an ssh-target edit plus a run of this script.
set -euo pipefail

GPUQ_PREFIX="${GPUQ_PREFIX:-/workspace}"
QUEUE_ROOT="${QUEUE_ROOT:-$GPUQ_PREFIX/queue}"
GPU_CLAIM_DIR="${GPU_CLAIM_DIR:-$GPUQ_PREFIX/lock/gpu}"
GPUQ_CONFIG="${GPUQ_CONFIG:-$GPUQ_PREFIX/gpuq.toml}"
SUPERVISOR_CONF_DIR="${SUPERVISOR_CONF_DIR:-/etc/supervisor/conf.d}"
# Which interpreter the runner is installed into. A box with several
# Pythons must not have this guessed for it: the runner has to live in
# the one that is 3.11+, which is not always the one called python3.
PYTHON="${PYTHON:-python3}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DRY_RUN=0
USE_SUPERVISOR=1
for arg in "$@"; do
  case "$arg" in
    --dry-run)       DRY_RUN=1 ;;
    --no-supervisor) USE_SUPERVISOR=0 ;;
    -h|--help)
      sed -n '2,8p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "bootstrap: unknown argument: $arg" >&2; exit 2 ;;
  esac
done

say() { printf '%s\n' "$*" >&2; }
run() { if [ "$DRY_RUN" -eq 1 ]; then say "would: $*"; else "$@"; fi; }

say "prefix:      $GPUQ_PREFIX"
say "queue root:  $QUEUE_ROOT"
say "claim dir:   $GPU_CLAIM_DIR"
say "config:      $GPUQ_CONFIG"
say "python:      $PYTHON"

# 1. install the package
# Check the interpreter first: pip's requires-python failure names a version
# but not what to do about it, and this runs on boxes nobody built by hand.
"$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' || {
  say "bootstrap: need Python 3.11+ (tomllib); $PYTHON is $("$PYTHON" -V 2>&1)"
  say "           set PYTHON=/path/to/python3.11 if the box has another one"
  exit 1
}

if [ "$DRY_RUN" -eq 1 ]; then
  say "would: $PYTHON -m pip install -e $REPO_DIR"
else
  "$PYTHON" -m pip install --quiet -e "$REPO_DIR"
fi

# 2. state directories
for d in pending running done failed logs work; do
  run mkdir -p "$QUEUE_ROOT/$d"
done
run mkdir -p "$GPU_CLAIM_DIR"

# 3. config, written once and never overwritten
if [ "$DRY_RUN" -eq 1 ]; then
  say "would: write $GPUQ_CONFIG if absent"
elif [ -f "$GPUQ_CONFIG" ]; then
  say "config exists, leaving it alone: $GPUQ_CONFIG"
else
  sed -e "s|^root = .*|root = \"$QUEUE_ROOT\"|" \
      "$REPO_DIR/gpuq.example.toml" > "$GPUQ_CONFIG"
  say "wrote $GPUQ_CONFIG — declare your projects in it, then rerun"
fi

# 4. clone declared checkouts
#
# Never fatal. The first run writes a config full of example placeholders and
# tells you to edit it, so a failed clone here is the expected case, not an
# error -- and aborting would stop the supervisor program file from being
# installed at all. Report per project and carry on; the runner fails jobs
# for an unclonable project with a legible message of its own.
if [ "$DRY_RUN" -eq 0 ] && [ -f "$GPUQ_CONFIG" ]; then
  GPUQ_CONFIG="$GPUQ_CONFIG" "$PYTHON" - <<'PYEOF' || say "checkout step reported problems; continuing"
import os
import sys
from pathlib import Path
from gpuqueue.config import load_config
from gpuqueue.git_ops import ensure_checkout

cfg = load_config(Path(os.environ["GPUQ_CONFIG"]))
for name, project in cfg.projects.items():
    try:
        print(f"checkout {name}: {ensure_checkout(project)}", file=sys.stderr)
    except Exception as e:
        print(f"checkout {name}: SKIPPED -- {e}", file=sys.stderr)
PYEOF
fi

# 5. supervisor program file, shipped rather than hand-written
if [ "$USE_SUPERVISOR" -eq 1 ]; then
  run mkdir -p "$SUPERVISOR_CONF_DIR"
  if [ "$DRY_RUN" -eq 1 ]; then
    say "would: install $SUPERVISOR_CONF_DIR/gpuq-runner.conf"
  else
    sed -e "s|@QUEUE_ROOT@|$QUEUE_ROOT|g" \
        -e "s|@GPU_CLAIM_DIR@|$GPU_CLAIM_DIR|g" \
        -e "s|@GPUQ_CONFIG@|$GPUQ_CONFIG|g" \
        -e "s|@GPUQ_PREFIX@|$GPUQ_PREFIX|g" \
        "$REPO_DIR/supervisor/gpuq-runner.conf" \
        > "$SUPERVISOR_CONF_DIR/gpuq-runner.conf"
    if command -v supervisorctl >/dev/null 2>&1; then
      supervisorctl reread  || say "supervisorctl reread failed; is supervisord running?"
      supervisorctl update  || true
      supervisorctl restart gpuq-runner || supervisorctl start gpuq-runner || true
    else
      say "supervisorctl not found; program file installed but not started"
    fi
  fi
fi

say "bootstrap complete"
