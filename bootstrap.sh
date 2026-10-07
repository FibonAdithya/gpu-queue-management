#!/usr/bin/env bash
# bootstrap.sh — take a bare box to a working runner, idempotently.
#
# Host identity lives in one variable: GPUQ_PREFIX. Rebuilding a destroyed
# box is an ssh-target edit plus a run of this script.
#
# Flags: --dry-run, --init supervisor|systemd|none (default: chosen from
# what the box runs). See docs/deploying.md for the variables it honours.
set -euo pipefail

GPUQ_PREFIX="${GPUQ_PREFIX:-/workspace}"
QUEUE_ROOT="${QUEUE_ROOT:-$GPUQ_PREFIX/queue}"
GPU_CLAIM_DIR="${GPU_CLAIM_DIR:-$GPUQ_PREFIX/lock/gpu}"
GPUQ_CONFIG="${GPUQ_CONFIG:-$GPUQ_PREFIX/gpuq.toml}"
SUPERVISOR_CONF_DIR="${SUPERVISOR_CONF_DIR:-/etc/supervisor/conf.d}"
# Where the runner goes when $PYTHON has no pip. The interpreter itself is
# left alone: no root needed, and nothing a distro marks as externally
# managed is touched.
GPUQ_VENV="${GPUQ_VENV:-$GPUQ_PREFIX/venv}"
GET_PIP_URL="${GET_PIP_URL:-https://bootstrap.pypa.io/get-pip.py}"
# Root gets a system unit; anyone else a user unit, which needs lingering
# enabled to outlive their login.
if [ "$(id -u)" -eq 0 ]; then
  SYSTEMD_UNIT_DIR="${SYSTEMD_UNIT_DIR:-/etc/systemd/system}"
  SYSTEMCTL=(systemctl); SYSTEMD_WANTED_BY="multi-user.target"
else
  SYSTEMD_UNIT_DIR="${SYSTEMD_UNIT_DIR:-${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user}"
  SYSTEMCTL=(systemctl --user); SYSTEMD_WANTED_BY="default.target"
  # An ssh command line does not always carry this, and `systemctl --user`
  # cannot find the user manager without it.
  export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
fi
# Exists exactly when systemd is the running init. `systemctl` on PATH is
# not that: container images ship the binary with no manager behind it.
SYSTEMD_RUN_DIR="${SYSTEMD_RUN_DIR:-/run/systemd/system}"
# Which interpreter the runner is installed into. A box with several
# Pythons must not have this guessed for it: the runner has to live in
# the one that is 3.11+, which is not always the one called python3.
PYTHON="${PYTHON:-python3}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DRY_RUN=0
# What keeps the runner alive: supervisor | systemd | none | auto.
INIT=auto
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run)       DRY_RUN=1 ;;
    --init)          INIT="${2:-}"; [ $# -lt 2 ] || shift ;;
    --init=*)        INIT="${1#--init=}" ;;
    --no-supervisor) INIT=none ;;      # the older spelling of --init none
    -h|--help)
      sed -n '2,8p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "bootstrap: unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done
case "$INIT" in
  supervisor|systemd|none) ;;
  auto)
    # Supervisor wherever it exists: it is what every box deployed so far
    # runs, and a re-run must not start a second runner under a second
    # manager. systemd only when it is actually running. Neither: the
    # supervisor program file, installed and not started, as before.
    if ! command -v supervisorctl >/dev/null 2>&1 &&
       [ -d "$SYSTEMD_RUN_DIR" ] && command -v systemctl >/dev/null 2>&1; then
      INIT=systemd
    else
      INIT=supervisor
    fi ;;
  *) echo "bootstrap: --init takes supervisor, systemd or none, not '$INIT'" >&2; exit 2 ;;
esac

say() { printf '%s\n' "$*" >&2; }
run() { if [ "$DRY_RUN" -eq 1 ]; then say "would: $*"; else "$@"; fi; }

say "prefix:      $GPUQ_PREFIX"
say "queue root:  $QUEUE_ROOT"
say "claim dir:   $GPU_CLAIM_DIR"
say "config:      $GPUQ_CONFIG"
say "python:      $PYTHON"
say "init:        $INIT"

# 1. install the package
# Check the interpreter first: pip's requires-python failure names a version
# but not what to do about it, and this runs on boxes nobody built by hand.
"$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' || {
  say "bootstrap: need Python 3.11+ (tomllib); $PYTHON is $("$PYTHON" -V 2>&1)"
  say "           set PYTHON=/path/to/python3.11 if the box has another one"
  exit 1
}

# No pip in $PYTHON: build a venv from it and install there. Common on
# Debian and Ubuntu, where pip and ensurepip are both separate packages --
# so a plain `python -m venv` fails on exactly the boxes that need this,
# and the venv is built without pip and pip fetched into it instead.
if ! "$PYTHON" -c 'import pip' 2>/dev/null; then
  if [ "$DRY_RUN" -eq 1 ]; then
    say "would: create $GPUQ_VENV from $PYTHON, which has no pip"
  else
    if [ ! -x "$GPUQ_VENV/bin/python" ]; then
      say "$PYTHON has no pip; creating $GPUQ_VENV"
      mkdir -p "$(dirname "$GPUQ_VENV")"
      if "$PYTHON" -c 'import ensurepip' 2>/dev/null; then
        "$PYTHON" -m venv "$GPUQ_VENV"
      else
        "$PYTHON" -m venv --without-pip "$GPUQ_VENV"
      fi
    fi
    if ! "$GPUQ_VENV/bin/python" -c 'import pip' 2>/dev/null; then
      say "fetching pip into $GPUQ_VENV from $GET_PIP_URL"
      # Fetched with the interpreter itself, so this needs no curl or wget.
      if ! { "$GPUQ_VENV/bin/python" -c \
               'import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])' \
               "$GET_PIP_URL" "$GPUQ_VENV/get-pip.py" &&
             "$GPUQ_VENV/bin/python" "$GPUQ_VENV/get-pip.py" --quiet; }; then
        say "bootstrap: could not install pip into $GPUQ_VENV"
        say "           install it another way and rerun, e.g. on Debian/Ubuntu:"
        say "           apt-get install python3-venv   (or python3-pip)"
        exit 1
      fi
      rm -f "$GPUQ_VENV/get-pip.py"
    fi
  fi
  PYTHON="$GPUQ_VENV/bin/python"
fi
# Absolute from here on: the service managers below run with their own
# PATH, and env.sh is sourced from shells that have another one again.
PYTHON_ABS="$(command -v "$PYTHON" || echo "$PYTHON")"

if [ "$DRY_RUN" -eq 1 ]; then
  say "would: $PYTHON -m pip install -e $REPO_DIR"
else
  "$PYTHON" -m pip install --quiet -e "$REPO_DIR"
fi

# 2. state directories
# "done" quoted: bash reads it as a plain word here, but shellcheck 0.11
# cannot tell that from a missing semicolon before the loop's own `done`.
for d in pending running "done" failed logs work; do
  run mkdir -p "$QUEUE_ROOT/$d"
done
run mkdir -p "$GPU_CLAIM_DIR"

# 3. config, written once and never overwritten
if [ "$DRY_RUN" -eq 1 ]; then
  say "would: write $GPUQ_CONFIG if absent"
elif [ -f "$GPUQ_CONFIG" ]; then
  say "config exists, leaving it alone: $GPUQ_CONFIG"
else
  # claim_dir for the same reason as root: the example hardcodes the
  # default prefix, and on a box with another one the runner would write
  # its ledger somewhere no bare `gpu-claim` reads. One card, two ledgers,
  # each admitting against a total the other's holders are missing from.
  # `cli_runner` warns about exactly this at startup; not creating it is
  # better.
  sed -e "s|^root = .*|root = \"$QUEUE_ROOT\"|" \
      -e "s|^claim_dir = .*|claim_dir = \"$GPU_CLAIM_DIR\"|" \
      "$REPO_DIR/gpuq.example.toml" > "$GPUQ_CONFIG"
  say "wrote $GPUQ_CONFIG — declare your projects in it, then rerun"
fi

# env.sh: what a shell needs to talk to this box's queue. An ssh command
# line inherits none of the daemon's environment, so without it a remote
# `gpuq` is either not on PATH or reading the built-in default queue root,
# and a hand-run `gpu-claim` writes to a claim directory the runner never
# reads. Derived entirely from the above, so rewritten on every run.
if [ "$DRY_RUN" -eq 1 ]; then
  say "would: write $GPUQ_PREFIX/env.sh"
else
  # Where pip put `gpuq`, asked rather than assumed to be beside the
  # interpreter: a non-root install into a system Python lands in the user
  # scheme (~/.local/bin), and Debian's root scheme is /usr/local/bin.
  SCRIPTS_DIR="$("$PYTHON" - <<'PYEOF'
import os
import sysconfig

dirs = [sysconfig.get_path("scripts"),
        sysconfig.get_path("scripts", f"{os.name}_user")]
print(next((d for d in dirs if os.path.exists(os.path.join(d, "gpuq"))), dirs[0]))
PYEOF
)"
  cat > "$GPUQ_PREFIX/env.sh" <<ENVEOF
# Written by bootstrap.sh; rerunning it rewrites this file.
# Use:  . $GPUQ_PREFIX/env.sh
export PATH="$SCRIPTS_DIR:\$PATH"
export QUEUE_ROOT="$QUEUE_ROOT"
export GPU_CLAIM_DIR="$GPU_CLAIM_DIR"
export GPUQ_CONFIG="$GPUQ_CONFIG"
ENVEOF
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

# 6. agent skill, so anything working on this box knows to queue its work
GPUQ_SKILLS_DIR="${GPUQ_SKILLS_DIR:-$HOME/.claude/skills}"
run mkdir -p "$GPUQ_SKILLS_DIR/gpu-jobs"
if [ "$DRY_RUN" -eq 1 ]; then
  say "would: install skill to $GPUQ_SKILLS_DIR/gpu-jobs/SKILL.md"
else
  # Copied, not symlinked: the skill must survive the repo checkout moving,
  # and an agent reading a dangling symlink gets nothing and no explanation.
  cp "$REPO_DIR/skills/gpu-jobs/SKILL.md" "$GPUQ_SKILLS_DIR/gpu-jobs/SKILL.md"
  say "skill installed: $GPUQ_SKILLS_DIR/gpu-jobs/SKILL.md"
fi

# 5. the unit that keeps the runner alive, shipped rather than hand-written
install_unit() {   # $1 shipped template, $2 destination
  sed -e "s|@PYTHON@|$PYTHON_ABS|g" \
      -e "s|@QUEUE_ROOT@|$QUEUE_ROOT|g" \
      -e "s|@GPU_CLAIM_DIR@|$GPU_CLAIM_DIR|g" \
      -e "s|@GPUQ_CONFIG@|$GPUQ_CONFIG|g" \
      -e "s|@GPUQ_PREFIX@|$GPUQ_PREFIX|g" \
      -e "s|@WANTED_BY@|$SYSTEMD_WANTED_BY|g" \
      "$1" > "$2"
}

if [ "$INIT" = "supervisor" ]; then
  run mkdir -p "$SUPERVISOR_CONF_DIR"
  if [ "$DRY_RUN" -eq 1 ]; then
    say "would: install $SUPERVISOR_CONF_DIR/gpuq-runner.conf"
  else
    install_unit "$REPO_DIR/supervisor/gpuq-runner.conf" \
                 "$SUPERVISOR_CONF_DIR/gpuq-runner.conf"
    if command -v supervisorctl >/dev/null 2>&1; then
      supervisorctl reread  || say "supervisorctl reread failed; is supervisord running?"
      supervisorctl update  || true
      supervisorctl restart gpuq-runner || supervisorctl start gpuq-runner || true
    else
      say "supervisorctl not found; program file installed but not started"
    fi
  fi
elif [ "$INIT" = "systemd" ]; then
  run mkdir -p "$SYSTEMD_UNIT_DIR"
  if [ "$DRY_RUN" -eq 1 ]; then
    say "would: install $SYSTEMD_UNIT_DIR/gpuq-runner.service"
  else
    install_unit "$REPO_DIR/systemd/gpuq-runner.service" \
                 "$SYSTEMD_UNIT_DIR/gpuq-runner.service"
    if command -v systemctl >/dev/null 2>&1; then
      "${SYSTEMCTL[@]}" daemon-reload || say "${SYSTEMCTL[*]} daemon-reload failed; is systemd running?"
      "${SYSTEMCTL[@]}" enable gpuq-runner || true
      "${SYSTEMCTL[@]}" restart gpuq-runner || true
      if [ "$(id -u)" -ne 0 ]; then
        say "installed as a user unit: it stops when you log out unless lingering is on."
        say "  once, as root:  loginctl enable-linger $(id -un)"
      fi
    else
      say "systemctl not found; unit installed but not started"
    fi
  fi
fi

# Last, and in a fixed form: deploy.sh reads this line to learn where the
# runner went, which is not $PYTHON on a box that had no pip.
say "runner python: $PYTHON_ABS"
say "bootstrap complete"
