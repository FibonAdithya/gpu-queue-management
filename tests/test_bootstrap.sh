#!/usr/bin/env bash
# tests/test_bootstrap.sh — run with: bash tests/test_bootstrap.sh
set -uo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
fails=0
check() { if eval "$2"; then echo "ok   - $1"; else echo "FAIL - $1"; fails=$((fails+1)); fi; }

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

# The checks that actually run bootstrap need a 3.11+ interpreter with pip.
# Prefer the repo venv, then any newer python on PATH. If there is none,
# skip those checks loudly rather than reporting a pass we did not earn.
PYTHON=""
for cand in "$repo/.venv/bin/python" python3.13 python3.12 python3.11 python3; do
  if command -v "$cand" >/dev/null 2>&1 &&
     "$cand" -c 'import sys;sys.exit(0 if sys.version_info>=(3,11) else 1)' 2>/dev/null &&
     "$cand" -c 'import pip' 2>/dev/null; then
    PYTHON="$cand"; break
  fi
done
export PYTHON

check "bootstrap.sh is executable" "[ -x '$repo/bootstrap.sh' ]"
check "supervisor conf is shipped" "[ -f '$repo/supervisor/gpuq-runner.conf' ]"
check "shellcheck-clean (skipped if absent)" \
  "! command -v shellcheck >/dev/null || shellcheck '$repo/bootstrap.sh'"
check "sets -euo pipefail" "grep -q 'set -euo pipefail' '$repo/bootstrap.sh'"
check "supervisor conf runs the runner" \
  "grep -q 'command=.*gpuqueue.cli_runner' '$repo/supervisor/gpuq-runner.conf'"
check "supervisor conf autorestarts" \
  "grep -q 'autorestart=true' '$repo/supervisor/gpuq-runner.conf'"
check "supervisor conf passes GPU_CLAIM_DIR" \
  "grep -q 'GPU_CLAIM_DIR' '$repo/supervisor/gpuq-runner.conf'"

if [ -z "$PYTHON" ]; then
  echo "SKIP - bootstrap install checks: no Python 3.11+ with pip on this box"
  echo "---"; [ "$fails" -eq 0 ] && echo "all passed" || { echo "$fails failed"; exit 1; }
  exit 0
fi

out="$(GPUQ_PREFIX="$tmp/ws" SUPERVISOR_CONF_DIR="$tmp/conf" \
       bash "$repo/bootstrap.sh" --dry-run --no-supervisor 2>&1)"
check "dry run touches nothing" "[ ! -d '$tmp/ws' ]"
check "dry run reports the queue root it would create" \
  "grep -q '$tmp/ws/queue' <<<'$out'"

GPUQ_PREFIX="$tmp/ws" SUPERVISOR_CONF_DIR="$tmp/conf" \
  bash "$repo/bootstrap.sh" --no-supervisor >/dev/null 2>&1
check "creates the queue tree" "[ -d '$tmp/ws/queue/pending' ]"
check "creates the claim dir" "[ -d '$tmp/ws/lock/gpu' ]"
check "writes a config when absent" "[ -f '$tmp/ws/gpuq.toml' ]"
# The example hardcodes /workspace. Left in place on a box with another
# prefix, the runner's ledger and a bare gpu-claim's are two directories.
check "config claim_dir follows the prefix" \
  "grep -q 'claim_dir = \"$tmp/ws/lock/gpu\"' '$tmp/ws/gpuq.toml'"

before="$(cat "$tmp/ws/gpuq.toml")"
echo "# edited by hand" >> "$tmp/ws/gpuq.toml"
GPUQ_PREFIX="$tmp/ws" SUPERVISOR_CONF_DIR="$tmp/conf" \
  bash "$repo/bootstrap.sh" --no-supervisor >/dev/null 2>&1
check "second run is idempotent and preserves an edited config" \
  "grep -q 'edited by hand' '$tmp/ws/gpuq.toml'"

GPUQ_PREFIX="$tmp/ws" SUPERVISOR_CONF_DIR="$tmp/conf" \
  bash "$repo/bootstrap.sh" >/dev/null 2>&1
check "installs the supervisor program file" \
  "[ -f '$tmp/conf/gpuq-runner.conf' ]"
# supervisord runs with its own PATH, which will not contain a venv's bin
# directory. A bare console-script name there fails with "ERROR (no such file)".
check "supervisor command uses an absolute interpreter, not a bare name" \
  "grep -qE '^command=/.* -m gpuqueue.cli_runner' '$tmp/conf/gpuq-runner.conf'"
check "no placeholders left unsubstituted" \
  "! grep -q '@[A-Z_]*@' '$tmp/conf/gpuq-runner.conf'"

echo "---"; [ "$fails" -eq 0 ] && echo "all passed" || { echo "$fails failed"; exit 1; }
