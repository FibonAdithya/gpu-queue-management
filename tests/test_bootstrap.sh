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
check "systemd unit is shipped" "[ -f '$repo/systemd/gpuq-runner.service' ]"
check "systemd unit runs the runner" \
  "grep -q '^ExecStart=.*gpuqueue.cli_runner' '$repo/systemd/gpuq-runner.service'"
check "systemd unit restarts the runner" \
  "grep -q '^Restart=always' '$repo/systemd/gpuq-runner.service'"
check "systemd unit passes GPU_CLAIM_DIR" \
  "grep -q 'GPU_CLAIM_DIR=' '$repo/systemd/gpuq-runner.service'"

# No check below may reach this machine's real service manager: with
# --init left to choose, a box running systemd gets a unit installed and
# started. Every run names its --init, and these fakes are first on PATH
# for the ones that mean to call a manager.
mkdir -p "$tmp/fakebin"
cat > "$tmp/fakebin/systemctl" <<FAKE
#!/bin/bash
echo "\$*" >> '$tmp/systemctl.log'
FAKE
chmod +x "$tmp/fakebin/systemctl"
export PATH="$tmp/fakebin:$PATH"
export SYSTEMD_UNIT_DIR="$tmp/units"

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
  bash "$repo/bootstrap.sh" --init supervisor >/dev/null 2>&1
check "installs the supervisor program file" \
  "[ -f '$tmp/conf/gpuq-runner.conf' ]"
# supervisord runs with its own PATH, which will not contain a venv's bin
# directory. A bare console-script name there fails with "ERROR (no such file)".
check "supervisor command uses an absolute interpreter, not a bare name" \
  "grep -qE '^command=/.* -m gpuqueue.cli_runner' '$tmp/conf/gpuq-runner.conf'"
check "no placeholders left unsubstituted" \
  "! grep -q '@[A-Z_]*@' '$tmp/conf/gpuq-runner.conf'"
check "--init supervisor installs no systemd unit" \
  "[ ! -e '$tmp/units/gpuq-runner.service' ]"

# env.sh: an ssh command line gets none of the daemon's environment, so
# without this a remote `gpuq` is either not on PATH or reading the
# default queue root rather than this box's.
check "writes env.sh under the prefix" "[ -f '$tmp/ws/env.sh' ]"
check "env.sh puts gpuq on PATH" \
  "[ -n \"\$(env -i PATH=/usr/bin:/bin bash -c '. $tmp/ws/env.sh && command -v gpuq')\" ]"
check "env.sh exports this box's queue root and claim dir" \
  "[ \"\$(env -i PATH=/usr/bin:/bin bash -c '. $tmp/ws/env.sh && echo \$QUEUE_ROOT:\$GPU_CLAIM_DIR:\$GPUQ_CONFIG')\" = '$tmp/ws/queue:$tmp/ws/lock/gpu:$tmp/ws/gpuq.toml' ]"

# ------------------------------------------------------------------ --init
out="$(GPUQ_PREFIX="$tmp/ws" SUPERVISOR_CONF_DIR="$tmp/conf-sd" \
       bash "$repo/bootstrap.sh" --dry-run --init systemd 2>&1)"
check "dry run names the systemd unit it would install" \
  "grep -q 'would: install $tmp/units/gpuq-runner.service' <<<'$out'"
check "dry run installs no systemd unit" "[ ! -e '$tmp/units/gpuq-runner.service' ]"
check "dry run calls no service manager" "[ ! -s '$tmp/systemctl.log' ]"

GPUQ_PREFIX="$tmp/ws" SUPERVISOR_CONF_DIR="$tmp/conf-sd" \
  bash "$repo/bootstrap.sh" --init systemd >/dev/null 2>&1
check "--init systemd installs the unit" "[ -f '$tmp/units/gpuq-runner.service' ]"
# Same reason as the supervisor command: systemd's PATH has no venv in it.
check "systemd ExecStart uses an absolute interpreter" \
  "grep -qE '^ExecStart=/.* -m gpuqueue.cli_runner' '$tmp/units/gpuq-runner.service'"
check "systemd unit has no placeholders left unsubstituted" \
  "! grep -q '@[A-Z_]*@' '$tmp/units/gpuq-runner.service'"
check "systemd unit carries this box's claim dir" \
  "grep -q 'GPU_CLAIM_DIR=$tmp/ws/lock/gpu' '$tmp/units/gpuq-runner.service'"
check "--init systemd installs no supervisor program file" \
  "[ ! -e '$tmp/conf-sd/gpuq-runner.conf' ]"
check "--init systemd reloads, enables and restarts the unit" \
  "grep -q 'daemon-reload' '$tmp/systemctl.log' &&
   grep -q 'enable gpuq-runner' '$tmp/systemctl.log' &&
   grep -q 'restart gpuq-runner' '$tmp/systemctl.log'"

rm -f "$tmp/units/gpuq-runner.service"; : > "$tmp/systemctl.log"
GPUQ_PREFIX="$tmp/ws" SUPERVISOR_CONF_DIR="$tmp/conf-none" \
  bash "$repo/bootstrap.sh" --init none >/dev/null 2>&1
check "--init none installs neither" \
  "[ ! -e '$tmp/units/gpuq-runner.service' ] && [ ! -e '$tmp/conf-none/gpuq-runner.conf' ]"
check "--init none calls no service manager" "[ ! -s '$tmp/systemctl.log' ]"

GPUQ_PREFIX="$tmp/ws" bash "$repo/bootstrap.sh" --dry-run --init upstart >/dev/null 2>&1
check "an unknown --init is refused with exit 2" "[ \$? -eq 2 ]"

# Left to choose: supervisor wins where it exists, because that is what
# every box deployed so far runs and a re-run must not move the runner to
# a second manager beside the first.
printf '#!/bin/bash\necho "$*" >> "%s"\n' "$tmp/supervisorctl.log" > "$tmp/fakebin/supervisorctl"
chmod +x "$tmp/fakebin/supervisorctl"
mkdir -p "$tmp/run-systemd"
out="$(GPUQ_PREFIX="$tmp/ws" SUPERVISOR_CONF_DIR="$tmp/conf-auto" SYSTEMD_RUN_DIR="$tmp/run-systemd" \
       bash "$repo/bootstrap.sh" --dry-run 2>&1)"
check "left to choose, supervisor wins when supervisorctl exists" \
  "grep -q 'would: install $tmp/conf-auto/gpuq-runner.conf' <<<'$out'"
rm -f "$tmp/fakebin/supervisorctl"
if command -v supervisorctl >/dev/null 2>&1; then
  echo "SKIP - systemd auto-selection: this box has a real supervisorctl"
else
  out="$(GPUQ_PREFIX="$tmp/ws" SUPERVISOR_CONF_DIR="$tmp/conf-auto" SYSTEMD_RUN_DIR="$tmp/run-systemd" \
         bash "$repo/bootstrap.sh" --dry-run 2>&1)"
  check "left to choose, systemd is used when it is running and supervisor is absent" \
    "grep -q 'would: install $tmp/units/gpuq-runner.service' <<<'$out'"
  # systemctl on PATH is not systemd running: a container image ships the
  # binary and has no manager behind it.
  out="$(GPUQ_PREFIX="$tmp/ws" SUPERVISOR_CONF_DIR="$tmp/conf-auto" SYSTEMD_RUN_DIR="$tmp/no-such-dir" \
         bash "$repo/bootstrap.sh" --dry-run 2>&1)"
  check "left to choose, a systemctl with no running systemd is not used" \
    "! grep -q 'gpuq-runner.service' <<<'$out'"
fi

# ----------------------------------------------------- a box without pip
#
# Stand-in interpreters: the real one, except that the two imports
# bootstrap asks about fail. What a Debian or Ubuntu python3 looks like
# before python3-pip and python3-venv are installed.
real_py="$(command -v "$PYTHON")"
if ! "$real_py" -c 'import ensurepip' 2>/dev/null; then
  echo "SKIP - pip-less install checks: $real_py cannot build a venv with pip"
  echo "---"; [ "$fails" -eq 0 ] && echo "all passed" || { echo "$fails failed"; exit 1; }
  exit 0
fi
cat > "$tmp/py-nopip" <<FAKE
#!/bin/bash
[ "\$1" = "-c" ] && [ "\$2" = "import pip" ] && exit 1
exec '$real_py' "\$@"
FAKE
cat > "$tmp/py-bare" <<FAKE
#!/bin/bash
[ "\$1" = "-c" ] && { [ "\$2" = "import pip" ] || [ "\$2" = "import ensurepip" ]; } && exit 1
exec '$real_py' "\$@"
FAKE
chmod +x "$tmp/py-nopip" "$tmp/py-bare"

out="$(PYTHON="$tmp/py-nopip" GPUQ_PREFIX="$tmp/ws2" \
       bash "$repo/bootstrap.sh" --dry-run --init none 2>&1)"
check "dry run on a pip-less interpreter says it would create a venv" \
  "grep -q 'would: create $tmp/ws2/venv' <<<'$out'"
check "dry run on a pip-less interpreter creates nothing" "[ ! -d '$tmp/ws2' ]"

PYTHON="$tmp/py-nopip" GPUQ_PREFIX="$tmp/ws2" SUPERVISOR_CONF_DIR="$tmp/conf2" \
  bash "$repo/bootstrap.sh" --init supervisor >"$tmp/nopip.out" 2>&1
check "a pip-less interpreter gets a venv under the prefix" \
  "[ -x '$tmp/ws2/venv/bin/python' ]"
check "the runner is installed into that venv" \
  "'$tmp/ws2/venv/bin/python' -c 'import gpuqueue' 2>/dev/null"
check "the program file runs the venv's interpreter, not the pip-less one" \
  "grep -q '^command=$tmp/ws2/venv/bin/python -m gpuqueue.cli_runner' '$tmp/conf2/gpuq-runner.conf'"
check "env.sh points at the venv" "grep -q '$tmp/ws2/venv/bin' '$tmp/ws2/env.sh'"
check "bootstrap reports the interpreter the runner ended up in" \
  "grep -qx 'runner python: $tmp/ws2/venv/bin/python' '$tmp/nopip.out'"
PYTHON="$tmp/py-nopip" GPUQ_PREFIX="$tmp/ws2" SUPERVISOR_CONF_DIR="$tmp/conf2" \
  bash "$repo/bootstrap.sh" --init supervisor >/dev/null 2>&1
check "a second run on a pip-less interpreter reuses the venv" "[ \$? -eq 0 ]"

# No ensurepip either: the venv is built without pip and pip is fetched
# into it. The stand-in get-pip records that it ran, and under which
# interpreter, then installs pip the only way available offline.
cat > "$tmp/get-pip.py" <<FAKE
import subprocess, sys
open('$tmp/get-pip.ran', 'w').write(sys.prefix)
subprocess.check_call([sys.executable, '-m', 'ensurepip', '--default-pip'], stdout=subprocess.DEVNULL)
FAKE
PYTHON="$tmp/py-bare" GPUQ_PREFIX="$tmp/ws3" GET_PIP_URL="file://$tmp/get-pip.py" \
  bash "$repo/bootstrap.sh" --init none >/dev/null 2>&1
check "with no ensurepip, pip is fetched into the venv, not the system interpreter" \
  "[ \"\$(cat '$tmp/get-pip.ran' 2>/dev/null)\" = '$tmp/ws3/venv' ]"
check "with no ensurepip, the runner still ends up installed" \
  "'$tmp/ws3/venv/bin/python' -c 'import gpuqueue' 2>/dev/null"

PYTHON="$tmp/py-bare" GPUQ_PREFIX="$tmp/ws4" GET_PIP_URL="file://$tmp/no-such-get-pip.py" \
  bash "$repo/bootstrap.sh" --init none >"$tmp/nofetch.out" 2>&1
check "a pip that cannot be fetched stops bootstrap" "[ \$? -ne 0 ]"
check "a pip that cannot be fetched says what to install instead" \
  "grep -q 'python3-venv' '$tmp/nofetch.out'"

echo "---"; [ "$fails" -eq 0 ] && echo "all passed" || { echo "$fails failed"; exit 1; }
