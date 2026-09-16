#!/usr/bin/env bash
# tests/test_deploy.sh — run with: bash tests/test_deploy.sh
#
# deploy.sh talks to a remote box, so the parts worth testing are the ones
# that decide things: which interpreter to install into, what class of box
# nvidia-smi is describing, and what verdict those add up to. Those are
# plain functions over text, and this sources deploy.sh to call them
# directly rather than standing up a box.
#
# The transport is covered separately through $DEPLOY_SSH, which replaces
# `ssh` with a fake that replays canned probe output and records what it
# was asked to run. That is what keeps "probe changes nothing" honest.
set -uo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
fails=0
check() { if eval "$2"; then echo "ok   - $1"; else echo "FAIL - $1"; fails=$((fails+1)); fi; }

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

# Call one of deploy.sh's decision functions in a subshell. `set +e` because
# deploy.sh runs `set -e` at file scope and these functions are expected to
# return non-zero for "refuse", which must not abort this harness.
run_fn() { ( source "$repo/deploy.sh" >/dev/null 2>&1; set +e; "$@" ); }

# ---------------------------------------------------------------- structure

check "deploy.sh exists" "[ -f '$repo/deploy.sh' ]"
check "deploy.sh is executable" "[ -x '$repo/deploy.sh' ]"
check "sets -euo pipefail" "grep -q 'set -euo pipefail' '$repo/deploy.sh'"
check "shellcheck-clean (skipped if absent)" \
  "! command -v shellcheck >/dev/null || shellcheck '$repo/deploy.sh'"
check "--help exits 0" "bash '$repo/deploy.sh' --help >/dev/null 2>&1"
check "--help names the phases" \
  "bash '$repo/deploy.sh' --help 2>&1 | grep -qi 'probe' &&
   bash '$repo/deploy.sh' --help 2>&1 | grep -qi 'verify'"
# Without the BASH_SOURCE guard, sourcing runs main and tries to ssh at a
# box that was never named -- which is also how every check below would
# start failing for a reason that has nothing to do with what it asserts.
check "sourcing runs nothing" \
  "( source '$repo/deploy.sh' >/dev/null 2>&1 ); [ \$? -eq 0 ]"

# ------------------------------------------------------------ choose_python
#
# The judgment call this script exists to stop a human making by hand: an
# ML image ships several interpreters and the runner belongs in the one
# that is 3.11+, which is usually the one holding torch and is almost never
# the one called `python3`.

probe_two_pythons() {
  cat <<'EOF'
py	/usr/bin/python3	3.12.3	1	0
py	/venv/main/bin/python	3.11.9	1	1
EOF
}
check "prefers the 3.11+ interpreter holding torch over a newer bare one" \
  "[ \"\$(probe_two_pythons | run_fn choose_python)\" = '/venv/main/bin/python' ]"

probe_no_torch() {
  cat <<'EOF'
py	/usr/bin/python3	3.11.2	1	0
py	/venv/main/bin/python	3.12.14	1	0
EOF
}
check "picks the newest 3.11+ when none has torch" \
  "[ \"\$(probe_no_torch | run_fn choose_python)\" = '/venv/main/bin/python' ]"

probe_too_old() {
  cat <<'EOF'
py	/usr/bin/python3	3.10.6	1	0
py	/usr/bin/python2	2.7.18	0	0
EOF
}
check "refuses when nothing is 3.11+" \
  "! probe_too_old | run_fn choose_python >/dev/null 2>&1"
# tomllib is stdlib-only from 3.11; a 3.10 that pip happens to work on
# would install and then fail at import on the first job.
#
# Captured rather than piped straight into grep: this harness runs under
# `pipefail`, and choose_python exits non-zero on purpose here, so a
# pipeline would report that refusal as the check's own failure no matter
# what the message said.
floor_msg="$(probe_too_old | run_fn choose_python 2>&1 || true)"
check "names the version floor when it refuses" \
  "grep -q '3\.11' <<< \"\$floor_msg\""

probe_torch_too_old() {
  cat <<'EOF'
py	/usr/bin/python3.10	3.10.6	1	1
py	/venv/main/bin/python	3.12.14	1	0
EOF
}
check "will not pick a torch interpreter that is below the floor" \
  "[ \"\$(probe_torch_too_old | run_fn choose_python)\" = '/venv/main/bin/python' ]"

probe_no_pip() {
  cat <<'EOF'
py	/usr/bin/python3	3.12.3	0	0
py	/venv/main/bin/python	3.11.9	1	0
EOF
}
check "skips a 3.11+ interpreter that has no pip" \
  "[ \"\$(probe_no_pip | run_fn choose_python)\" = '/venv/main/bin/python' ]"

# -------------------------------------------------------- classify_gpu_pids
#
# The check today's deploy turned up. nvidia-smi listing processes is not
# enough to call preflight a real guard: the pids may be the host's, and
# then every pid-keyed mechanism in gpuq is comparing two namespaces.

check "a reported pid present in /proc is a local-pid box" \
  "[ \"\$(run_fn classify_gpu_pids '2245, 614 MiB, python' '2245')\" = 'local-pids' ]"
check "a reported pid absent from /proc is a host-pid box" \
  "[ \"\$(run_fn classify_gpu_pids '3006382, 614 MiB, [Not Found]' '')\" = 'host-pids' ]"
check "no processes reported is idle, not a verdict about pids" \
  "[ \"\$(run_fn classify_gpu_pids '' '')\" = 'idle' ]"
check "[Not Supported] is its own class" \
  "[ \"\$(run_fn classify_gpu_pids '[Not Supported]' '')\" = 'unsupported' ]"
# Mixed is the case that decides whether the check is a real one: one
# resolvable pid does not make the box safe if another is not.
check "any unresolvable pid makes it a host-pid box" \
  "[ \"\$(run_fn classify_gpu_pids '2245, 1 MiB, a
3006382, 614 MiB, [Not Found]' '2245')\" = 'host-pids' ]"

# ------------------------------------------------------------------ verdict

check "local pids and working capacity discovery means sharing works" \
  "run_fn verdict local-pids 12288 1 | grep -qi 'sharing works'"
check "host pids means one GPU job at a time" \
  "run_fn verdict host-pids 12288 1 | grep -qi 'one GPU job at a time'"
# The other road to the same place: without a total, every GPU job is
# admitted exclusively, which is the same operational fact.
check "capacity discovery failure also means one GPU job at a time" \
  "run_fn verdict local-pids none 1 | grep -qi 'one GPU job at a time'"
check "no nvidia-smi means no GPU lane" \
  "run_fn verdict nosmi none 0 | grep -qi 'no GPU lane'"
# An idle card is the absence of evidence, and on 2026-09-16 it produced
# "sharing works" for a KVM box nothing had been measured on. Asserted in
# both directions: a verdict that printed nothing would pass the negative.
check "an idle card is not reported as sharing works" \
  "! run_fn verdict idle 12288 1 | grep -qi 'sharing works'"
check "an idle card is reported as unverified" \
  "run_fn verdict idle 12288 1 | grep -qi 'unverified'"
# Only an actual local-pid result earns the good verdict; a class this
# function has never heard of must not fall through to it.
check "an unrecognised pid class is unverified, not sharing works" \
  "run_fn verdict '' 12288 1 | grep -qi 'unverified'"

# --------------------------------------------------------------- ref_warning
#
# The box clones from origin over HTTPS, so it can only check out what
# origin actually has. Defaulting the ref to the local HEAD sha -- which is
# the obvious thing to write -- deploys a commit that exists on nobody's
# remote the moment you have an unpushed commit, and `git checkout <sha>`
# on the box fails with "unknown revision" after the clone has succeeded.

# Exit status asserted as well as emptiness: a ref_warning that does not
# exist at all also prints nothing, so silence alone is not evidence that
# the agreeing case was recognised.
check "no warning when local HEAD is what origin has" \
  "out=\"\$(run_fn ref_warning abc1234 abc1234)\"; rc=\$?;
   [ \$rc -eq 0 ] && [ -z \"\$out\" ]"
check "warns when local HEAD is ahead of origin" \
  "run_fn ref_warning aaaaaaa bbbbbbb | grep -qi 'origin'"
check "the warning names both commits so you can tell which you got" \
  "run_fn ref_warning aaaaaaadeadbeef bbbbbbbdeadbeef | grep -q aaaaaaa &&
   run_fn ref_warning aaaaaaadeadbeef bbbbbbbdeadbeef | grep -q bbbbbbb"
# An empty remote sha means ls-remote resolved nothing -- a ref that is not
# there at all, which must not read as "no warning, carry on".
check "a ref missing from origin is refused, not merely unremarked" \
  "! run_fn ref_warning aaaaaaa '' >/dev/null 2>&1"

# --------------------------------------------------------- transport / fake
#
# $DEPLOY_SSH replaces ssh. The fake records every script it is handed, so
# a phase that was supposed to change nothing can be held to it.

cat > "$tmp/fake-ssh" <<'FAKE'
#!/usr/bin/env bash
# Stands in for `ssh <box> bash -s`. Records the script, answers by op.
script="$(cat)"
op="$(printf '%s\n' "$script" | sed -n 's/^#gpuq-op:\(.*\)$/\1/p' | head -1)"
printf '%s\n' "$op" >> "$DEPLOY_FAKE_LOG"
case "$op" in
  probe)
    cat <<'EOF'
py	/usr/bin/python3	3.12.3	1	0
py	/venv/main/bin/python	3.12.14	1	0
smi	1
gpu	NVIDIA GeForce RTX 3060	GPU-a4fe9d31	12288
supervisor	1
prefixwritable	1
git	1
EOF
    ;;
  *) : ;;
esac
FAKE
chmod +x "$tmp/fake-ssh"

export DEPLOY_SSH="$tmp/fake-ssh"

DEPLOY_FAKE_LOG="$tmp/ops-probe.log" ; export DEPLOY_FAKE_LOG
: > "$DEPLOY_FAKE_LOG"
bash "$repo/deploy.sh" fake-box --probe-only >"$tmp/probe.out" 2>&1
check "--probe-only exits 0 on a good box" "[ \$? -eq 0 ]"
check "--probe-only reports the card" "grep -q '3060' '$tmp/probe.out'"
check "--probe-only reports the chosen interpreter" \
  "grep -q '/venv/main/bin/python' '$tmp/probe.out'"
# The whole point of a probe: it is safe to run against a box you have not
# decided to install on yet.
check "--probe-only never runs install or verify" \
  "! grep -qE '^(install|verify)$' '$DEPLOY_FAKE_LOG'"
check "--probe-only did run the probe" "grep -qx 'probe' '$DEPLOY_FAKE_LOG'"

# The default ref must be resolvable on origin, which a local HEAD sha is
# not the moment you have an unpushed commit. Asserted against the script
# deploy.sh actually sends, because ref_warning being correct says nothing
# about do_install calling it with the right ref in the first place.
cat > "$tmp/fake-capture" <<'FAKE'
#!/usr/bin/env bash
script="$(cat)"
op="$(printf '%s\n' "$script" | sed -n 's/^#gpuq-op:\(.*\)$/\1/p' | head -1)"
printf '%s' "$script" > "$DEPLOY_FAKE_DIR/$op.script"
case "$op" in
  probe)
    printf 'py\t/venv/main/bin/python\t3.12.14\t1\t1\nsmi\t1\n'
    printf 'gpu\tNVIDIA GeForce RTX 3060\tGPU-a4fe\t12288\n'
    printf 'supervisor\t1\nprefixwritable\t1\ngit\t1\n' ;;
  install) printf 'abc1234\nbootstrap complete\n' ;;
  *) : ;;
esac
FAKE
chmod +x "$tmp/fake-capture"
mkdir -p "$tmp/scripts"
DEPLOY_FAKE_DIR="$tmp/scripts"; export DEPLOY_FAKE_DIR
DEPLOY_FAKE_LOG="$tmp/ops-cap.log"; export DEPLOY_FAKE_LOG

# Run against a purpose-built repo rather than this one. Reading the real
# checkout's branch and remote would make the result depend on whether the
# branch this suite happens to be running on has been pushed yet -- which
# is environment, not behaviour, and it flips this test's answer.
#
# The shape that matters is a branch origin has, plus a local commit it
# does not: exactly the state that made a HEAD-sha default deploy nothing.
gitq() { git -c user.email=t@t -c user.name=t -c init.defaultBranch=main "$@"; }
gitq init --quiet --bare "$tmp/origin.git"
gitq init --quiet "$tmp/work"
cp "$repo/deploy.sh" "$tmp/work/deploy.sh"
gitq -C "$tmp/work" add deploy.sh
gitq -C "$tmp/work" commit --quiet -m pushed
gitq -C "$tmp/work" remote add origin "$tmp/origin.git"
gitq -C "$tmp/work" push --quiet -u origin main
pushed_sha="$(gitq -C "$tmp/work" rev-parse HEAD)"
echo unpushed >> "$tmp/work/deploy-note.txt"
gitq -C "$tmp/work" add deploy-note.txt
gitq -C "$tmp/work" commit --quiet -m 'local only'
head_sha="$(gitq -C "$tmp/work" rev-parse HEAD)"

check "the fixture really is ahead of its origin" \
  "[ '$head_sha' != '$pushed_sha' ]"

DEPLOY_SSH="$tmp/fake-capture" bash "$tmp/work/deploy.sh" fake-box >/dev/null 2>&1 || true

check "install was reached with a script to inspect" \
  "[ -s '$tmp/scripts/install.script' ]"
check "the deployed ref is the branch, not the local HEAD sha" \
  "grep -q \"checkout --quiet 'main'\" '$tmp/scripts/install.script'"
check "an unpushed local HEAD is never handed to the box as a ref" \
  "! grep -q '$head_sha' '$tmp/scripts/install.script'"

# ------------------------------------------------- verify on a driver-only box
#
# The pid check needs a CUDA process to look at, and on an idle card verify
# has to make one. It could only do that with nvcc or torch, so on
# 2026-09-16 a KVM box with neither verified nothing and was called "sharing
# works". The driver's own libcuda can open a context on any box that has
# a driver at all.
#
# So: run the verify script deploy.sh really sends, on a fake box with a
# driver-only toolset. Its nvidia-smi lists a process only once something
# has loaded libcuda, and names that process's real pid, so a fallback that
# does not exist leaves the pid check with no subject.
fb="$tmp/fakebox"; mkdir -p "$fb/bin" "$fb/sys"
# A PATH holding only what the script needs, so an nvcc or a torch on the
# machine running this suite cannot take the path under test away from it.
for t in bash sh awk cat tr sed sleep rm touch head; do
  ln -s "$(command -v "$t")" "$fb/sys/$t"
done
cat > "$fb/bin/python" <<FAKE
#!/bin/bash
[ "\$1" = "-c" ] || exit 1
case "\$2" in
  *libcuda*) echo \$\$ > '$fb/ctx.pid'; exec sleep 30 ;;
  *) exit 1 ;;   # no torch, and no gpuqueue to import
esac
FAKE
cat > "$fb/bin/nvidia-smi" <<FAKE
#!/bin/bash
[ -s '$fb/ctx.pid' ] && printf '%s, 104 MiB, python\n' "\$(cat '$fb/ctx.pid')"
exit 0
FAKE
printf '#!/bin/bash\nexit 0\n' > "$fb/bin/gpu-claim"
printf '#!/bin/bash\nexit 0\n' > "$fb/bin/gpuq"
chmod +x "$fb/bin/"*

rm -f "$tmp/scripts/verify.script"
DEPLOY_SSH="$tmp/fake-capture" bash "$repo/deploy.sh" fake-box --verify-only \
  --python "$fb/bin/python" >/dev/null 2>&1 || true
env -i PATH="$fb/sys" HOME="$tmp" bash "$tmp/scripts/verify.script" \
  > "$fb/verify.out" 2>/dev/null
fb_localpids="$(sed -n '/^localpids<<<$/,/^localpids>>>$/p' "$fb/verify.out" | sed '1d;$d')"

check "verify on a driver-only box opens a context through libcuda" \
  "[ -s '$fb/ctx.pid' ]"
check "verify on a driver-only box has a local pid to test" \
  "[ -n \"\$fb_localpids\" ] && [ \"\$fb_localpids\" = \"\$(cat '$fb/ctx.pid' 2>/dev/null)\" ]"
check "verify on a driver-only box cleans up its probe" \
  "! kill -0 \"\$(cat '$fb/ctx.pid' 2>/dev/null || echo 999999)\" 2>/dev/null"

# A box that refuses at probe must not be installed on. This fake reports
# only a 3.10, which is below the floor.
cat > "$tmp/fake-old" <<'FAKE'
#!/usr/bin/env bash
script="$(cat)"
op="$(printf '%s\n' "$script" | sed -n 's/^#gpuq-op:\(.*\)$/\1/p' | head -1)"
printf '%s\n' "$op" >> "$DEPLOY_FAKE_LOG"
case "$op" in
  probe) printf 'py\t/usr/bin/python3\t3.10.6\t1\t0\nsmi\t1\nsupervisor\t1\nprefixwritable\t1\ngit\t1\n' ;;
  *) : ;;
esac
FAKE
chmod +x "$tmp/fake-old"

DEPLOY_FAKE_LOG="$tmp/ops-old.log"; export DEPLOY_FAKE_LOG
: > "$DEPLOY_FAKE_LOG"
DEPLOY_SSH="$tmp/fake-old" bash "$repo/deploy.sh" fake-box >"$tmp/old.out" 2>&1
rc=$?
check "a box below the version floor exits 2" "[ $rc -eq 2 ]"
# Paired with the positive below on purpose: "never installed" is satisfied
# by a deploy.sh that does nothing at all, so on its own it is not a test.
check "a box below the version floor was still probed" \
  "grep -qx 'probe' '$DEPLOY_FAKE_LOG'"
check "a box below the version floor is never installed on" \
  "! grep -qx 'install' '$DEPLOY_FAKE_LOG'"

echo "---"
[ "$fails" -eq 0 ] && echo "all passed" || { echo "$fails failed"; exit 1; }
