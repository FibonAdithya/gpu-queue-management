#!/usr/bin/env bash
# deploy.sh — take a named ssh box to a verified gpuq installation.
#
# `bootstrap.sh` owns the on-box half and is unchanged by this script.
# What this owns is everything an operator had to do by hand around it:
# reading the box before installing, deciding which of its several
# interpreters the runner belongs in, getting the code across, and then
# establishing what the box actually gives you -- which is not the same
# question as whether the install succeeded.
#
# The last part is why this exists rather than a README paragraph. A box
# can install perfectly and still not enforce anything, and two of the
# three ways that happens are completely silent from the outside (see
# docs/deploying.md). Those are checks, not prose.
#
# Usage:
#   ./deploy.sh <ssh-alias> [options]
#
# Sourcing this file defines its functions and runs nothing, which is how
# tests/test_deploy.sh exercises the decisions without a box.
set -euo pipefail

# Lowest Python that has `tomllib` in the stdlib. There is no `tomli`
# fallback, so this is a hard floor rather than a preference.
PY_FLOOR="3.11"

BOX=""
PREFIX="/workspace"
REF=""
PYTHON_OVERRIDE=""
MODE="full"           # full | probe | verify
SLOW=0
DRY_RUN=0

usage() {
  cat <<'EOF'
deploy.sh — take a named ssh box to a verified gpuq installation.

  ./deploy.sh <ssh-alias> [options]

Phases, in order. Each is safe to re-run.

  probe    Read the box and change nothing. Chooses the interpreter,
           and refuses here if the box cannot host the runner at all.
  install  Get the code across, then run bootstrap.sh on the box.
  verify   Establish what the box actually gives you: capacity
           discovery, the preflight guard, and whether nvidia-smi's
           pids mean anything locally. On a box with no GPU, that the
           runner is up and the queue answers.

Options:
  --probe-only        Run probe, print the report, stop. Changes nothing.
  --verify-only       Skip probe and install; verify an existing install.
  --python PATH       Interpreter to install into. Default: chosen by probe.
  --prefix PATH       GPUQ_PREFIX on the box. Default: /workspace.
  --ref REF           Branch or tag on origin. Default: current branch.
  --slow              Also run the 140s orphan-sweep test, which is
                      otherwise derived from the pid-namespace check.
  --dry-run           Print what each phase would do.
  -h, --help          This.

Exit status: 0 verified, 1 a check failed, 2 the box was refused at probe.
EOF
}

# ---------------------------------------------------------------- decisions
#
# Plain functions over text. No ssh, no state -- so they are testable
# directly, and so the rules they encode are visible in one place rather
# than spread through the phases that apply them.

# True when version $1 is >= version $2.
ver_ge() {
  [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -1)" = "$2" ]
}

# Choose the interpreter to install the runner into.
#
# Reads probe lines on stdin; prints the chosen path, or fails naming the
# floor. The rule, in order: it must be >= PY_FLOOR; one with pip beats
# one without; and among those an interpreter holding torch wins -- that
# is where the ML stack lives, gpuq has no dependencies so it cannot
# conflict with it, and jobs that say `-- python train.py` need it on
# PATH. Ties break to the highest version.
#
# pip is a preference and not a requirement: bootstrap.sh builds a venv
# from an interpreter that lacks it. It still outranks torch, because that
# venv would not hold torch either, and installing where pip already is
# changes nothing else on the box.
#
# Deliberately not "whatever is called python3": on the boxes this targets
# that is routinely the wrong answer, and picking it is the mistake this
# function exists to stop someone repeating at 2am.
choose_python() {
  local best="" best_ver="" best_rank=-1
  local tag path ver haspip hastorch rank
  while IFS=$'\t' read -r tag path ver haspip hastorch; do
    [ "$tag" = "py" ] || continue
    ver_ge "$ver" "$PY_FLOOR" || continue
    # pip beats torch beats version; version only breaks ties within a
    # class.
    rank=0
    [ "$hastorch" = "1" ] && rank=1
    [ "$haspip" = "1" ] && rank=$((rank + 2))
    if [ "$rank" -gt "$best_rank" ] ||
       { [ "$rank" -eq "$best_rank" ] && ver_ge "$ver" "$best_ver"; }; then
      best="$path"; best_ver="$ver"; best_rank="$rank"
    fi
  done
  if [ -z "$best" ]; then
    echo "deploy: no interpreter on this box is Python $PY_FLOOR+." >&2
    echo "        gpuq needs $PY_FLOOR for stdlib tomllib; there is no fallback." >&2
    echo "        Install one, or point --python at it if the probe missed it." >&2
    return 1
  fi
  printf '%s\n' "$best"
}

# What class of box is nvidia-smi describing?
#
# $1 is the raw --query-compute-apps output, $2 the pids that were found
# in /proc. Listing processes is NOT enough to call preflight a real
# guard: an unprivileged container can be shown the host's pid namespace,
# and then `ledger.attribute` is comparing two sets that never intersect,
# so every CUDA process on the box reads as unledgered -- a holder's own
# included. See docs/deploying.md, "nvidia-smi may report pids from the
# host's namespace".
classify_gpu_pids() {
  local apps="$1" local_pids="$2" pid
  case "$apps" in
    *"[Not Supported]"*) printf 'unsupported\n'; return 0 ;;
  esac
  [ -n "$(printf '%s' "$apps" | tr -d '[:space:]')" ] || { printf 'idle\n'; return 0; }
  # Any single unresolvable pid settles it. A box where one process
  # happens to resolve and another does not is still a box where
  # attribution is unreliable, and "mostly translatable" is not a
  # property anything here can be built on.
  while read -r pid; do
    [ -n "$pid" ] || continue
    printf '%s\n' "$local_pids" | grep -qx "$pid" || { printf 'host-pids\n'; return 0; }
  done < <(printf '%s\n' "$apps" | sed -n 's/^[[:space:]]*\([0-9][0-9]*\).*/\1/p')
  printf 'local-pids\n'
}

# One sentence for what this box gives you.
#
# $1 pid class, $2 total VRAM in MiB or "none", $3 whether nvidia-smi exists.
# The two roads to "one GPU job at a time" are different faults with the
# same operational consequence, and saying so is more use than naming the
# mechanism here -- the mechanism is in the check rows above it.
verdict() {
  local pidclass="$1" total="$2" has_smi="$3"
  if [ "$has_smi" != "1" ] || [ "$pidclass" = "nosmi" ]; then
    printf 'usable, no GPU lane (nvidia-smi absent; GPU jobs are refused by design)\n'
    return 0
  fi
  if [ "$total" = "none" ]; then
    printf 'usable, one GPU job at a time (capacity discovery failed)\n'
    return 0
  fi
  case "$pidclass" in
    host-pids)
      printf 'usable, one GPU job at a time (nvidia-smi reports host pids)\n' ;;
    unsupported)
      printf 'usable, one GPU job at a time (no preflight guard; advisory lock only)\n' ;;
    local-pids)
      printf 'usable, GPU sharing works\n' ;;
    # idle, or anything unrecognised: nothing was measured, and the good
    # verdict is the one that must be earned rather than defaulted to.
    *)
      printf 'usable, GPU sharing unverified (no CUDA process to test pids with)\n' ;;
  esac
}

# Whether the box will get the commit you are looking at.
#
# The box clones from origin, so it can only ever check out what origin
# has. $1 is the local HEAD, $2 the sha origin resolves the deployed ref
# to. Empty $2 means origin has no such ref at all, which is a refusal
# rather than a warning -- the clone would succeed and the checkout would
# then fail with "unknown revision", after bootstrap has been skipped.
#
# Silence means agreement. Anything else is printed for the caller to show.
ref_warning() {
  local local_sha="$1" remote_sha="$2"
  if [ -z "$remote_sha" ]; then
    echo "deploy: origin has no such ref. The box clones from origin, so it" >&2
    echo "        cannot check out a commit that only exists on your machine." >&2
    return 1
  fi
  [ "$local_sha" = "$remote_sha" ] && return 0
  printf 'local %s is not what origin has (%s) — the box gets origin\n' \
    "$local_sha" "$remote_sha"
  return 0
}

# ---------------------------------------------------------------- transport

# Run a script on the box. The op name is prepended as a marker comment so
# that a test fake can tell the phases apart; ssh ignores it as a comment.
remote() {
  local op="$1"
  local -a ssh_cmd
  read -r -a ssh_cmd <<< "${DEPLOY_SSH:-ssh}"
  { printf '#gpuq-op:%s\n' "$op"; cat; } | "${ssh_cmd[@]}" "$BOX" bash -s
}

say()  { printf '%s\n' "$*" >&2; }
head_() { printf '\n  %-8s %s\n' "$1" "$2" >&2; }
row()  { printf '           %-22s %-20s %s\n' "$1" "$2" "$3" >&2; }

# --------------------------------------------------------------- phase 1

# Which service manager bootstrap is told to use. Decided here and passed
# explicitly, so the probe report and the install cannot disagree.
init_choice() {
  if [ "$P_SUPERVISOR" = "1" ]; then printf 'supervisor\n'
  elif [ "$P_SYSTEMD" = "1" ]; then printf 'systemd\n'
  else printf 'none\n'
  fi
}

PROBE_OUT=""
P_PYTHON=""; P_GPU=""; P_TOTAL="none"; P_SMI=0
P_SUPERVISOR=0; P_SYSTEMD=0; P_WRITABLE=0; P_GIT=0

do_probe() {
  PROBE_OUT="$(remote probe <<EOF
for p in python python3 python3.11 python3.12 python3.13 python3.14 \
         /venv/main/bin/python /opt/conda/bin/python \
         '$PREFIX/venv/bin/python'; do
  command -v "\$p" >/dev/null 2>&1 || continue
  full="\$(command -v "\$p")"
  v="\$("\$full" -c 'import sys;print("%d.%d.%d"%sys.version_info[:3])' 2>/dev/null)" || continue
  "\$full" -c 'import pip' 2>/dev/null && pip=1 || pip=0
  "\$full" -c 'import torch' 2>/dev/null && th=1 || th=0
  printf 'py\t%s\t%s\t%s\t%s\n' "\$full" "\$v" "\$pip" "\$th"
done | sort -u
if command -v nvidia-smi >/dev/null 2>&1; then
  printf 'smi\t1\n'
  printf 'gpu\t%s\t%s\t%s\n' \
    "\$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)" \
    "\$(nvidia-smi --query-gpu=uuid --format=csv,noheader 2>/dev/null | head -1)" \
    "\$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1 | tr -dc '0-9')"
else
  printf 'smi\t0\n'
fi
command -v supervisorctl >/dev/null 2>&1 && printf 'supervisor\t1\n' || printf 'supervisor\t0\n'
# The directory, not the binary: it exists exactly when systemd is the
# running init, and an image can ship systemctl with nothing behind it.
[ -d /run/systemd/system ] && command -v systemctl >/dev/null 2>&1 &&
  printf 'systemd\t1\n' || printf 'systemd\t0\n'
command -v git >/dev/null 2>&1 && printf 'git\t1\n' || printf 'git\t0\n'
if mkdir -p '$PREFIX' 2>/dev/null && touch '$PREFIX/.gpuq-probe' 2>/dev/null; then
  rm -f '$PREFIX/.gpuq-probe'; printf 'prefixwritable\t1\n'
else
  printf 'prefixwritable\t0\n'
fi
EOF
)" || { say "deploy: could not reach '$BOX' over ssh"; return 2; }

  local tag a c
  # The gpu line's third field is the card UUID, which nothing here uses.
  while IFS=$'\t' read -r tag a _ c; do
    case "$tag" in
      smi)            P_SMI="$a" ;;
      gpu)            P_GPU="$a"; P_TOTAL="${c:-none}" ;;
      supervisor)     P_SUPERVISOR="$a" ;;
      systemd)        P_SYSTEMD="$a" ;;
      git)            P_GIT="$a" ;;
      prefixwritable) P_WRITABLE="$a" ;;
    esac
  done <<< "$PROBE_OUT"
  [ -n "$P_TOTAL" ] || P_TOTAL="none"

  head_ probe "$BOX"
  if [ "$P_SMI" = "1" ]; then
    row "card" "$P_GPU" "${P_TOTAL} MiB"
  else
    row "card" "none" "nvidia-smi absent — GPU lane refuses every job"
  fi

  if [ -n "$PYTHON_OVERRIDE" ]; then
    P_PYTHON="$PYTHON_OVERRIDE"
    row "interpreter" "$P_PYTHON" "(--python)"
  else
    P_PYTHON="$(printf '%s\n' "$PROBE_OUT" | choose_python)" || return 2
    local why="highest $PY_FLOOR+"
    printf '%s\n' "$PROBE_OUT" | grep -qP "^py\t\Q$P_PYTHON\E\t[^\t]*\t1\t1$" &&
      why="$PY_FLOOR+, holds torch"
    printf '%s\n' "$PROBE_OUT" | grep -qP "^py\t\Q$P_PYTHON\E\t[^\t]*\t0\t" &&
      why="no pip — bootstrap creates $PREFIX/venv from it"
    row "interpreter" "$P_PYTHON" "($why)"
  fi

  [ "$P_WRITABLE" = "1" ] || {
    say "deploy: '$PREFIX' is not writable on $BOX. Pass --prefix somewhere that is."
    return 2
  }
  row "prefix" "$PREFIX" "writable"
  case "$(init_choice)" in
    supervisor) ;;
    systemd) row "init" "systemd" "no supervisor — installing a systemd unit" ;;
    *) row "init" "none" "WARN — no supervisor or systemd; run gpuq-runner yourself" ;;
  esac
  [ "$P_GIT" = "1" ] || row "git" "absent" "WARN — will stream a tarball"
  return 0
}

# --------------------------------------------------------------- phase 2

REPO_DIR=""
do_install() {
  local url ref here local_sha remote_sha warn
  here="$(dirname "${BASH_SOURCE[0]}")"
  url="$(git -C "$here" remote get-url origin 2>/dev/null || echo "")"
  # HTTPS, because the box is not assumed to hold a key for this repo and
  # this one is public. A box that cannot reach it gets the working tree
  # streamed instead.
  url="${url/git@github.com:/https://github.com/}"
  # The branch, not the local HEAD sha. See ref_warning: a sha default
  # deploys nothing the moment you have an unpushed commit, and does it by
  # failing after the clone rather than before it.
  ref="${REF:-$(git -C "$here" rev-parse --abbrev-ref HEAD 2>/dev/null || echo main)}"
  local_sha="$(git -C "$here" rev-parse HEAD 2>/dev/null || echo "")"
  remote_sha="$(git -C "$here" ls-remote origin "$ref" 2>/dev/null |
                awk 'NR==1{print $1}')"
  REPO_DIR="$PREFIX/gpu-queue-management"

  head_ install "$REPO_DIR"
  if ! warn="$(ref_warning "$local_sha" "$remote_sha")"; then
    say "        (ref: '$ref')"
    return 1
  fi
  [ -n "$warn" ] && row "ref" "$ref" "WARN — $warn"
  if [ "$DRY_RUN" = "1" ]; then
    row "would clone" "$url" "@ $ref"
    row "would run" "bootstrap.sh" "PYTHON=$P_PYTHON GPUQ_PREFIX=$PREFIX"
    return 0
  fi

  local init
  init="$(init_choice)"

  local out
  if ! out="$(remote install <<EOF
set -e
[ -d '$REPO_DIR/.git' ] || git clone --quiet '$url' '$REPO_DIR'
# Check out what was fetched, not a local branch of the same name: on a
# box that already has a clone, that branch is the previous deploy's
# commit, and switching to it would reinstall that.
cd '$REPO_DIR' && git fetch --quiet origin '$ref' && git checkout --quiet --detach FETCH_HEAD
git -C '$REPO_DIR' rev-parse --short HEAD
cd '$REPO_DIR' && PYTHON='$P_PYTHON' GPUQ_PREFIX='$PREFIX' ./bootstrap.sh --init $init 2>&1 | tail -5
EOF
)"; then
    say "deploy: install failed on $BOX"
    printf '%s\n' "$out" >&2
    return 1
  fi
  row "checkout" "$(printf '%s\n' "$out" | head -1)" "$ref"
  if printf '%s\n' "$out" | grep -q 'bootstrap complete'; then
    row "bootstrap" "complete" ""
    # Where the runner actually went. On a box with no pip that is the
    # venv bootstrap built, and verify has to look there.
    local installed
    installed="$(printf '%s\n' "$out" | sed -n 's/^runner python: //p' | tail -1)"
    if [ -n "$installed" ] && [ "$installed" != "$P_PYTHON" ]; then
      P_PYTHON="$installed"
      row "runner python" "$P_PYTHON" "(created by bootstrap)"
    fi
  else
    row "bootstrap" "PROBLEM" "see output below"
    printf '%s\n' "$out" >&2
    return 1
  fi
}

# --------------------------------------------------------------- phase 3

do_verify() {
  local out fails=0
  head_ verify "$BOX"
  out="$(remote verify <<EOF
export PATH='$(dirname "$P_PYTHON")':\$PATH
export GPUQ_CONFIG='$PREFIX/gpuq.toml' GPU_CLAIM_DIR='$PREFIX/lock/gpu'
runner="\$(supervisorctl status gpuq-runner 2>/dev/null | awk '{print \$2}')"
if [ -z "\$runner" ]; then
  # Not under supervisor. A system unit, or a user unit -- which an ssh
  # command line can only ask about with the runtime dir set.
  export XDG_RUNTIME_DIR="\${XDG_RUNTIME_DIR:-/run/user/\$(id -u 2>/dev/null)}"
  if [ "\$(systemctl is-active gpuq-runner 2>/dev/null)" = "active" ] ||
     [ "\$(systemctl --user is-active gpuq-runner 2>/dev/null)" = "active" ]; then
    runner=RUNNING
  fi
fi
printf 'runner\t%s\n' "\$runner"
'$P_PYTHON' -c 'from gpuqueue.gpuid import total_vram_mb; print("total\t%s" % (total_vram_mb() or "none"))' 2>/dev/null || printf 'total\tnone\n'
gpuq list >/dev/null 2>&1 && printf 'gpuqlist\t1\n' || printf 'gpuqlist\t0\n'
# No card: nothing below has a subject, and gpu-claim refuses by design.
if [ '$P_SMI' != 1 ]; then exit 0; fi

# Passive first: if the card already has work on it, the pid question is
# answerable by reading, which costs nothing and touches nobody's job.
apps="\$(nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader 2>/dev/null)"
allocated=0
if [ -z "\$(printf '%s' "\$apps" | tr -d '[:space:]')" ]; then
  # Idle. Allocate something so the checks below have a subject.
  if command -v nvcc >/dev/null 2>&1; then
    cat > /tmp/gpuq-probe.cu <<'CU'
#include <cstdio>
#include <unistd.h>
int main(){void*p=nullptr;if(cudaMalloc(&p,256ull*1024*1024)!=cudaSuccess)return 1;
printf("%d\n",getpid());fflush(stdout);sleep(30);return 0;}
CU
    if nvcc -o /tmp/gpuq-probe /tmp/gpuq-probe.cu >/dev/null 2>&1; then
      /tmp/gpuq-probe >/dev/null 2>&1 &
      PROBE_PID=\$!; allocated=1; sleep 6
    fi
  elif '$P_PYTHON' -c 'import torch' 2>/dev/null; then
    '$P_PYTHON' -c 'import torch,time;x=torch.zeros(64*1024*1024,device="cuda");time.sleep(30)' >/dev/null 2>&1 &
    PROBE_PID=\$!; allocated=1; sleep 10
  fi
  # No toolkit, no torch, or a compile that failed. The driver's own
  # libcuda can still open a context, which is all the pid check needs --
  # without this, a box with only a driver verifies nothing.
  if [ "\$allocated" = "0" ]; then
    '$P_PYTHON' -c 'import ctypes,time;c=ctypes.CDLL("libcuda.so.1");d=ctypes.c_int();x=ctypes.c_void_p();assert c.cuInit(0)==0 and c.cuDeviceGet(ctypes.byref(d),0)==0 and c.cuCtxCreate_v2(ctypes.byref(x),0,d)==0;time.sleep(30)' >/dev/null 2>&1 &
    PROBE_PID=\$!; allocated=1; sleep 6
  fi
  apps="\$(nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader 2>/dev/null)"
fi
printf 'allocated\t%s\n' "\$allocated"
printf 'apps<<<\n%s\napps>>>\n' "\$apps"
# Which of the reported pids this namespace can actually address.
printf 'localpids<<<\n'
printf '%s\n' "\$apps" | sed -n 's/^[[:space:]]*\([0-9][0-9]*\).*/\1/p' | while read -r p; do
  [ -d "/proc/\$p" ] && printf '%s\n' "\$p"
done
printf 'localpids>>>\n'
# Preflight against a card that now has something unclaimed on it.
if [ -n "\$(printf '%s' "\$apps" | tr -d '[:space:]')" ]; then
  gpu-claim -- true >/dev/null 2>&1; printf 'preflight\t%s\n' "\$?"
else
  printf 'preflight\tskip\n'
fi
if [ "\$allocated" = "1" ] && [ -n "\${PROBE_PID:-}" ]; then
  kill "\$PROBE_PID" 2>/dev/null || true
  wait "\$PROBE_PID" 2>/dev/null || true
fi
rm -f /tmp/gpuq-probe /tmp/gpuq-probe.cu
# An idle card with nothing to allocate: claim round-trip is still worth it.
gpu-claim -- true >/dev/null 2>&1 && printf 'claim\t1\n' || printf 'claim\t0\n'
EOF
)" || { say "deploy: verify could not run on $BOX"; return 1; }

  local runner total gpuqlist preflight claim apps localpids
  runner="$(printf '%s\n' "$out"  | sed -n 's/^runner\t//p'   | head -1)"
  total="$(printf '%s\n' "$out"   | sed -n 's/^total\t//p'    | head -1)"
  gpuqlist="$(printf '%s\n' "$out"| sed -n 's/^gpuqlist\t//p' | head -1)"
  preflight="$(printf '%s\n' "$out"|sed -n 's/^preflight\t//p'| head -1)"
  claim="$(printf '%s\n' "$out"   | sed -n 's/^claim\t//p'    | head -1)"
  apps="$(printf '%s\n' "$out" | sed -n '/^apps<<<$/,/^apps>>>$/p' | sed '1d;$d')"
  localpids="$(printf '%s\n' "$out" | sed -n '/^localpids<<<$/,/^localpids>>>$/p' | sed '1d;$d')"

  if [ "$runner" = "RUNNING" ]; then row "runner" "RUNNING" "ok"
  else row "runner" "${runner:-absent}" "FAIL"; fails=$((fails+1)); fi

  if [ "$gpuqlist" = "1" ]; then row "gpuq list" "exit 0" "ok"
  else row "gpuq list" "nonzero" "FAIL"; fails=$((fails+1)); fi

  # A box with no card has no claim to round-trip, no pids to classify and
  # nothing for preflight to refuse. Those are not failures there: the CPU
  # lane needs none of them, and the verdict says the GPU lane is absent.
  if [ "$P_SMI" != "1" ]; then
    row "GPU checks" "n/a" "no nvidia-smi — CPU lane only"
    VERDICT_PIDCLASS="nosmi"; VERDICT_TOTAL="none"
    return "$( [ "$fails" -gt 0 ] && echo 1 || echo 0 )"
  fi

  if [ "$total" != "none" ]; then row "capacity discovery" "${total} MiB" "ok"
  else row "capacity discovery" "none" "WARN — every GPU job admitted exclusively"; fi

  if [ "$claim" = "1" ]; then row "claim round-trip" "exit 0" "ok"
  else row "claim round-trip" "nonzero" "FAIL"; fails=$((fails+1)); fi

  local pidclass
  pidclass="$(classify_gpu_pids "$apps" "$localpids")"
  case "$pidclass" in
    local-pids)  row "pid namespace" "local" "ok" ;;
    host-pids)   row "pid namespace" "HOST PIDS" "WARN — see docs/deploying.md#host-pids" ;;
    unsupported) row "pid namespace" "not supported" "WARN — preflight is advisory only" ;;
    idle)        row "pid namespace" "unknown" "WARN — no CUDA process to test with" ;;
  esac

  case "$preflight" in
    69)   row "preflight guard" "exit 69" "ok — refuses a busy card" ;;
    0)    row "preflight guard" "exit 0" "WARN — did not refuse a busy card" ;;
    skip) row "preflight guard" "untested" "WARN — card idle, nothing to refuse" ;;
    *)    row "preflight guard" "exit $preflight" "WARN — unexpected" ;;
  esac

  # Derived, not measured, unless --slow paid for the measurement. The
  # sweep cannot signal a pid it cannot address, so a host-pid box tells
  # you the answer without spending 140s to watch nothing happen.
  if [ "$SLOW" = "1" ]; then
    row "orphan sweep" "measured" "$(verify_sweep)"
  elif [ "$pidclass" = "host-pids" ]; then
    row "orphan sweep" "inert (derived)" "WARN — pids are unaddressable"
  else
    row "orphan sweep" "not measured" "pass --slow to measure (140s)"
  fi

  VERDICT_PIDCLASS="$pidclass"; VERDICT_TOTAL="$total"
  return "$( [ "$fails" -gt 0 ] && echo 1 || echo 0 )"
}

verify_sweep() {
  local survived
  survived="$(remote sweep <<EOF
export PATH='$(dirname "$P_PYTHON")':\$PATH
if ! command -v nvcc >/dev/null 2>&1; then echo skip; exit 0; fi
cat > /tmp/gpuq-sweep.cu <<'CU'
#include <cstdio>
#include <unistd.h>
int main(){void*p=nullptr;if(cudaMalloc(&p,256ull*1024*1024)!=cudaSuccess)return 1;sleep(150);return 0;}
CU
nvcc -o /tmp/gpuq-sweep /tmp/gpuq-sweep.cu >/dev/null 2>&1 || { echo skip; exit 0; }
/tmp/gpuq-sweep & P=\$!
sleep 140
if kill -0 \$P 2>/dev/null; then echo survived; else echo killed; fi
kill \$P 2>/dev/null || true
rm -f /tmp/gpuq-sweep /tmp/gpuq-sweep.cu
EOF
)"
  case "$survived" in
    survived) printf 'WARN — inert: unclaimed CUDA survived 2 sweeps\n' ;;
    killed)   printf 'ok — swept an unclaimed process\n' ;;
    *)        printf 'skipped — no nvcc to build a probe\n' ;;
  esac
}

# ------------------------------------------------------------------ report

VERDICT_PIDCLASS="idle"; VERDICT_TOTAL="none"

do_report() {
  local v
  v="$(verdict "$VERDICT_PIDCLASS" "$VERDICT_TOTAL" "$P_SMI")"
  printf '\n  %-8s %s\n' "VERDICT" "$v" >&2
  printf '\n  Row for docs/deploying.md — Boxes:\n\n' >&2
  local hw="no GPU"
  [ "$P_SMI" = "1" ] && hw="${P_GPU:-unnamed GPU}, $P_TOTAL MiB"
  # The backticks are Markdown for the docs table, not command substitution.
  # shellcheck disable=SC2016
  printf '| `%s` | %s | `%s` | %s. Verified %s |\n\n' \
    "$BOX" "$hw" "$P_PYTHON" "$v" "$(date +%F)" >&2
}

# -------------------------------------------------------------------- main

main() {
  while [ $# -gt 0 ]; do
    case "$1" in
      -h|--help)     usage; return 0 ;;
      --probe-only)  MODE="probe" ;;
      --verify-only) MODE="verify" ;;
      --slow)        SLOW=1 ;;
      --dry-run)     DRY_RUN=1 ;;
      --python)      PYTHON_OVERRIDE="${2:-}"; shift ;;
      --prefix)      PREFIX="${2:-}"; shift ;;
      --ref)         REF="${2:-}"; shift ;;
      -*)            say "deploy: unknown option: $1"; usage; return 2 ;;
      *)             BOX="$1" ;;
    esac
    shift
  done

  [ -n "$BOX" ] || { say "deploy: name an ssh alias. Try --help."; return 2; }

  if [ "$MODE" = "verify" ]; then
    # Still needs an interpreter path to build the remote PATH from.
    do_probe || return 2
    do_verify || { do_report; return 1; }
    do_report
    return 0
  fi

  do_probe || return 2
  if [ "$MODE" = "probe" ]; then
    printf '\n  %-8s %s\n\n' "" "probe only — nothing was installed." >&2
    return 0
  fi

  do_install || return 1
  do_verify || { do_report; return 1; }
  do_report
  return 0
}

if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
  main "$@"
fi
