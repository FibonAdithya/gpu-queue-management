# Design

Date: 2026-08-05

## Problem

A single GPU box is shared by several agents or people. Two failure modes
follow:

1. **Silent contention.** Two processes take the card at once. On an 8GB
   consumer card this surfaces as a CUDA OOM some way into a run, or as two
   runs that are each mysteriously slow. Neither says what actually happened.
2. **Blocked workers.** If the only discipline is "wait your turn on ssh",
   whoever is waiting can do nothing else, and if their session dies the queued
   work is simply lost.

The shape that motivates it: several agents working in parallel, each wanting
to fetch data, profile it, and train a model ladder, against one consumer card
with 8GB of VRAM and many CPU cores. The CPU work parallelizes freely; the
training does not.

## Constraints

- The target box is an **unprivileged container** (a hosted PyTorch image). No
  Docker-in-Docker, no kernel modules, no sysctls. Long-running processes are
  managed by **supervisor**.
- The box is **ephemeral**. It may be destroyed and rebuilt; nothing on it may
  be hand-made, and host identity must be a single variable.
- Consumers may be **agents, not people**. Interfaces must be inspectable
  without a running service and repairable with ordinary shell commands.

## Two lanes

Not all work needs the card. Data fetching and CPU-bound analysis are
parallelizable; training is not. One serial queue wastes the cores, and one
parallel pool thrashes the VRAM.

```
producer (agent / human)
      │  writes job spec, gets id, moves on
      ▼
$QUEUE_ROOT/pending/<id>.json
      │
      ├──► cpu lane ── N concurrent (default 4)
      └──► gpu lane ── 1 slot, behind gpu-claim
                          │
                          ▼
                     gpuq-runner
              reap → claim → run → artifacts → commit
```

The CPU default is **4** rather than the core count. Typical CPU jobs here are
BLAS-bound and already thread internally; admitting one per core
oversubscribes and slows everything. Tune per box, and measure before tuning.

## Queue

A directory tree, with states as subdirectories and transitions by atomic
`rename(2)`:

```
$QUEUE_ROOT/
  pending/    <id>.json
  running/    <id>.json
  done/       <id>.json
  failed/     <id>.json
  logs/       <id>.{out,err}
```

No database, no daemon dependency for inspection. The entire system state is
legible to `ls`, and a stuck job is repaired with `mv`. A queue that needs a
running service to inspect becomes opaque exactly when something has gone
wrong — which is when you need to see it.

`rename(2)` within a filesystem is atomic, so a job is in exactly one state at
any instant and two runner threads cannot both claim it.

### Job spec

```json
{
  "id": "myproject-v0-train-01",
  "lane": "gpu",
  "project": "myproject",
  "commit": "a1b2c3d",
  "branch": "experiment/v0",
  "cmd": ["python", "-m", "src.train",
          "--config", "configs/v0.yaml"],
  "artifacts": ["runs/v0/summary.json",
                "runs/v0/run_config.yaml"],
  "timeout_s": 21600,
  "attempts": 0,
  "dedupe_key": "myproject:v0:a1b2c3d"
}
```

`commit` is pinned, not merely `branch`. The runner checks out that exact tree,
so a returned result is attributable to a configuration that can be read back.
A branch reference alone lets the tree move under a queued job and produce a
number nobody can reproduce.

`dedupe_key` makes resubmission idempotent. Submitting an identical job while
one is pending or running is a no-op returning the existing id.

## Runner

One supervisor-managed process per box: the sole launcher of queued work.

Loop: reap → poll `pending/` → admit what the lanes allow → move to `running/`
→ execute → collect artifacts → move to `done/` or `failed/`.

**Workers never touch git.** Concurrent CPU jobs committing into one checkout
would corrupt the index. Workers write artifacts to disk only; the runner's
single main loop performs every git operation between polls. Repository
mutation is serialized by construction rather than by discipline.

Each project the runner serves is declared in configuration:

```toml
[queue]
root = "/workspace/queue"
cpu_slots = 4

[project.myproject]
remote   = "git@github.com:you/myproject.git"
checkout = "/workspace/checkouts/myproject"
venv     = "/workspace/checkouts/myproject/.venv"
commit_artifacts = true
```

The runner owns its checkouts. Nothing else git-operates in them; that is the
property which stops a shared box accumulating drifted working copies.

## Reaper

Runs on every poll of the runner loop, because the state it recovers from —
a runner that died — is one where nothing else will happen to trigger it:

- Read each claim file; if its pid is dead, release the claim and log it.
- Kill CUDA processes that no live job owns.
- Remove `.part` files and partial output directories.
- Requeue `running/` jobs whose pid is gone — **once**, tracked by `attempts`.

Plus a per-job wall-clock watchdog.

Reaping lives in the runner, not in a supervising agent, because it has to run
when nothing else is alive. That is precisely when a leaked job needs reaping —
which is also why it cannot be triggered by job completions: an idle runner
would then never reap, in exactly the situation that calls for it.

The steps are split by cost. Releasing claims, requeueing abandoned jobs and
clearing debris are file operations and run every poll. Killing orphaned CUDA
processes needs `nvidia-smi` and a walk of the process tree, and is a safety
net rather than a recovery path, so it runs at most once per
`orphan_cuda_interval_s` (default 60) and stays out of the loop that gates job
admission.

The requeue-once rule is load-bearing. Without an attempt counter, a
crash-looping job occupies the only card indefinitely.

## Lock protocol

`gpu-claim` is usable directly and is what the GPU lane uses internally:

```
gpu-claim -- python -m src.train --config ...
```

Three things must be pinned for independent implementations to interoperate:

| | |
|---|---|
| Lock path | fixed directory (`$GPU_CLAIM_DIR`, default `/var/lock/gpu`), file named by GPU UUID |
| Key derivation | `torch.cuda.get_device_properties(dev).uuid`; fall back to `name-index` on builds without `.uuid` |
| Claim file | JSON alongside the lock: `pid`, `owner`, `cmd`, `started_at` |

Keying on the UUID rather than the index is not cosmetic. Two processes with
different `CUDA_VISIBLE_DEVICES` mappings both see their card as index 0, so an
index-keyed lock hands them different locks for the same physical GPU.

Enforcement stays **advisory** — `flock` cannot be otherwise between
unprivileged processes — with one addition. Preflight queries the card for
foreign CUDA processes and refuses to start when it finds any, naming the pid
and command. This cannot stop a determined direct run. It converts accidental
contention into a fast, readable failure instead of an OOM half an hour in.

## Failure handling

| Failure | Response |
|---|---|
| Job exits non-zero | → `failed/`, stderr tail captured into the spec so a consumer reads it without ssh |
| Runner dies mid-job | supervisor restarts; reaper requeues once via `attempts`, then fails |
| Wall-clock exceeded | watchdog kills, marks failed, no retry — a hung job is a bug, not a transient |
| CUDA OOM | detected distinctly and never retried blindly; it is a configuration error to surface |
| Duplicate submission | deduplicated on `dedupe_key`, returns the existing id |
| Box destroyed | everything under `$QUEUE_ROOT` is lost; committed artifacts survive in git |

## Bootstrap

`bootstrap.sh` takes a bare box to a working runner, idempotently: install the
package, create `$QUEUE_ROOT`, clone declared project checkouts, write the
supervisor program file, start the runner.

The supervisor configuration ships in this repo rather than being added by
hand. That is what makes a rebuilt box identical rather than similar.

Host identity lives in one variable so that rebuilding is an ssh-target edit
plus a bootstrap run.

## Not in scope

- **Multi-GPU scheduling.** The lane abstraction would extend to it; nothing
  here anticipates it. One card, one box.
- **Multi-host scheduling.** `flock` is host-local by nature. Coordinating
  across boxes needs a different mechanism and a different design.
- **Durable artifact storage.** Consumers commit what they want to keep. A
  project may name a separate results repository, which is how a box holds a
  read-only key for code and a write key that reaches nothing else; beyond
  that, retention is the consumer's problem.
- **Authentication.** Anyone who can write to `$QUEUE_ROOT` can queue work.
  The box's ssh access is the security boundary.
