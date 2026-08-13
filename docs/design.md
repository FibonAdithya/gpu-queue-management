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
      └──► gpu lane ── admitted against declared VRAM,
                       capped at gpu_max_jobs (default 2)
                          │
                          ▼
                     gpuq-runner
              reap → claim → run → artifacts → commit
```

The CPU default is **4** rather than the core count. Typical CPU jobs here are
BLAS-bound and already thread internally; admitting one per core
oversubscribes and slows everything. Tune per box, and measure before tuning.

The GPU lane admits against capacity rather than a count. A job declares
`--vram-mb`; admission sums the declarations of current holders against the
card's total less a reserve. A job that declares nothing takes the whole
card, which is what makes the change invisible to anything written before
it.

Two dimensions, doing different jobs. Declared VRAM is a **safety** budget:
it is what stops a co-tenant turning into an OOM. `gpu_max_jobs` is a
**latency** budget: VRAM alone would admit sixteen 500 MiB jobs onto an 8 GB
card, all time-slicing, each slower than it would have been queued — and
with independent submitters that cost lands on a stranger.

Both are enforced in the ledger, against every holder of the card. The
runner's own `_capacity("gpu")` still refuses past `gpu_max_jobs`, but only
as a cheap pre-filter that avoids taking the mutex — the authority is
`ledger.acquire`, because a cap the runner applied to its own lane alone
would leave four hand-run `gpu-claim`s on the card and the runner would then
admit two more on top. "With independent submitters that cost lands on a
stranger" is precisely the case where the submitters are not this runner.

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

Four things must be pinned for independent implementations to interoperate:

| | |
|---|---|
| Lock path | fixed directory (`$GPU_CLAIM_DIR`, default `/var/lock/gpu`), file named by GPU UUID |
| Key derivation | the card's UUID, lowercased with any `GPU-`/`MIG-` prefix stripped; fall back to `name-index` where no UUID is reported. gpuq reads it from `nvidia-smi` and never imports torch (see `gpuid.py`); an implementation that reads `torch.cuda.get_device_properties(dev).uuid` gets the bare hex where nvidia-smi gives `GPU-<hex>`, and `gpuid.normalize_gpu_uuid` is what makes those one key rather than two locks on one card |
| Ledger | `<key>.lock.d/<pid>.<token>.json` per holder: `pid`, `usage_pid`, `vram_mb`, `owner`, `cmd`, `started_at`, `key` |
| Mutex | `<key>.lock`, `flock`ed only while reading the ledger and writing one record |

Keying on the UUID rather than the index is not cosmetic. Two processes with
different `CUDA_VISIBLE_DEVICES` mappings both see their card as index 0, so an
index-keyed lock hands them different locks for the same physical GPU.

`flock` guards the accounting, not the card. It is taken for the
milliseconds needed to read the holders, decide, and rename one record into
place — never for the duration of a run. `vram_mb: null` means exclusive:
it fits only into an empty ledger and nothing fits alongside it.

A refusal and a permanent refusal are different answers. "The card is full"
clears when a holder exits, so waiting for it is reasonable and `--wait`
polls. A declaration larger than the whole card never fits however empty the
ledger gets, so it is refused up front rather than waited on: `gpu-claim`
exits 69 (unavailable) rather than 75 (try again later), and the runner fails
such a job instead of leaving it pending. An implementation that reports the
permanent case as ordinary busyness gives its callers a loop with no exit.

One record per holder rather than one document listing them, because the
property that matters when something is stuck is that `ls` shows who is on
the card and `rm` clears one wedged holder. A shared mutated document gives
both up exactly then, since a torn write blinds every participant at once.

Release is symmetric to acquire, and it is not optional: a holder removes
its own record when it is done. `gpu_claim` does this in a `finally`, so a
normal exit, an exception and a signal all release the same way. This has
to be explicit because nothing else will do it — a live pid is a live
holder as far as the ledger is concerned, so a process that keeps running
after it is done with the card, but never removes its record, keeps its
declared VRAM off the ledger for everyone else until it exits. Records
left behind by a pid that is already dead are a different case, covered
by cleanup rather than protocol: `release_stale` clears those every poll,
which is recovery from a crash, not a substitute for a holder releasing
itself.

Enforcement stays advisory, with two additions. Preflight refuses to start
when it finds a CUDA process no live record accounts for. And a watchdog on
the reaper's sweep kills a holder using more than it declared, on two
consecutive samples. Neither prevents an overage — the victim OOMs in
milliseconds and conviction takes up to two sweeps. What they convert is an
anonymous CUDA OOM into a named one.

### Pinning the job to the card

Holding the lock says which card is yours. A gpu-lane job is also **told**,
via `CUDA_VISIBLE_DEVICES` set to the card's uuid, by both the runner and
`gpu-claim`. cpu-lane jobs are left alone: they never took the card.

Two things this buys. The allocation becomes binding rather than advisory — a
pinned process cannot see, and so cannot accidentally take, a card it was not
given. And the queue becomes usable by consumers that *refuse to guess* a
device: a trainer resolving `device: auto` under a strict policy has no way to
know which card it was handed unless something says so, and before this it
could not run under the queue at all.

The value is nvidia-smi's own uuid spelling — not `gpu_key`'s normalized form,
and not an index. `gpu_key` is lowercased and prefix-stripped because it names
a lock file; the driver resolves the string as the driver spells it. An index
is wrong here for the same reason the lock is not keyed on one: it is not
stable under a remap, so pinning `0` re-introduces the guess.

On a box whose driver reports no uuid, the name-index fallback is still a
usable lock key but is not something the driver can resolve. Nothing is
pinned, a warning is logged, and jobs run exactly as they did before.

## Failure handling

| Failure | Response |
|---|---|
| Job exits non-zero | → `failed/`, stderr tail captured into the spec so a consumer reads it without ssh |
| Runner dies mid-job | supervisor restarts; reaper requeues once via `attempts`, then fails |
| Wall-clock exceeded | watchdog kills, marks failed, no retry — a hung job is a bug, not a transient |
| CUDA OOM | a configuration error, not retried — except once, when a co-tenant was convicted of overuse; the convicted holder itself is never a victim |
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
- **Hard per-process VRAM caps.** MPS (`CUDA_MPS_PINNED_DEVICE_MEM_LIMIT`)
  or MIG would prevent an overage rather than convict it. MPS needs a
  daemon; MIG is unavailable on consumer cards; and injecting
  `torch.cuda.set_per_process_memory_fraction` would end this tool's
  assuming nothing about what it runs.
- **Compute or SM-share accounting.** No portable way to declare or measure
  it. `gpu_max_jobs` is the crude substitute.
