# Capacity-based GPU admission — Design

Date: 2026-08-10

Let a job declare how much VRAM it needs, and admit GPU work against the
card's capacity instead of against a count of one. Independent submitters
share the card. The advisory lock stops being held for the duration of a run
and becomes a short-lived mutex over a ledger of who holds what.

Resolves issue #8.

## Purpose

The GPU lane admits exactly one job at a time. For a job that needs the card
that is correct. For a small one it serializes work that would run
concurrently: a WGAN-GP run measured on the RTX 4060 box on 2026-08-10 used
243 MiB of 8188 (3.0%) and held the card at 15% utilization, because its
kernels finish faster than they can be launched. Two such jobs on the same
card reached 62% utilization at 483 MiB total.

One thing that measurement does *not* show, and this design is built around:
**VRAM was never the binding constraint.** The idle time was kernel-launch
latency and host synchronisation, on an axis with 97% headroom. VRAM is the
axis on which sharing is *unsafe* — it is what turns a co-tenant into an OOM
— but it does not predict whether two jobs will overlap usefully. Two
compute-saturated 500 MiB jobs both pass a VRAM check and then time-slice:
identical throughput, double the latency for both.

So capacity has two dimensions here, and they do different jobs. Declared
VRAM is a **safety** budget, and is what §3 and §5 are about. A separate
`gpu_max_jobs` count is a **latency** budget, and is why §4 keeps a slot
limit at all rather than admitting everything that fits.

The issue recorded a counterargument: a consumer can already pack several
cells into one job that fans out itself, so "consumers handle their own
fan-out" would be a defensible boundary. That is true for one submitter's
sweep and it is not what this design serves. The chosen requirement is
**independent submitters** — two unrelated agents each submitting a small GPU
job, both admitted — and packing cannot deliver that, because the two jobs
have no common parent to pack them into.

## 1. What today prevents this

Three mechanisms enforce exclusivity, not one. Issue #8 names only the first,
which is the least of them.

| | |
|---|---|
| `runner.py:227` | `limit = self.cfg.cpu_slots if lane == "cpu" else 1` — a literal, trivially replaced. |
| `claim.py:98` | `gpu_claim` takes `LOCK_EX` on a GPU-UUID-keyed file and **holds it for the whole run**, writing one `<key>.lock.json` describing *the* holder. A second job gets `ClaimBusy` regardless of any slot count. |
| `preflight.py:86` | Refuses to start on *any* foreign CUDA process. It passes today between concurrent runner jobs only because they are `ps` descendants of the runner — incidental, not designed. |

The second is the load-bearing one, and it is also the public contract:
`docs/design.md` pins the lock path, key derivation and claim-file shape as
"three things that must be pinned for independent implementations to
interoperate", and `gpu-claim` is advertised as usable directly. Capacity
accounting therefore cannot be a runner-internal detail. It has to live in
the protocol, with `gpu-claim` and the runner as equal participants.

## 2. The ledger

`flock` cannot be a counting semaphore, so it changes role. `<key>.lock`
remains, and is held only for the milliseconds needed to read the current
holders, decide, and write a record. It guards the accounting, not the card.

Holders live one file per holder under `<key>.lock.d/`, named
`<pid>.<token>.json` with a 6-hex-digit token, because the runner holds
several records at once and they would otherwise collide on pid.

```json
{
  "pid": 4213,
  "usage_pid": 51188,
  "vram_mb": 512,
  "owner": "gpuq:wgan-20260810T142211Z-a1b2c3",
  "cmd": ["python", "-m", "src.train"],
  "started_at": "2026-08-10T14:22:11Z",
  "key": "GPU-abc123..."
}
```

One file per holder rather than one document listing all of them, because
`docs/design.md`'s stated property is that state is legible to `ls` and
repairable with ordinary shell commands. A shared mutated document gives that
up exactly when something is stuck: a torn write blinds every participant at
once, and no single holder can be cleared with `rm`.

**`pid`** is the process whose liveness governs the record — the runner for a
queued job, the `gpu-claim` process for a direct run. `release_stale` keeps
working on it unchanged.

**`usage_pid`** is the process tree charged against this record, and is the
only thing §5 and §6 attribute VRAM through. `gpu-claim` sets it to its own
pid at acquire. The runner sets it to `null` at acquire and fills it in after
`_launch`, because the job's pid does not exist yet when the card is taken.
A record with `usage_pid: null` holds a *reservation* and owns no processes —
which is exactly right, since a job that has not launched has no CUDA
processes to own.

**`vram_mb: null` means exclusive**, and is the default. An exclusive record
fits only into an empty ledger, and nothing fits alongside one. That single
rule is what makes today's `gpu-claim -- python ...` and every undeclared
`gpuq submit` behave exactly as they do now.

Acquire, entirely under the mutex: read every record, drop those whose `pid`
is dead, sum `vram_mb`, admit if `sum + mine <= capacity - reserve`, write
the record, unlock. Release unlinks the record.

`ClaimBusy` gains a message worth reading — free MiB, requested MiB, and the
holders by pid and owner — replacing today's single-holder sentence.

### One attribution function

Three call sites need to answer "which holder owns this CUDA process?":
preflight (§5), the orphan reaper, and the watchdog (§6). They get one
function, in the ledger module, so they cannot disagree:

```
attribute(apps) -> ({record: [app]}, unledgered: [app])
```

A CUDA pid belongs to record R when it is in `{R.usage_pid}` or among its
`_descendants`. Everything else is unledgered. Using the process tree rather
than the process group covers both shapes: the runner starts jobs with
`start_new_session=True`, and `gpu-claim`'s child inherits its group only
when `gpu-claim` happens to be a group leader.

## 3. Declaration surface

`JobSpec` gains `vram_mb: int | None = None`. `validate()` requires a
positive int when set.

```
gpuq submit --vram-mb 512 --lane gpu --project p --commit c --branch b -- python -m src.train
gpu-claim --vram-mb 512 -- python -m src.train
```

`gpuq` is a client and never loads the runner config, so it cannot range-check
the declaration at submit time. A declaration larger than `capacity - reserve`
is caught at admission and **fails the job**, rather than leaving it pending
forever — mirroring the existing "no usable GPU: do not queue forever" rule
in `_take_card`.

Four keys under `[queue]`:

| Key | Default | |
|---|---|---|
| `gpu_vram_mb` | `nvidia-smi` reported total | Capacity override for boxes where the query is unavailable or wrong. |
| `gpu_vram_reserve_mb` | 512 | Headroom held back from admission. |
| `gpu_max_jobs` | 2 | Latency budget; see §4. |
| `enforce_vram` | `true` | Off switch for §6. |

`gpu-claim --status` already prints `list_claims()` and needs no new flag; it
prints the ledger.

A declaration is measured the same way it is enforced: `nvidia-smi`'s per-pid
used memory, which includes the ~250 MiB CUDA context and PyTorch's caching
allocator high-water mark rather than live tensor bytes. That is the number
the 243 MiB measurement came from, and it is what §6 compares against. A
declaration derived from `torch.cuda.max_memory_allocated()` will be too
small; `docs/deploying.md` should say so.

## 4. Admission

`_capacity("gpu")` returns `gpu_max_jobs - in_lane` instead of `1 - in_lane`.
`_take_card` forwards `spec.vram_mb` into `gpu_claim`, and `ClaimBusy`
continues to mean "wait", not "fail".

`gpu_max_jobs` defaults to 2 because that is the extent of what has been
measured — 15% utilization to 62% — and extrapolating past a single data
point is how a shared box acquires a scheduler nobody trusts. Without this
cap, VRAM alone would admit sixteen 500 MiB jobs onto an 8 GB card, all of
them time-slicing, each one slower than it would have been queued. With
independent submitters that cost lands on a stranger. A box that measures
more can raise it.

## 5. Preflight: foreign becomes unledgered

Preflight's rule changes from "refuse on any foreign CUDA process" to
**"refuse on any unledgered CUDA process"**, using §2's attribution. The
protection against an accidental direct run is unchanged; it simply stops
treating a legitimate co-tenant as an intruder.

That attribution rested on a bug this design surfaced, which is **already
fixed** in #12, ahead of this work. `own_pids()` exempted each live record's
pid but expanded `_descendants()` only for the caller's own pid, so a direct
`gpu-claim` run's CUDA process — the *child* of the recorded pid — was
covered by nothing, and `reaper.kill_orphan_cuda` `SIGKILL`ed it within
`orphan_cuda_interval_s`. It was reproduced before being fixed; the test
double-forks so the pair sits outside pytest's process tree, and reads
`/proc` State rather than `claim.pid_alive`, which reports a zombie as alive.

So this work starts from a `main` where a claim's pid already stands for its
whole process tree. §2's `attribute()` narrows that from "which pids does the
protocol account for" to "which *record* owns this pid" — the finer question
the watchdog needs, and one nothing currently answers.

## 6. The VRAM watchdog

Enforcement, chosen over trusting declarations, because with independent
submitters an under-declared job kills a *stranger's* six-hour run and
nothing in the spec can afterwards say who was at fault. That is
`docs/design.md`'s failure mode 1 — "two runs that are each mysteriously
slow, neither says what actually happened" — which is the failure this whole
system exists to prevent.

It rides the sweep that already runs `nvidia-smi` on `orphan_cuda_interval_s`
(default 60), so it costs one arithmetic pass over data already fetched. For
each ledger record, sum `used_mb` over its attributed pids and compare against
`vram_mb`. Records with `vram_mb: null` are never over — they declared the
whole card.

A record over its declaration on **two consecutive sweeps** is killed, its
process group first `SIGTERM`ed then `SIGKILL`ed as `_kill_group` already
does. Two sweeps rather than one because the caching allocator's high-water
mark moves in steps, and a single sample over the line is not evidence of a
persistent overage.

**This is attribution, not prevention, and the spec should not pretend
otherwise.** The victim OOMs in milliseconds; the watchdog convicts in up to
two minutes. What it buys is that the failure is *legible* afterwards — the
over-user dies with `declared 512 MiB, using 3070 MiB` in its error, instead
of two jobs sharing a bare CUDA OOM between them. Real prevention needs an
MPS daemon or MIG (§ Out of scope).

Killed jobs are marked failed and **not** requeued: exceeding your own
declaration is a configuration error, the same class as an OOM, and
`docs/design.md` already says that class is never retried blindly.

Two guards. If `compute_apps()` returns `None`, nothing is enforced — same
posture as preflight's "proceeding on the advisory lock alone". And if
*every* visible CUDA process is unattributable while at least one record
carries a `usage_pid`, attribution is broken rather than everyone being an
intruder: log it and enforce nothing. Under MPS, `nvidia-smi` reports the MPS
server rather than its clients, and that guard is what stops the reaper
mistaking every client for an orphan and killing the box's work.

## 7. Retrying the victim

With sharing, "a CUDA OOM is your own configuration error" is only true if
the two cases can be told apart. They can, in the one case that matters: a
job that OOMs while the watchdog has convicted a *different* holder since
that job started was not misconfigured — it was hit.

The runner keeps the timestamp of each conviction. In `_settle`, a
`result.oom` failure whose job started before a conviction of another record
is requeued through the existing `attempts` counter, which already bounds it
to one retry, with the reason recorded in `spec.error`. Every other OOM
behaves exactly as today.

This section is separable. Cutting it leaves the design correct and the
victim's OOM merely unexplained; it is the difference between blame being
assignable and blame being acted on.

## 8. Migration

The upgrade window is genuinely unsafe in both directions, and neither
direction is fixed by code alone.

An **old `gpu-claim`** holds `LOCK_EX` for its entire run and would block the
new ledger mutex indefinitely. Acquire therefore uses a bounded `flock` wait
(10s) and reports "an older gpu-claim is holding this card exclusively"
rather than hanging.

An **old runner** with new `gpu-claim` users is worse: it globs `*.lock.json`,
finds no records inside `<key>.lock.d/`, and its `kill_orphan_cuda` kills
legitimate co-tenants. There is no defence available to the new code here.
What bounds it is the property the README already argues for — one shared
installation, never vendored — plus `bootstrap.sh` upgrading the package and
restarting the runner in one pass. `docs/deploying.md` gains a line saying
the upgrade must not be done piecemeal.

An old *reader* is safe: seeing zero records makes preflight exempt fewer
pids and refuse to start, which fails closed.

`from_dict` rejects unknown fields, so a spec carrying `vram_mb` read by an
old runner raises `SpecError`. That is the right failure — loud beats an old
runner silently ignoring a declaration and admitting on wrong numbers.

## 9. Known failure modes

| | |
|---|---|
| Compute contention between honest declarers | Not addressed by VRAM accounting at all. `gpu_max_jobs` bounds it; nothing measures it. |
| Watchdog convicts too late to save the victim | Inherent (§6). §7 recovers the victim; nothing prevents the loss. |
| MPS in use | Attribution collapses; §6's second guard disables enforcement rather than killing indiscriminately. |
| A job that spikes past its declaration for under two sweeps | Not convicted. May still have OOMed a co-tenant, which then looks like the co-tenant's own fault. |
| Fragmentation | Two 4000 MiB declarations fit an 8188 MiB card on paper and may still OOM. `gpu_vram_reserve_mb` is the only lever. |
| Reservation without usage | A record between acquire and launch holds VRAM budget it is not using. Bounded by how long `_prepare_workdir` takes. |
| A convicted direct `gpu-claim` holder is never told why | A queued job gets `_describe_failure`, which writes "declared 512 MiB, actually using 3070" into `spec.error` where `gpuq show` prints it. A direct `gpu-claim -- ...` holder has no equivalent: its command is SIGTERMed then SIGKILLed, the user sees exit 137 and an empty stderr, and the conviction reason exists only in the runner's log — which is not where that user is looking. Accepted rather than fixed: a new user-facing channel for one case is more machinery than the case is worth. Documented so the next person to see an unexplained 137 on this box knows where to look. |

## 10. Verification

- Ledger arithmetic: fits, does not fit, exclusive into empty, exclusive
  blocks a declared claim, declared claim blocked by an exclusive holder.
- Two processes acquiring concurrently: exactly one wins, and the loser's
  `ClaimBusy` names the holder.
- `release_stale` frees records whose `pid` is dead; records with a live
  `pid` and a dead `usage_pid` are charged nothing but not released.
- Attribution maps a job's dataloader children to the right record, and an
  unrelated CUDA process to unledgered.
- Watchdog: over on two consecutive sweeps kills; over on one does not;
  `vram_mb: null` is never over; both §6 guards disable enforcement.
- Preflight admits a ledgered co-tenant and refuses an unledgered process.
- The §5 reaper bug: a direct `gpu-claim` child survives a sweep. Already
  covered by the tests landed with #12, which must keep passing.
- Backward compatibility: a job with no `--vram-mb` is admitted alone and
  excludes everything else, byte-for-byte today's behaviour.

## Out of scope

- **Hard per-process VRAM caps.** MPS
  (`CUDA_MPS_PINNED_DEVICE_MEM_LIMIT`) or MIG would prevent overage rather
  than convict it. MPS needs a daemon on the box; MIG is unavailable on
  consumer cards. gpuq assumes nothing about what it runs, and injecting
  `torch.cuda.set_per_process_memory_fraction` would end that.
- **Compute or SM-share accounting.** No portable way to declare or measure
  it. `gpu_max_jobs` is the crude substitute.
- **Predicting speedup.** Nothing here tells a submitter whether sharing will
  help. That stays a measurement the submitter makes.
- **Multi-GPU and multi-host**, unchanged from `docs/design.md`.
