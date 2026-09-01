# Cgroup-scoped claims — Design

Date: 2026-09-01

Let a claim cover a process tree that is not the claimant's own, by naming a
cgroup instead of a pid tree. Container workloads become ordinary co-tenants
of the card. Separately, give the orphan sweep a signal its victim can
survive long enough to log, and a record its victim's operator can read.

Resolves issue #24.

## Purpose

The orphan sweep SIGKILLs CUDA processes no live claim accounts for. On
2026-09-01 it killed four CUDA children of a long-lived container,
`tig-pentesting-tig-scorer-1`, and no claim could have covered them:
exemption is by host pid and its descendants (`preflight.py:79-88`), and
`gpu-claim` claims only for its own process tree (`claim.py:322`,
`usage_pid=os.getpid()`). The container is not a descendant of anything a
claim can name from the host shell, and wrapping the caller in `gpu-claim`
does not help, because the CUDA runs in the container rather than in the
wrapped tree.

This is not a missed claim. It is unledgerable with the interfaces that
exist, and containerised GPU work is the general case on this box, not an
exotic one.

The second half of the issue is that the failure was undiagnosable. The
sweep's only signal is SIGKILL, and a SIGKILLed child writes no stderr, so
the caller sees `exit -9` and an empty message. On 2026-09-01 an agent read
that as a fault in its own algorithm, rewrote a correct IVF index, and
submitted a worse method. The kill was correct by policy; nothing on the
victim's side could tell it from its own bug.

## 1. Why a pid tree cannot express this

Measured on 2026-09-01, on a container started with two children, then given
a third via `docker exec`:

```
  PID    PPID  CMD
1772376 1772352 sleep 600     <- container init (docker inspect .State.Pid)
1772431 1772376 sleep 600     <- child of init
1772536 1772352 sleep 500     <- docker exec'd
```

`procs.descendants(1772376)` returned `[1772431]`. **A `docker exec`'d
process is a child of the containerd-shim, not of container init**, so it is
outside the container's pid tree as `ps --ppid` walks it.

This rules out the issue's own preferred fix. `gpu-claim --pid <container
init pid>` would cover work spawned *by* the container's main process and
miss anything exec'd into it — for the same workload, on the same card, with
no way for the operator to tell which case they are in.

All three processes shared one cgroup, and the host shell did not:

```
1772376  0::/system.slice/docker-4e5f9b3b….scope
1772431  0::/system.slice/docker-4e5f9b3b….scope
1772536  0::/system.slice/docker-4e5f9b3b….scope   <- the exec'd one
 (shell)  0::/user.slice/user-1000.slice/…
```

The cgroup is the boundary that matches what an operator means by "this
container", and the pid tree is not.

### Deployment facts this relies on

Checked on `tig-gpu`, 2026-09-01:

| | |
|---|---|
| runner | pid 2280676, `/opt/gpuq/venv/bin/python -m gpuqueue.cli_runner`, cgroup `/system.slice/supervisor.service` — a **host** process, not containerised |
| namespace | `/proc/1/cgroup` reads `0::/init.scope` — root cgroup namespace, real systemd paths |
| hierarchy | cgroup v2 unified (`0::` lines) |
| the workload | `tig-pentesting-tig-scorer-1`, init pid 2818873, cgroup `/system.slice/docker-43faa0ee….scope`, running `uvicorn scorer_api:app` and spawning CUDA per request |

`docs/design.md:24` says "the target box is an **unprivileged container**".
That no longer describes this deployment and is corrected as part of this
work.

## 2. Where the change lands

Not in a new exemption path. `ledger.attribute` is already the single place
preflight, the orphan reaper and the VRAM watchdog ask who owns a pid, and
its docstring says why they must share it: "they share one implementation so
they cannot disagree about who owns a pid, which is the disagreement that
gets a legitimate job killed."

Making `attribute` scope-aware means a container's CUDA process is returned
in `owned` rather than `unledgered`, and all three consumers follow without
knowing scopes exist:

- the reaper never sees it, so it is not killed;
- preflight never sees it, so it does not refuse a start;
- the watchdog charges its VRAM to the scoped record, which is what makes
  this a claim rather than a mute exemption.

`preflight.own_pids` is **not** changed. Its pid-tree belt-and-braces stays
exactly as issue #19 left it, and the reasoning in its docstring — that
over-exempting is the safe way to be wrong at the last check before a
SIGKILL — is untouched.

## 3. The scope module

New `gpuqueue/cgroups.py`, importing nothing from `gpuqueue`. That is the
rule `procs.py` follows and the reason it is not a cycle: `ledger` needs
this, and `claim` imports `ledger`.

```python
cgroup_of(pid, proc_root="/proc") -> str | None
in_scope(pid, scope) -> bool
refuse_reason(scope) -> str | None
```

**Only the reverse direction is used** — pid to cgroup path, read from
`/proc/<pid>/cgroup`, which is world-readable. Nothing reads
`/sys/fs/cgroup`, so this needs no mount visibility and works from a runner
less privileged than tig-gpu's. The forward direction (cgroup to pid set)
would have been the obvious implementation and is deliberately not used.

`cgroup_of` returns the path from the `0::` line. On a cgroup-v1-only box
there is no such line, and it returns `None` — which becomes a refusal at
claim time with a stated reason, not a claim that is silently ineffective.

`in_scope` is a **prefix** match — `cg == scope or cg.startswith(scope +
"/")` — so a container that creates its own sub-cgroups is still covered.
The separator is load-bearing: bare `startswith` would put `/a/bc` inside
`/a/b`.

`proc_root` is a parameter so the parser is tested against a real fixture
directory rather than a mock. See §8.

### The structural guard

A mistyped `--scope-pid` can exempt far more than intended. Pid 1 resolves to
the root cgroup; any host shell pid resolves to a whole login session
(`/user.slice/user-0.slice/session-1848.scope`, measured above). Both look
like ordinary claims and both disable orphan protection for the card.

`refuse_reason` refuses:

| rejected | because |
|---|---|
| a non-absolute path | not a cgroup path |
| fewer than 2 components | `/`, and every top-level slice beneath it |
| terminal `session-N.scope` or `user-N.slice` | systemd session containers, never a workload |

Two rules, not three. `/`, `/init.scope`, `/system.slice` and `/user.slice`
are all one component deep, so the depth rule already refuses each of them
and a named set would be redundant with it. They are still *listed*, but
only as a message table: `--scope-pid 1` should say "cgroup `/`, which is
the whole box" rather than "fewer than 2 components", and an error an
operator cannot act on is the failure mode this whole design is about.
Recognising a path for a better message is not the same as refusing it, and
the tests in §8 assert the refusal against the depth rule so that deleting
an entry from the message table cannot open a hole.

The session rule *is* separate, because the depth rule does not cover it:
`/user.slice/user-0.slice/session-1848.scope` is three components deep and
passes a depth check. This is the shape *every* host shell pid resolves to,
so it is the likeliest accident, not the least.

No config allowlist. An unset key would have to mean something, and both
answers are bad: deny-all disables the feature on every existing box,
allow-all makes the key decorative.

## 4. The record

`ledger.Record` gains two fields, both written to the claim JSON:

| field | meaning |
|---|---|
| `scope_pid` | the anchor pid the claim named |
| `scope_cgroup` | the path resolved **at claim time** |

Read back with `.get(...)` defaults. `_load` already returns `None` on any
exception, and a record written before this change lacks both keys; a
`d["scope_pid"]` would make `_load` return `None` for every existing record,
which blinds the reaper to live claims. That is issue #19's failure with a
new cause, and it would arrive on upgrade, against records already on disk.

### Why both fields

Storing the resolved path alone is not enough, and storing the anchor alone
is not enough.

A scope is honoured only when `pid_alive(scope_pid)` **and**
`cgroup_of(scope_pid) == scope_cgroup`. If the container restarts, the anchor
dies, or the kernel recycles the pid onto an unrelated process, the two
disagree and the scope covers nothing rather than drifting onto a stranger's
cgroup.

A scope that has gone void is **logged by the runner**, not silently
dropped. A claim that has quietly stopped covering anything is the same
class of silent failure this issue is about.

`attribute` stays pure and does not log: it is called from preflight, the
reaper and the watchdog, and a print from inside it would fire three times
per sweep from three processes. The liveness test lives in
`ledger.scope_is_live(rec)`, `attribute` consults it, and `reap()` calls it
once more over the same records to put the void ones in its return dict,
beside `stale_claims` and `stuck_claims` — which is where the runner already
looks for "a claim that is no longer what it says it is".

### Ownership is a union

`usage_pid` is still set to `gpu-claim`'s own pid when `--scope-pid` is
given. A claim covers its own process tree *and* the scope: the wrapped
command may itself touch the card, and over-exempting is the safe way to be
wrong here, as `own_pids` already argues.

In `attribute`, pid-tree matches are tested before scope matches, so the
existing "first match wins" rule keeps charging a process to the more
specific owner. Cost is one small file read per (visible app × scoped
record); apps number a handful, scoped records fewer, and the call sites are
preflight and the timer-gated sweep.

## 5. Declaration surface

```
$ gpu-claim --vram-mb 3000 --scope-pid 2818873 -- ./run_experiment.py
gpu-claim: scope /system.slice/docker-43faa0ee….scope (1 live process)
```

Resolution and refusal happen at claim time, so the operator learns
immediately rather than from a SIGKILL an hour later:

```
$ gpu-claim --scope-pid 1 -- ./x
gpu-claim: --scope-pid 1 resolves to cgroup '/', which is the whole box;
  refusing. Name a pid inside the workload you mean.
```

Exit 2 on every refusal above: nothing about the card is wrong, the command
line is. This follows `--vram-mb`'s precedent at `cli_claim.py:195`.

The live process count is a best-effort `/proc` walk, at claim time only. It
is the operator's sanity check that they named a container and not the box —
"1 live process" against "247" is the difference — and it is omitted
silently if the walk fails.

`--scope-pid` is runtime-agnostic. It works for docker, podman, `systemd-run`
or a bare process, and adds no dependency. `--container <name>` sugar that
shells out to `docker inspect` is **out of scope**: it would couple the queue
to one runtime and need socket access from both `gpu-claim` and the reaper.

`--status` prints the scope alongside the rest of the record, so "what is
covered right now" stays answerable without a running service, per the
interface constraint at `docs/design.md:29`.

### Preflight must know the prospective scope

§2 says preflight needs no changes, and that is true of every call made once
a claim exists — the record is on disk and `attribute` charges the container
to it. This is the other moment. `preflight()` runs *before* the claim is
taken (`cli_claim.py:214`), so there is no record yet to attribute against.
If the target container is already running CUDA when the claim is made,
preflight sees an unledgered process and refuses with `EX_UNAVAILABLE`.

So `--scope-pid` feeds the prospective scope into `preflight()` as well as
into the record. Without this the feature fails exactly when it is needed —
claiming a busy container is refused, and claiming an idle one races the next
request.

## 6. The kill ladder

`reaper.kill_orphan_cuda` goes straight to SIGKILL (`_kill`, `reaper.py:29`).
The VRAM watchdog's `_kill_tree` already does SIGTERM → grace → SIGKILL, and
its docstring says why: "a convicted trainer that gets only SIGKILL flushes
no logs and writes no checkpoint, so the operator loses the run *and* the
evidence." The orphan sweep has the same problem and not the same treatment.

It gains the ladder, **batched**: SIGTERM every victim, one shared grace,
then SIGKILL the survivors. Sequential per-victim grace would stall the
runner tick by N × grace; batched it is one grace regardless of victim count.
`_exited` (`reaper.py:156`) already handles the zombie case.

`ORPHAN_TERM_GRACE_S = 5.0`, deliberately shorter than `_kill_tree`'s 10s. A
convicted holder is one we want to checkpoint; an unledgered process is
contending for a card someone may be blocked on. 5s is enough for a handler
to log. The cost is paid once per `orphan_cuda_interval_s` (default 60),
inside the timer-gated branch, not on every tick.

`kill_orphan_cuda` returns `list[dict]` rather than `list[int]`, and reads
each victim's **own cgroup before signalling**. That field is what tells an
operator "this was my container, not my algorithm" — it is the single most
useful thing in the record.

## 7. Making the record findable

`<queue root>/kills.jsonl`, appended by `reap()`, which already holds the
`QueueRoot`:

```json
{"ts":"2026-09-01T10:06:57Z","pid":2791919,
 "cmd":"tig-runtime build-index",
 "cgroup":"/system.slice/docker-43faa0ee….scope",
 "reason":"orphan_sweep_unledgered",
 "ledgers_consulted":["/workspace/lock/gpu","/var/lock/gpu"]}
```

`ledgers_consulted` is the list `_consulted_dirs` already builds for the log
line, so the record names the ledgers actually read rather than the ones
assumed.

A file alone is not enough. The issue's complaint is that the only record was
the runner log, "which the killed process cannot read and its owner does not
think to check" — a second file nobody thinks to check barely improves on
that. So:

- **`gpuq kills`** prints recent kills. This is the thing an agent that sees
  `killed by signal 9 (SIGKILL)` can be *told to run*.
- `skills/gpu-jobs` gains that pointer, so the agent is told.

The file is capped at the last 1000 entries, rewritten when exceeded.
Append-only with no bound is how a rare-event log becomes a disk-full
incident.

## 8. Verification

TDD, and the mutation each test kills:

| test | mutation |
|---|---|
| `cgroup_of` parses `0::/path` from a fixture proc dir; `None` on a v1-only file | returning line 1 regardless of the `0::` prefix |
| `in_scope("/a/b")` covers `/a/b/c`, rejects `/a/bc` | `startswith(scope)` without the separator |
| `refuse_reason` rejects `/user.slice/user-0.slice/session-1848.scope` | dropping the session rule — that path is 3 deep and passes a depth-only check |
| `refuse_reason` rejects `/system.slice` and `/` with the depth rule, asserted with the message table emptied | moving the top-level slices from the message table into the refusal logic, where deleting an entry opens a hole |
| `refuse_reason` admits `/system.slice/docker-….scope` | a guard so broad it refuses the only real use |
| `attribute` charges an in-scope app to the scoped record | deleting the scope branch |
| a record whose `scope_pid` is dead covers nothing | honouring `scope_cgroup` without checking the anchor |
| a record whose `scope_pid` is alive **in a different cgroup** covers nothing | checking only `pid_alive`, not the path |
| a record written without the new keys still loads and still exempts | `d["scope_pid"]` instead of `d.get` |
| pid-tree ownership is still tested before scope ownership | reordering the two passes |
| `kill_orphan_cuda` spares in-scope, kills out-of-scope | the whole feature; this is #24 end to end |
| SIGTERM precedes SIGKILL | reverting to bare `_kill()` |
| N victims cost one grace, not N | the sequential-grace variant |
| `preflight` with a prospective scope does not refuse | §5's trap |
| 1001 appends leave 1000 entries | unbounded growth |

`cgroup_of` takes `proc_root` so it is exercised against a real fixture. A
suite that monkeypatched `cgroup_of` at every call site would still pass with
the parser deleted.

Then on `tig-gpu`, against the live `tig-pentesting-tig-scorer-1`: claim it,
drive CUDA work through it, and confirm a sweep spares it — and that removing
the claim makes the sweep kill it again, so the test discriminates.

## 9. Known failure modes

- **A scope that outlives its workload.** A claim naming a container that
  goes idle exempts nothing while still counting VRAM against the card. That
  is the correct trade — it is a declaration, like `--vram-mb` — but it means
  a forgotten `gpu-claim --scope-pid` holds capacity. The existing stale
  sweep handles it: the record dies with the wrapping process.
- **A container restart mid-claim** voids the scope by design (§4). The
  container's new CUDA work is then unledgered and killable. This is logged,
  but the operator's claim will not silently start covering the new instance.
- **cgroup v1.** No `0::` line, so `--scope-pid` refuses. The box is v2 and
  this is a stated refusal, not a silent no-op.
- **A pid outside the reaper's namespace.** If a future deployment runs the
  runner inside a cgroup namespace, `/proc/<pid>/cgroup` returns namespaced
  paths, and a path recorded by a claimant in a different namespace will not
  compare equal. Both reads that matter happen in the reaper's own process,
  so this degrades to "scope covers nothing" and is logged — it does not
  mis-exempt. Not solved here; `design.md:24` is corrected so the assumption
  is at least written down.

## Out of scope

- **`--container <name>` sugar.** §5.
- **The `gpuq bug` / `[autofix].enabled` coupling.** `cli_gpuq.py:162` gates
  filing a hand-written report on the *dispatch* switch, though its own
  docstring says the path "dispatches nothing". This is why issue #24 had to
  be filed with `gh`. It is a real bug and it is a separate one.
- **Cgroup-based *enforcement*.** Nothing here writes to `/sys/fs/cgroup` or
  sets a memory limit. Consistent with `design.md`'s exclusion of hard
  per-process VRAM caps: this convicts, it does not prevent.
