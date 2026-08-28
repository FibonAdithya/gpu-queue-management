# Exempting a claim the reaper's environment cannot see

Fixes #19. `kill_orphan_cuda` SIGKILLs a direct `gpu-claim` run when the daemon's
`GPU_CLAIM_DIR` comes from supervisor and the interactive shell does not inherit
it. Measured 14/29 runs (48%) killed on the deployment box.

## Root cause

`claim.claim_dir()` resolves `$GPU_CLAIM_DIR` **in the calling process**.

- The claim writer is the user's shell. Unset there, so it writes to
  `DEFAULT_CLAIM_DIR` = `/var/lock/gpu`.
- `preflight.own_pids()` (`preflight.py:68`) resolves it in the **reaper's**
  process, which supervisor gave `/workspace/lock/gpu`.

`reaper.kill_orphan_cuda` (`reaper.py:84`) builds `exempt = set(protect) |
own_pids()`. The comment at `reaper.py:62-83` justifies the two-directory split
by an invariant: `own_pids` is a *strict superset* of what `attribute` owns, so
disagreement can only add exemptions. That holds only when the writer and the
reaper resolve `$GPU_CLAIM_DIR` identically. Across two environments they do
not — the sets are disjoint, not nested — and the direct run is unledgered,
unexempt, and SIGKILLed on the next sweep.

`test_a_divergent_runner_claim_dir_still_spares_a_direct_run` cannot catch this:
it varies `cfg.claim_dir` while the claim is still written into the same
`$GPU_CLAIM_DIR` that `own_pids()` reads. One process cannot express two
environments.

## Second finding, not in the issue

`reap()` returns `killed_pids` and **nothing logs it**. `grep -rn killed_pids
src/` finds only the construction site at `reaper.py:330`. The runner SIGKILLs a
process and says nothing at any level. This is why the symptom reached the
reporter as an empty-stderr `exit -9` with nothing pointing at gpuq.

## Changes

### 1. `claim.all_claim_dirs()` — the directories a claim on this box could be in

```python
def all_claim_dirs() -> list[Path]:
    ...  # [claim_dir(), Path(DEFAULT_CLAIM_DIR)], deduped on .resolve()
```

`all_claim_dirs`, not `claim_dirs`: `reap()` already has `claim_dir` in scope
(`reaper.py:311`) and will import this alongside it. Two names one character
apart, in one function, both meaning something different, is how the next
reader introduces the bug this plan is fixing.

`DEFAULT_CLAIM_DIR` is read from the module global **at call time**, not
import-bound. That is load-bearing twice over: it is what lets the suite stub
the constant (see 5), and what keeps a single definition of "where claims live".

Dedup on `.resolve()` so a symlinked or trailing-slash spelling of the same
directory is one entry, not two. `resolve()` defaults to `strict=False` and so
does *not* raise for a path that is simply absent -- the `except OSError` is
there for ELOOP on a symlink cycle and EACCES on an unreadable intermediate
component, and falls back to the unresolved path rather than dropping the
directory.

### 2. `preflight.own_pids(directory=None)` — union over `claim_dirs()`

An explicit `directory=` still means exactly that one directory: callers who
name a directory are answering a different question, and the tests that pass one
are asserting about it specifically.

Only `directory=None` — the reaper's bare call, the one that matters — widens to
the union. `ledger.all_records` returns `[]` for a directory that is not there
(`ledger.py:159`), so a box with no `/var/lock/gpu` pays a `stat` and nothing
else.

This is the safe direction to be wrong, and `own_pids`' own docstring already
says so: "it is the last exemption before a SIGKILL, so over-exempting is the
safe way to be wrong." The residual hazard — a *live* pid in `/var/lock/gpu`
that is coincidentally an orphan — is pid-reuse, which the primary directory has
had all along.

That last sentence understates it by the sweep interval, and #21 tracks the
difference: `reap()` calls `release_stale(cfg.claim_dir)` and nothing sweeps
`DEFAULT_CLAIM_DIR`, so a dead record there persists for the life of the box
rather than for one tick. Fail-open, so not a blocker for this change, but not
the same window either.

`own_pids` has exactly one production caller (`reaper.py:84`); `preflight()` and
`unledgered_processes()` are untouched, and that is a decision, not an
oversight. Those are the *refuse to start* path, where over-exempting fails
open: a stray CUDA process ledgered in the other directory would be attributed
and the run would start onto a contended card. Widening them would be the
mirror-image bug, so a test pins them narrow.

Cost: `descendants()` is a recursive `ps` per record (`procs.py:33-41`). The
union doubles the number of *directories* walked, not the number of records, and
a box with no `/var/lock/gpu` pays a single `is_dir()` (`ledger.py:159`) per
sweep. This runs on the `orphan_cuda_interval_s` timer, not per tick.

### 3. `reaper` — say which ledgers were consulted, and that a kill happened

- `reap()` gains `"exemption_dirs"`: `[str(d) for d in all_claim_dirs()]`
  populated **exactly when `kill_orphan_cuda()` was actually called** -- that is,
  `include_orphan_cuda and cfg.kill_orphan_cuda and apps is not None` -- and `[]`
  otherwise. Not "when the sweep ran": with `kill_orphan_cuda = false` and
  `enforce_vram = true` the sweep runs and consults no exemption at all, and
  with `apps is None` it measures nothing. A key populated on those ticks would
  have the runner name ledgers nothing read. Additive; every existing
  `result["killed_pids"]` assertion still holds.
- `Runner._reap` logs a warning naming the killed pids and those directories,
  and only when `killed_pids` is non-empty. It reads both through `.get()`,
  because `test_runner.py` monkeypatches `reap` to return partial dicts
  (`tests/test_runner.py:911`).

Both `kill_orphan_cuda` and the log go through the same `all_claim_dirs()`, so
the line names what was actually consulted.

### 3b. The comments that are now false

`reaper.py:62-83` justifies the two-directory split with a *strict superset*
invariant, and `own_pids`' docstring (`preflight.py:57-66`) calls itself
"deliberately coarser". This change is the counterexample to the first: across
two process environments the sets are disjoint, not nested. Both must be
rewritten to state the new basis -- the exemption is a union over every
directory a claim on this box could be in -- and to keep the standing
prohibition on plumbing `cfg.claim_dir` through, with its own reason rather than
the disproved one.

This is a deliverable, not tidying. A comment asserting an invariant that does
not hold is exactly what let this ship.

### 4. `cli_claim` — warn when the runner reads somewhere else

New `config.claim_dir_setting(path=None) -> Path | None`: `[queue].claim_dir`
from `GPUQ_CONFIG`, defensive in the three ways `vram_policy` and `max_holders`
already are (not `load_config`; an unreadable or absent file is `None`, not an
error). `None` means "no configured runner directory to disagree with" — the
daemon may still have one in its environment, which this process cannot see, so
we say nothing rather than guess.

Emitted once, early in `main()`, before the `--status` and `--reap` branches:
`--status` reporting a claim as healthy that the reaper cannot see is named in
the issue as part of what made this hard to find.

Two consequences, and the second is conditional, because change 2 removes it for
the common case:

- Always, on divergence: the runner's ledger does not count this claim, so it
  can admit a job on top of it.
- Additionally, when we are writing to neither the configured directory nor
  `DEFAULT_CLAIM_DIR`: no exemption covers this run and `kill_orphan_cuda` will
  SIGKILL it.

The reaper's *own* `$GPU_CLAIM_DIR` is not knowable from this process, so the
message says which directories the reaper can see in terms of what it is —
its own environment plus the default — rather than asserting a value.

### 5. `conftest` — stub `DEFAULT_CLAIM_DIR`

`_no_deployed_claims` (`conftest.py:37-50`) sets `GPU_CLAIM_DIR` to an isolated
tmp dir precisely so the suite never reads the live `/var/lock/gpu`. Change 2
reintroduces that hazard through a constant no env var can neutralize. Extend
the same fixture to `monkeypatch.setattr(claim, "DEFAULT_CLAIM_DIR", ...)` on a
second isolated tmp dir.

Second, not the same one: a fixture that pointed both at one directory would
make every new test pass whether or not the union exists.

The stub value is a **`str`**, matching `claim.py:23` -- it is consumed as
`os.environ.get("GPU_CLAIM_DIR", DEFAULT_CLAIM_DIR)`, and a `Path` there would
type-check by accident rather than by design.

### 6. `tests/test_reaper.py` — a holder fixture the new tests can aim

`direct_claim` (`tests/test_reaper.py:203-234`) writes its claim file into
`$GPU_CLAIM_DIR`, hard-coded. The first two tests below need the same detached
holder/child pair with the claim written into `DEFAULT_CLAIM_DIR` instead.

Split it: a `holder_process` fixture that spawns the pair, keeps the
`assert holder not in descendants(os.getpid())` guard and does the SIGKILL
cleanup, plus a `_write_claim(directory, holder)` helper. `direct_claim` becomes
those two composed, so `test_own_pids_covers_a_claim_holders_children` and
`test_a_divergent_runner_claim_dir_still_spares_a_direct_run` are unchanged in
behaviour.

Doing this by mutating `direct_claim` in place would move the two existing tests
onto the default directory, where the union makes them pass for a reason that
has nothing to do with what they assert.

## Tests

TDD, one at a time, each mutation-checked.

| Test | Mutation it catches |
|---|---|
| `test_own_pids_exempts_a_claim_written_under_the_default_dir` | change 2 reverted → `own_pids` reads only `$GPU_CLAIM_DIR`, holder and child absent |
| `test_a_claim_in_a_directory_the_daemon_never_reads_is_still_spared` | change 2 reverted → the child is SIGKILLed; the reproduction end to end through `reap()` |
| `test_all_claim_dirs_does_not_repeat_one_directory` | dedup dropped → 2 entries |
| `test_own_pids_with_an_explicit_directory_reads_only_that_one` | widening applied to the explicit-directory branch too |
| `test_reap_reports_the_directories_it_exempted_from` | `exemption_dirs` key dropped or left empty when `kill_orphan_cuda` ran |
| `test_no_exemption_dirs_when_the_sweep_did_not_run` | key populated when `include_orphan_cuda=False` |
| `test_no_exemption_dirs_when_only_the_vram_watchdog_ran` | key gated on the sweep rather than on `cfg.kill_orphan_cuda` — the case the first two miss |
| `test_preflight_still_refuses_a_claim_from_another_directory` | the union applied to `preflight()`/`unledgered_processes()`, failing open on the refuse-to-start path |
| `test_the_runner_logs_a_kill_and_the_ledgers_it_consulted` | the `log.warning` in `_reap` removed |
| `test_gpu_claim_warns_when_the_runner_reads_another_directory` | the warning removed |
| `test_gpu_claim_is_quiet_when_the_directories_agree` | warning made unconditional |
| `test_gpu_claim_is_quiet_when_no_config_declares_one` | `None` treated as a divergence |
| `test_the_warning_names_sigkill_only_for_a_third_directory` | the conditional clause made unconditional — this is the claim change 2 makes false for the default dir. Reaches the branch with `monkeypatch.delenv("GPU_CLAIM_DIR")`, so `claim_dir()` falls through to the stubbed constant; that works only because change 5 stubs the constant rather than the variable |
| `test_status_warns_before_it_prints_a_claim_the_reaper_cannot_see` | the warning placed after the `--status` early return |
| `test_claim_dir_setting_reads_the_queue_table` | reader returns `None` always, making the whole warning dead |
| `test_claim_dir_setting_is_none_for_an_unreadable_config` | reader raises instead |

The first two need the real detached process pair, and its existing
`assert holder not in descendants(os.getpid())` guard is what keeps them from
passing on `own_pids`' unconditional `{os.getpid()} | descendants(...)` term.
Without that guard both assertions are tautological.

`test_a_claim_in_a_directory_the_daemon_never_reads_is_still_spared` drives
`reap()`, so it **must** undo the autouse stub at `tests/test_reaper.py:29`
(`monkeypatch.setattr(rp, "own_pids", _pf.own_pids)`) and stub `rp.compute_apps`,
exactly as the two existing reap-level tests do at lines 251 and 289. Left
stubbed, `own_pids` returns `set()` and the test passes with the fix reverted —
it would assert nothing at all.

Expected suite: 500 → ~516.

`test_a_divergent_runner_claim_dir_still_spares_a_direct_run` must stay green
and must stay *meaningful*: it asserts the `cfg.claim_dir`-vs-`$GPU_CLAIM_DIR`
split, which change 2 does not touch.

## Explicitly not done

Unifying the two directories, or passing `cfg.claim_dir` into `own_pids()`.
`reaper.py:76-83` is explicit that either removes the protection, and the issue
declines to propose it. This change widens the exemption set; it does not
narrow or redirect it.
