# Autonomous bug-fix PRs for gpuq — Design

Date: 2026-08-05

When `gpuq` breaks in its own code, file a GitHub issue carrying the evidence
and let a headless Claude Code run in GitHub Actions open a PR against it. The
owner reviews and merges. Nothing merges itself.

## Purpose

Agents submitting work to this queue cannot fix the queue. When the runner
throws — a bad worktree, a `git_ops` failure, a queue-state inconsistency —
the job dies, the agent reports "my job failed", and the defect sits until a
human happens to read a log. The failures that matter most are the ones nobody
is watching for.

The trigger is the tool's own stack trace, not a person noticing. The output is
a reviewable PR, not a live change.

## 1. Trust boundaries

Three planes, each holding a different credential, none able to do the others'
job.

| Plane | Runs | Holds | Can |
|---|---|---|---|
| **GPU box** | runner, classifier, issue filer | fine-grained PAT | open/comment issues on this repo |
| **Actions** | `anthropics/claude-code-action` | `CLAUDE_CODE_OAUTH_TOKEN` | push a branch, open a PR |
| **Owner** | review | — | merge |

The box's PAT is scoped to **`issues: write` on `gpu-queue-management` alone,
and explicitly not `contents`**. `gpuq.example.toml` already makes this
argument about the results key — *"anyone who can queue a job here can use that
key"* — and it applies harder to a token that could otherwise write source.

Its worst case is *not* just issue spam, though. The issue body it can file
is also the fixer's prompt (§6), and the `JobSpec` embedded in that body —
`cmd`, `branch`, `project` — is caller-supplied, not filtered. Anyone who can
queue a job on the box can therefore, in principle, put chosen text in front
of the headless Claude Code run. This is why the workflow prompt, ahead of
its numbered steps, tells the fixer to read the JobSpec block as data about
a failure and never as instructions: the risk is real, not merely theoretical, and the sentence
in an earlier draft of this document calling the worst case "issue spam"
alone was too reassuring. What actually bounds it is two things outside this
module entirely — branch protection on `main`, so nothing the fixer writes
lands without a human reading it, and the owner's own review of the PR before
merge, for the ordinary reason that an autonomous PR from *any* source should
be read before it merges. An earlier draft called both of these existing
machinery. The review was; the branch protection was not — it was assumed
here and nowhere enabled, and GitHub gates the setting behind a paid plan for
private repositories, so acquiring it meant making this repository public.
Both are real now, and `docs/deploying.md` records how to verify it. The
worst case, stated accurately, is issue spam plus an attacker-influenced
prompt, held from becoming an attacker-influenced *merge* by those two
controls.

The OAuth token comes from `claude setup-token` and draws on the owner's Max
budget. It lives only as a repo secret. The workflow triggers on `issues` and
never on `pull_request_target`, so nothing fork-authored can reach it.

Branch protection on `main`: the Action opens PRs, it does not merge.

**Notification requires no new integration.** GitHub emails on issue and PR
activity, Hermes reads email, so Hermes surfaces the PR through triage it
already performs. No GitHub credential goes near the mailbox plane and the
air-gapped sandbox is untouched.

## 2. Classification — whose fault was it?

A new `src/gpuqueue/bugreport.py` answers one question about every failure:
*did gpuq's own code raise this?* The distinction already exists in
`runner.py:274` and is being made load-bearing rather than cosmetic.

**Caller fault — never files a bug:**

- `result.oom` — a configuration error, per the existing rule, not a transient.
- `result.timed_out` — the caller's job hung.
- plain `exit N` from the child process.
- `_collect_artifacts` raising "declared artifact not produced" — a gpuq
  exception carrying caller error, and the one case that must be classified out
  by hand.
- `StartFailed` where the cause is a missing binary or interpreter in the
  caller's `--` command. `StartFailed` is genuinely ambiguous; only causes that
  are not "the thing you asked to run does not exist" count as gpuq's fault.

**gpuq fault — files a bug:**

- `runner.py:193` — `checkout failed`, i.e. `_prepare_workdir` / `git_ops`.
- `runner.py:249` — artifact collection failing for any reason other than the
  above.
- unhandled exceptions in preflight, the reaper, queue state transitions, and
  the claim path.

This is a filter in code, not a model's judgment about blame. That is the point:
the agent that just failed is the least reliable narrator of whether the queue
is broken or its own request was.

## 3. Two filing paths

**Runner auto-files.** Only the runner knows an exception came from its own
stack. It files without asking anyone, labelled `gpuq-auto`, and this path
carries **no prose from any agent** — only facts.

**Agents file with `gpuq bug`.** Covers what the runner cannot see: failures in
`gpuq submit` / `wait` / `show`, config errors, CLI gaps. This path carries free
text and unreliable blame, so it is labelled `gpuq-reported` and dispatches
nothing until the owner adds `fix-me`.

The gate follows the trust boundary that already exists: structural evidence
runs autonomously, prose waits for a human.

## 4. Signature and dedup

The signature is the exception type, plus the `gpuqueue`-internal traceback
frames **by function name rather than line number** so it survives unrelated
edits, plus the phase (preflight / checkout / execute / artifacts / reap),
hashed to short hex. It is written into the issue body as `sig: <hex>` and
matched with `gh issue list --search`.

Three lookups, in order:

1. **Open issue with this signature** — comment, increment the occurrence
   count, dispatch nothing.
2. **Open PR referencing it** — comment on the PR. A bug that fails every job
   must not spawn one run per job; this is the lookup that prevents that.
3. **Closed within 30 days** — file new, linking the old as *"previously fixed
   in #N; that fix did not hold."* A failed previous fix is the most useful
   context available to the next attempt, and it costs nothing to pass on.

Agent-filed issues have no traceback and therefore no signature. They are not
deduplicated automatically; the owner closes duplicates when triaging the
`fix-me` label, which is the same act that authorises the run.

## 5. Limits and the off switch

- One in-flight run per signature.
- Three auto-dispatches per rolling 24h. Past the cap the issue **still files**,
  labelled `throttled`, with no run — evidence is never lost, budget cannot run
  away.
- **To run a `throttled` bug anyway, remove `throttled`.** That is the trigger;
  adding `fix-me` on top of it is not, because the workflow refuses any issue
  still carrying `throttled`. Either order works — the workflow also fires on
  the label being removed — but removing `throttled` is the act that does it,
  and an issue left carrying both labels runs nothing.
- A recurrence always bumps `occurrences` and `last seen` in the body; it also
  posts a comment at most once a day. An untriaged bug recurs every
  `report_cooldown_s`, and a comment per recurrence buried the traceback that
  matters under roughly a hundred a day.
- Dedup reads the newest 500 open `gpuq-auto` issues. That is a real ceiling:
  past it, older signatures fall out of the lookup and recur as duplicates.
  The runner logs a warning when the list comes back full — close bugs.
- Repo variable `GPUQ_AUTOFIX`, read by the workflow. Setting it to `off` from
  the GitHub mobile web UI stops everything without a commit.

The throttle is a safety control today and becomes a cost control if
programmatic usage moves off the subscription pool (§8).

## 6. What the issue carries

The issue body *is* the prompt, so it matters more than the workflow file. On
the auto path it carries facts only:

- the exception and full traceback
- `sig: <hex>`
- the `JobSpec` that triggered it
- queue state at the time
- the runner's gpuq commit
- occurrence count, first and last seen

## 7. What the Action is instructed to do

1. **Reproduce as a failing test in `tests/` first, then fix.** This gives the
   owner a merge criterion better than reading the diff.
2. Stay within `src/gpuqueue/` and `tests/`. Changes to `bootstrap.sh`,
   `supervisor/`, or the config schema are called out prominently in the PR
   body rather than made quietly.
3. **If the root cause is caller error the classifier mislabelled, close the
   issue with an explanation and change nothing.** Without this escape hatch
   every classifier false positive becomes a PR that makes `gpuq` more
   permissive — the exact failure the OOM-is-not-a-transient rule exists to
   prevent.
4. If verifying the fix genuinely requires a GPU, say so in the PR rather than
   implying a green CI run proves it.
5. One root cause per PR.

**A GitHub runner has no GPU**, so fixes must be verifiable by the existing
pytest suite on CPU. This fits the gpuq-fault categories well — they map onto
`test_git_ops`, `test_preflight`, `test_queue`, and `test_runner`, all of which
already run cardless — but it is a real ceiling on what this system can
confirm.

## 8. Cost

The OAuth token draws on the Max subscription rather than API billing. This is
current policy, not a guarantee: on 2026-05-13 Anthropic announced that Agent
SDK, `claude -p`, and GitHub Actions usage would move to a separate metered
credit from 2026-06-15, then paused the change on the day, with a revised
version possible on notice. The three-per-day throttle is the hook that becomes
a spend cap if it returns.

## 9. Known failure modes

- **A mislabelled caller error reaches the fixer.** Mitigated by §7.3, not
  eliminated. Watch for PRs that loosen validation.
- **A wrong fix merges and the bug recurs.** Caught by the 30-day closed
  lookup, which tells the next run the previous fix failed.
- **The box's PAT leaks to anyone who can queue a job.** By design it can only
  file issues.
- **A novel exception class fires repeatedly under distinct signatures.** The
  daily cap bounds it; the issues still accumulate as evidence.
- **A caller puts chosen text in front of the fixer.** The `JobSpec` in the
  issue body (§1, §6) is caller-supplied and unfiltered, and that body is the
  prompt. The workflow instructs the fixer to treat it as data, not
  instructions, but a prompt-injection attempt is not eliminated by an
  instruction any more than input validation is eliminated by a comment
  asking callers to be nice. What actually holds is downstream of the model:
  branch protection on `main` and the owner's review before merge, the same
  controls any autonomous-PR source needs regardless of how it was
  triggered. The worst case is issue spam *plus* an attacker-influenced
  prompt — not merely issue spam — bounded there rather than eliminated at
  the source.

## 10. Verification

1. Force a `git_ops` failure; confirm an issue files with a signature and a
   traceback.
2. Force the same failure twice more; confirm one issue, occurrence count 3,
   one dispatch.
3. Fail an OOM job; confirm **no** issue.
4. Fail a job with a missing artifact declaration; confirm **no** issue.
5. Confirm the Action opens a PR containing a failing-then-passing test.
6. Hand it a deliberately mislabelled caller error; confirm it closes rather
   than patches.
7. Set `GPUQ_AUTOFIX=off`; confirm filing continues and dispatch stops.
8. Exceed three dispatches in a day; confirm the fourth files as `throttled`.
9. Confirm the PR notification reaches Telegram through Hermes' existing mail
   triage.
10. Confirm the box's PAT cannot push a commit.

## Out of scope

- **Auto-merge.** The owner merges. The PR is the gate.
- **Fixing anything outside this repo.** Callers' training code is theirs.
- **GPU-dependent verification.** See §7.
- **Hermes dispatching runs.** Hermes notifies via existing email triage and
  holds no GitHub credential.
- **Fixing bugs found any way other than a runtime failure.** No proactive
  auditing, no scheduled sweeps.
