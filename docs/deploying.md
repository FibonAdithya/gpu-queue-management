# Deploying to a remote GPU box

A running document. It covers taking this repo to *any* remote box with a GPU —
nothing here is specific to one host. Add a row to [Boxes](#boxes) when you
deploy somewhere new, and add to [Gotchas](#gotchas) when something bites you.

Host identity lives in exactly one place: an ssh alias. Everything below is
written against `$BOX`, so pointing this at a different machine is a one-line
change.

```bash
export BOX=my-gpu-box     # an alias in ~/.ssh/config
```

## 1. Check the box before installing

Run this first. It answers every question the design depends on, and changes
nothing:

```bash
ssh $BOX 'bash -s' <<'EOF'
echo "== interpreters =="
for p in python python3 python3.11 python3.12 python3.13; do
  command -v $p >/dev/null 2>&1 && echo "$p -> $($p -V 2>&1)"
done
ls -d /venv/* 2>/dev/null    # image-provided environments, if any
echo "== GPU =="
nvidia-smi -L || echo "NO nvidia-smi"
nvidia-smi --query-gpu=uuid --format=csv,noheader
echo "== can it enumerate CUDA processes? =="
nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader
echo "== does it report a total? (capacity discovery) =="
nvidia-smi --query-gpu=memory.total --format=csv,noheader
echo "== supervisor / git =="
command -v supervisord supervisorctl; ls -d /etc/supervisor/conf.d; git --version
echo "== workspace =="
ls -d /workspace && touch /workspace/.probe && rm /workspace/.probe && echo writable
EOF
```

What the answers mean:

| Check | Requirement | If it fails |
|---|---|---|
| A Python **3.11+** | stdlib `tomllib`; there is no `tomli` fallback | Use another interpreter on the box (step 3), or install one |
| `nvidia-smi -L` | GPU identity comes from here — torch is never imported | Without it the GPU lane refuses every job, by design |
| `--query-compute-apps` | Decides whether preflight is a real guard | See [Preflight](#preflight-is-only-as-good-as-nvidia-smi) below |
| `--query-gpu=memory.total` | Capacity discovery for the GPU lane | Must print like `8188 MiB`. See [Capacity discovery](#capacity-discovery-fails-quietly) below |
| supervisor + `conf.d` | How the runner is kept alive | `./bootstrap.sh --no-supervisor` and run `gpuq-runner` yourself |
| `/workspace` writable | Default prefix for queue, locks, checkouts | Set `GPUQ_PREFIX` to somewhere writable |

### Capacity discovery fails quietly

Confirm on the box, once, before believing the GPU lane shares anything:

```bash
gpu-claim --status                       # should print an empty ledger, not an error
python3 -c 'from gpuqueue.gpuid import total_vram_mb; print(total_vram_mb())'
```

The second command must print the card's total in MiB. If it prints `None`,
`gpuq` cannot size the card, and **every GPU job is admitted exclusively** —
the lane behaves exactly as it did before declarations existed. Nothing
errors and nothing warns at submit time; jobs simply queue behind each other
while `gpuq list` looks entirely healthy. That is the one failure in this
feature that is invisible from the outside, which is why it is worth one
command up front.

`total_vram_mb` parses `nvidia-smi --query-gpu=memory.total`, expecting a
line like `8188 MiB`. A driver that words it differently returns `None` by
the same path as a missing binary. Set `[queue].gpu_vram_mb` explicitly to
override, and prefer a number the driver will actually hand out rather than
the nameplate total.

### Preflight is only as good as nvidia-smi

`--query-compute-apps` behaves three ways, and the difference matters:

- **Lists processes** → preflight is a real guard. It will refuse to start when
  a stranger holds the card, naming the pid.
- **Empty, exit 0** → the card is idle. Verify it can *actually* see processes,
  because a container that cannot will also look like this. Start a CUDA
  process and re-run the query. If your own process does not appear, treat it
  as the next case.
- **`[Not Supported]` or missing** → preflight degrades to a warning and the
  advisory lock is all you have. Everything still works; accidental contention
  just fails later and less legibly.

## 2. Get the code onto the box

Either is fine. Neither publishes anything.

```bash
# from a clone (the box needs read access to the remote)
ssh $BOX 'git clone <this-repo-url> /workspace/gpu-queue-management'

# or push the working tree straight over ssh
rsync -az --exclude .venv --exclude __pycache__ --exclude .pytest_cache \
  --exclude '*.egg-info' ./ $BOX:/workspace/gpu-queue-management/
```

If you rsync, git may complain about `dubious ownership` because file uids came
from your machine:

```bash
ssh $BOX 'chown -R root:root /workspace/gpu-queue-management &&
          git config --global --add safe.directory /workspace/gpu-queue-management'
```

## 3. Choose the interpreter, then bootstrap

**Do not assume `python3` is the right one.** Many ML images ship a separate
environment holding torch, and that is usually where the runner belongs — this
package has no dependencies, so installing into it cannot conflict with the ML
stack.

```bash
ssh $BOX 'cd /workspace/gpu-queue-management &&
          PYTHON=/venv/main/bin/python ./bootstrap.sh --dry-run'   # look first
ssh $BOX 'cd /workspace/gpu-queue-management &&
          PYTHON=/venv/main/bin/python ./bootstrap.sh'
```

`bootstrap.sh` is idempotent — rerun it as often as you like. It honours:

| Variable | Default | |
|---|---|---|
| `PYTHON` | `python3` | Interpreter the runner is installed into. Must be 3.11+ |
| `GPUQ_PREFIX` | `/workspace` | Everything else derives from this |
| `QUEUE_ROOT` | `$GPUQ_PREFIX/queue` | |
| `GPU_CLAIM_DIR` | `$GPUQ_PREFIX/lock/gpu` | Must be identical for **every** participant on the box, or you have two locks for one card |
| `GPUQ_CONFIG` | `$GPUQ_PREFIX/gpuq.toml` | Written once, never overwritten |
| `GPUQ_SKILLS_DIR` | `~/.claude/skills` | Where the agent skill is installed |
| `SUPERVISOR_CONF_DIR` | `/etc/supervisor/conf.d` | |

Flags: `--dry-run`, `--no-supervisor`.

**The first run always reports a failed clone.** It writes a config full of
example placeholders and tells you to edit it, so there is nothing real to
clone yet. It is not fatal and everything else still installs.

### Upgrading past the VRAM ledger

Upgrade the whole installation in one pass — `bootstrap.sh` does this, and
it is why the README argues for one shared installation and never a
vendored copy. A runner from before the ledger reads `<key>.lock.json` and
cannot see `<key>.lock.d/`, so it treats a new `gpu-claim` holder's trainer
as an orphan and kills it. The reverse direction is safe: new code reads an
old holder's file as an exclusive claim, and a pre-ledger `gpu-claim`
holding `flock` for its whole run is reported as such rather than hung on.

## 4. Declare your projects

Edit `$GPUQ_CONFIG` on the box, then rerun `bootstrap.sh` to clone them:

```toml
[queue]
root = "/workspace/queue"
cpu_slots = 4          # not the core count; see docs/design.md
claim_dir = "/workspace/lock/gpu"
# gpu_vram_mb = 8188   # default: ask the card
gpu_vram_reserve_mb = 512
gpu_max_jobs = 2
enforce_vram = true

[project.myproject]
remote   = "git@github.com:me/myproject.git"
checkout = "/workspace/checkouts/myproject"
venv     = "/venv/main"      # jobs get this on PATH
commit_artifacts = true
```

The four GPU-capacity keys, all under `[queue]`:

| Key | Default | |
|---|---|---|
| `gpu_vram_mb` | what `nvidia-smi` reports | Capacity override, for boxes where the query is unavailable or reports more than the driver will hand out. |
| `gpu_vram_reserve_mb` | `512` | Held back from admission. Two jobs that each fit exactly still fragment the same heap; this is the only lever against that. |
| `gpu_max_jobs` | `2` | A latency budget, not a safety one. VRAM alone would admit sixteen 500 MiB jobs onto an 8 GB card, all time-slicing and each slower than if it had waited. 2 is what has been measured (15% → 62% utilization); raise it on a box that has measured more. |
| `enforce_vram` | `true` | Kill a job using more than it declared. Off does not make over-use safe — it makes it unattributable. Write it unquoted: `"false"` in quotes is a config error, not `false`. |

The first three are read by `gpu-claim` as well as by the runner, because the
two of them share one ledger and a capacity they disagree about is a card
they will double-book. `gpu_max_jobs` is in that set because the cap is
enforced in the ledger, against every holder of the card — a limit only the
runner applied to its own jobs would leave four hand-run `gpu-claim`s
time-slicing on an 8 GB card, which is the contention it exists to bound.

`gpu-claim` finds this file at `$GPUQ_CONFIG`, or `/workspace/gpuq.toml`;
where it finds neither it sizes the card from `nvidia-smi`, holds back the
default 512 MiB and caps at the default 2 jobs, which is what a box with no
runner deployed wants. So if you override any of the three on a box where
people also run `gpu-claim` by hand, make sure `GPUQ_CONFIG` is exported in
their environment too — otherwise they will admit against the nameplate total
while the runner admits against yours.

**`gpuq-runner --config` does not move `gpu-claim`.** A standalone claim has
no way to know what flags the daemon was started with, so it reads
`$GPUQ_CONFIG`/`/workspace/gpuq.toml` regardless. Starting the runner on some
other path reopens exactly the divergence above: declare `gpu_vram_mb = 7000`
in `/etc/gpuq-alt.toml`, pass it with `--config`, and the runner admits
against 6488 MiB while `gpu-claim` sizes from `nvidia-smi`'s 8188 and admits
against 7676 — 1188 MiB of over-admission into one ledger. The runner warns
about this at startup; export `GPUQ_CONFIG` to the same path to clear it.

**These three keys are read once, at startup.** The runner caches the card's
total for the life of the daemon, where `gpu-claim` re-reads the file on every
invocation. Raising `gpu_vram_reserve_mb` after an OOM without restarting the
runner therefore splits the two views again — `gpu-claim` immediately admits
against `total - 2048` while the runner keeps admitting against `total - 512`.
Restart the runner after editing any of them.

`gpu_vram_mb` describes the card the runner manages, which is always card 0.
`gpu-claim --gpu-index 1` sizes card 1 from `nvidia-smi` regardless, and
applies the reserve.

### Telling submitters how to size `--vram-mb`

A declaration is measured the way it is enforced: `nvidia-smi`'s per-pid used
memory, in MiB. That number includes the ~250 MiB CUDA context and PyTorch's
caching allocator high-water mark, not live tensor bytes. **A declaration
sized from `torch.cuda.max_memory_allocated()` will be too small**, and the
watchdog will kill the job for exceeding a figure its author thought was
generous. Tell people to read the number off `nvidia-smi` during a run, and
to round up.

### The claim directory must be on a local filesystem

The ledger's correctness rests on `flock` over `$GPU_CLAIM_DIR` and on
`os.replace` being atomic within it. Both hold on a local filesystem. On NFS
they are unverified here — `flock` is emulated via POSIX locks on Linux
clients and its behaviour across a server restart is not something gpuq has
been tested against. Keep the claim directory on local disk; there is no
reason for it to be shared, since it describes one box's card.

Set `venv` even if the runner already lives there. It is what puts `python` on
a job's PATH — and on many images there is **no bare `python`**, only `python3`,
so a job that says `-- python train.py` fails without it.

A private `remote` needs a deploy key on the box.

### Where results go, and what key that needs

`commit_artifacts` writes a job's declared artifacts into a git repository.
Which one decides how much access the box needs — and *anyone who can queue a
job on the box can use that access*, because the box's ssh access is the
security boundary (`docs/design.md`, "Not in scope").

| Arrangement | Config | Key on the box | Survives the box being destroyed? |
|---|---|---|---|
| Local only | `commit_artifacts = true` | read-only | **No.** Pull results before teardown |
| Push to the code repo | `+ push = true` | **write, to your code repo** | Yes |
| **Split** | `+ results_remote`/`results_checkout` | read-only for code, write for results only | Yes |

The split is the one worth the extra moving part. The code repo key stays
read-only, and the writable key reaches nothing but a results repository:

```toml
[project.myproject]
remote   = "git@github.com:you/myproject.git"          # read-only deploy key
checkout = "/workspace/checkouts/myproject"
venv     = "/venv/main"
commit_artifacts = true

results_remote   = "git@github.com:you/myproject-results.git"   # write key
results_checkout = "/workspace/checkouts/myproject-results"
results_branch   = "main"
```

Artifacts land at **`<project>/<job-id>/<declared path>`** in the results repo.
A results repo aggregates many runs and often several projects; committing at
the bare declared path means each run silently overwrites the last, and
"results survive the box" would be true only of the most recent one. A results
repo is always pushed — durability is the entire reason it exists — so `push`
is not consulted in this mode and your code repo is never pushed to.

GitHub allows one deploy key per repository, so this needs two keys. On the box:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/code_ro   -N "" -C "gpuq code read-only"
ssh-keygen -t ed25519 -f ~/.ssh/results_rw -N "" -C "gpuq results write"
cat >> ~/.ssh/config <<'CONF'
Host github-code
  HostName github.com
  IdentityFile ~/.ssh/code_ro
  IdentitiesOnly yes
Host github-results
  HostName github.com
  IdentityFile ~/.ssh/results_rw
  IdentitiesOnly yes
CONF
```

Add `code_ro.pub` to the code repo's deploy keys **without** write access, and
`results_rw.pub` to the results repo **with** write access. Then use the host
aliases in the config so each repo gets the right key:

```toml
remote         = "git@github-code:you/myproject.git"
results_remote = "git@github-results:you/myproject-results.git"
```

Verify the read-only key really is read-only — a deploy key added with write
access by accident defeats the whole arrangement:

```bash
ssh $BOX 'cd /workspace/checkouts/myproject && git push --dry-run origin HEAD'
# expect: ERROR: Write access to repository not granted
```

### Letting gpuq file bugs against itself

Optional, and off unless you add `[autofix]` to the config. When gpuq's own
code raises — a bad worktree, a `git_ops` failure — the runner files a GitHub
issue carrying the traceback, and a workflow opens a PR against it. Caller
faults never file: an OOM, a timeout, a plain `exit N`, or a declared
artifact your job did not produce are your problem, not the queue's.

Three things to set up, once:

1. **A fine-grained PAT**, on github.com → Settings → Developer settings →
   Fine-grained tokens. Scope it to this repository alone, and give it
   `Issues: Read and write` and nothing else. Explicitly **not** `Contents` —
   anyone who can queue a job on this box can use this key, the same
   argument made for the results key above. Unlike that key, though, the
   issue body this one files is also the fixer's prompt, and the `JobSpec`
   inside it is caller-supplied and unfiltered — so the worst case here is
   not just issue spam, but issue spam plus a prompt an attacker could try
   to steer. What bounds that is outside this token entirely: branch
   protection on `main`, so nothing the fixer writes lands unreviewed, and
   your own read of the PR before you merge it. This token itself still
   cannot push — confirm that below — but "cannot push" is not the same
   claim as "worst case is issue spam"; treat the PR the same way you would
   treat one from any other outside contributor.

   **This repository is public**, so this is no longer a conditional
   warning: every issue autofix files here is world-readable, immediately
   and permanently. The issue body embeds the failing job's `JobSpec`,
   which includes its `cmd` verbatim. A `JobSpec` has no environment field,
   so nothing is copied out of the box's environment by construction — but
   a job submitted with a secret in its command line publishes that secret
   to everyone. Pass credentials through the environment, never as an
   argument.

   The job's stderr is *not* published, which is worth knowing because it
   is the likelier leak: `_describe_failure` writes up to 4 KB of it into
   `spec.error`, and the issue body dumps the whole `JobSpec`. The two
   paths are mutually exclusive by construction — `_describe_failure` runs
   only when `spec.error is None` on a caller fault, and caller faults
   never file — so a trainer that prints an API key to stderr does not
   publish it. Do not rely on that for `cmd`, which is published.

   If that trade is wrong for your jobs, `[autofix].repo` takes any
   `owner/name` and does not have to be the repository the code lives in:
   point it at a private one and the evidence stays private. The catch is
   that the issue is what triggers the workflow, so `autofix.yml` has to
   live in that repository too.

   Confirm it cannot push:

   ```bash
   GH_TOKEN=<the-pat> gh api -X PUT \
     repos/<owner>/gpu-queue-management/contents/probe.txt \
     -f message=probe -f content=eA== 2>&1 | head -3
   # expect: HTTP 403 — Resource not accessible by personal access token
   ```

2. **Export it where supervisord can see it**, so the runner inherits it.
   It is deliberately not written into `supervisor/gpuq-runner.conf` itself:
   that file is world-readable and lands in `/etc`, and `%(ENV_X)s`
   references there fail supervisord's config *parse* — not just the
   substitution — if the variable is unset anywhere in supervisord's own
   environment, which would stop the runner starting on every box that has
   never heard of autofix. Exporting it one level up avoids both problems:

   ```bash
   echo 'export GPUQ_GITHUB_TOKEN=<the-pat>' >> /etc/default/supervisor
   supervisorctl restart gpuq-runner
   ```

   The name matters, and `GPUQ_GITHUB_TOKEN` being unset means *no
   credentials* rather than "whatever else this box is logged in as". `gh`
   reads `GH_TOKEN` and `GITHUB_TOKEN` on its own and falls back to
   `gh auth login`'s stored credentials after that, so a box where anyone
   had ever run `gh auth login` would otherwise file issues as that person,
   with that person's permissions — and the 403 you just confirmed would be
   describing a token that is not the one in use. gpuq strips both variables
   from `gh`'s environment when `token_env` is unset; an unconfigured box
   logs a warning and runs jobs exactly as before.

3. **The OAuth token for the Action**, from `claude setup-token` on your own
   machine, stored as the repository secret `CLAUDE_CODE_OAUTH_TOKEN`. It
   draws on your Max subscription. It lives only as a repo secret and never
   goes near the box.

The runner creates the labels it needs (`gpuq-auto`, `gpuq-reported`,
`throttled`, `fix-me`) the first time it files.

A bug filed past the daily dispatch cap carries **both** `gpuq-auto` and
`throttled`, so it still turns up in a `label:gpuq-auto` triage query even
though no run happened for it. That second label is also why the workflow
will not pick it up just because you add `fix-me`: the gate skips anything
labelled `throttled` regardless of what else is on the issue. To dispatch a
throttled bug by hand, **remove `throttled` first, then add `fix-me`** — in
that order, since adding `fix-me` while `throttled` is still present does
nothing.

**The off switch** is a repository variable, not a commit: set
`GPUQ_AUTOFIX` to `off` (also accepted: `false`, `no`, `disabled`, `0`, any
case) under Settings → Secrets and variables → Actions → Variables. Filing
continues; dispatching stops. It can be flipped from the GitHub mobile web
UI. Leaving the variable unset is the default state of every repo and keeps
autofix on, exactly as before this variable was ever set.

**Branch protection on `main`** is what makes this safe to leave on. The
Action opens PRs; you merge them.

What is actually configured, so you can check this paragraph rather than
trust it: a pull request is required (zero approvals, so a solo maintainer
is not locked out of their own repository), `test (3.11)` and `test (3.12)`
must pass, and force-pushes and deletions are refused. Admins are exempt —
that is an escape hatch for you, not for the Action, whose token is not an
admin and is not exempt.

```bash
gh api repos/<owner>/gpu-queue-management/branches/main/protection \
  --jq '{pr_required: (.required_pull_request_reviews != null),
         checks: .required_status_checks.contexts,
         force_push: .allow_force_pushes.enabled}'
# expect: pr_required true, both test contexts, force_push false
```

Note that this setting is unavailable on a private repository on a free
plan — GitHub 403s it with "Upgrade to GitHub Pro or make this repository
public". Enabling it here is why this repository is public. If you fork
this into a private repo on a free plan, **this control does not exist for
you**, and the paragraph above is false until you either pay for it or
accept that nothing enforces the review.

## 5. Verify

```bash
ssh $BOX 'export PATH=/venv/main/bin:$PATH GPU_CLAIM_DIR=/workspace/lock/gpu
  supervisorctl status gpuq-runner
  gpu-claim -- nvidia-smi -L                 # holds the lock around a command
  gpu-claim --status                         # who holds the card right now
  gpuq list'
```

Then a real job, end to end:

```bash
ssh $BOX 'export PATH=/venv/main/bin:$PATH QUEUE_ROOT=/workspace/queue
  cd /workspace/checkouts/myproject
  gpuq submit --project myproject --commit "$(git rev-parse HEAD)" --branch main \
    --lane gpu --artifact runs/summary.json --wait --poll 2 -- python train.py'
```

Worth confirming once per box, since these are the properties the design rests on:

- **Two GPU jobs never overlap.** Submit two; check the runner log timestamps.
- **Preflight refuses a busy card.** Start a CUDA process *outside* the queue,
  then `gpu-claim -- true` — expect exit 69 naming the pid.
- **UUID agreement**, if the box has torch. Both spellings must normalize to one
  lock file, or two tools will take two locks on one card:
  ```bash
  python -c "import torch;from gpuqueue.gpuid import normalize_gpu_uuid,gpu_key;
  print(normalize_gpu_uuid(str(torch.cuda.get_device_properties(0).uuid)) == gpu_key())"
  ```

## Gotchas

Things that have actually bitten, and what they look like:

- **`gpuq-runner: ERROR (no such file)`** — supervisord runs with its own PATH,
  which will not contain a venv's bin directory. The shipped program file
  invokes an absolute interpreter (`@PYTHON@ -m gpuqueue.cli_runner`) for this
  reason. If you hand-edit it to a bare console-script name, this is the only
  symptom you get.
- **No bare `python`.** Modern images often ship only `python3`. Affects job
  `cmd`s (fix with `venv` in the project config) — `bootstrap.sh` itself already
  uses `$PYTHON` throughout.
- **`dubious ownership`** after rsync — see step 2.
- **A job outlives a killed runner.** Jobs run in their own session so the
  runner can kill a whole process group, which also means they survive it. While
  such an orphan is *alive*, nothing enforces its `timeout_s` — the watchdog died
  with the runner, and the reaper deliberately will not touch a job whose pid is
  alive. Once it exits, the reaper requeues the job once and normal service
  resumes. Preflight is what stops a new job colliding with it in the meantime.
- **Rebuilding a box loses `$QUEUE_ROOT`.** Only committed artifacts survive.
  That is the design's position, not an oversight.

## Boxes

Record deployments here so the next person knows what to expect. The row below
is an example of the useful level of detail — replace it with your own.

| Alias | Hardware | Interpreter | Notes |
|---|---|---|---|
| `<your-box>` | e.g. RTX 4060, 8 GB | e.g. `/venv/main/bin/python` 3.12, torch 2.x | Hosted unprivileged container, root. `nvidia-smi` enumerates compute apps, so preflight is a real guard — record this either way, it is the difference between a guard and a warning. Verified `<date>` |

The one field worth being precise about is whether `nvidia-smi` could enumerate
compute apps, because it varies by image and decides whether preflight actually
protects you. See [Preflight](#preflight-is-only-as-good-as-nvidia-smi).
