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
| supervisor + `conf.d` | How the runner is kept alive | `./bootstrap.sh --no-supervisor` and run `gpuq-runner` yourself |
| `/workspace` writable | Default prefix for queue, locks, checkouts | Set `GPUQ_PREFIX` to somewhere writable |

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

## 4. Declare your projects

Edit `$GPUQ_CONFIG` on the box, then rerun `bootstrap.sh` to clone them:

```toml
[queue]
root = "/workspace/queue"
cpu_slots = 4          # not the core count; see docs/design.md
claim_dir = "/workspace/lock/gpu"

[project.myproject]
remote   = "git@github.com:me/myproject.git"
checkout = "/workspace/checkouts/myproject"
venv     = "/venv/main"      # jobs get this on PATH
commit_artifacts = true
```

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

   One thing to know before pointing this at a **public** repo: the issue
   body embeds the failing job's `JobSpec`, which includes its `cmd`
   verbatim. A `JobSpec` has no environment field, so nothing is copied out
   of the box's environment by construction — but a job submitted with a
   secret in its command line publishes that secret. Pass credentials
   through the environment, never as an argument, on any box filing into a
   repo that is not private.

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
