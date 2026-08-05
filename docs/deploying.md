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

Record deployments here so the next person knows what to expect.

| Alias | Hardware | Interpreter | Notes |
|---|---|---|---|
| `tig-gpu` | RTX 4060, 8 GB | `/venv/main/bin/python` 3.12.13, torch 2.12.0+cu130 | vast.ai unprivileged container, root. `nvidia-smi` enumerates compute apps, so preflight is a real guard. Verified 2026-08-05 |
