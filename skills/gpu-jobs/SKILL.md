---
name: gpu-jobs
description: Use when running training, evaluation or any GPU work on this box, and when a long CPU job would otherwise block you - submit it to the queue instead of running it directly, then either wait for it or go do something else. Covers boxes with no GPU at all, and submitting to a remote box over ssh and reading the result back. Also use when you have anything to report about the queue itself - a bug, a measurement, or a proposal about how it behaves.
---

# Running work on a shared GPU box

This box has one GPU and several people or agents who want it. Running a
training script directly will either fail on an OOM half an hour in, or make
someone else's run mysteriously slow. Submit it instead.

## Submit

    gpuq submit --project <name> --commit "$(git rev-parse HEAD)" \
      --branch "$(git rev-parse --abbrev-ref HEAD)" \
      --lane gpu --artifact runs/v0/summary.json \
      -- python -m src.train --config configs/v0.yaml

It prints a job id and returns at once. Then choose:

**You need the result now:**

    id=$(gpuq submit … )
    gpuq wait "$id"        # 0 done, 1 failed, 124 you gave up waiting

**You have other work to do:** do it, then come back and `gpuq wait "$id"`
whenever you like. If the job already finished, `wait` returns immediately —
there is no penalty for waiting late, and no need to decide up front.

Several datasets to process? Submit them all, then wait on them one at a time.
The runner will already have been working through them.

## Which lane

- `--lane gpu` — training, anything that calls CUDA. Declare `--vram-mb` and
  it shares the card; without it, it takes the whole card alone. If your code
  takes its own exclusive lock on the card, do not declare it — see below.
- `--lane cpu` — data fetching, profiling, analysis. Several at once.

Putting CPU work in the GPU lane blocks everyone else's training for no
reason. Putting GPU work in the CPU lane means several jobs hit the card at
once, which is the failure this queue exists to prevent.

## On a box with no GPU

The queue runs on CPU-only boxes too. Everything above applies, with these
differences:

- Use `--lane cpu` for every job. It is also the default. A `--lane gpu`
  job is not run on the CPU: it fails with `no usable GPU`.
- Leave out `--vram-mb`. It has no meaning in the CPU lane.
- `gpu-claim` has nothing to claim and exits non-zero. You do not need it.

Several agents can use the box at once without coordinating with each
other:

- Each job runs in its own worktree of the commit it pinned, so two jobs
  never see each other's files, even two jobs from the same repository at
  different commits.
- The runner starts at most `[queue].cpu_slots` jobs at a time (4 unless
  the box's config says otherwise) and the rest wait. Submit everything
  you have. Do not hold work back to be polite, and do not start anything
  directly to get around a full queue.
- If two agents might submit the same piece of work, give it a
  `--dedupe-key`. A second submit with the same key while the first is
  pending or running returns the first job's id and queues nothing.

A job is killed when it runs longer than `--timeout-s`, which defaults to
3600 seconds. Set it for anything that takes more than an hour.

## Submitting to a box from another machine

The queue is a directory on the box, so everything goes through ssh. Source
the box's `env.sh` first in every command. It is written by `bootstrap.sh`
at the install prefix (`/workspace` unless the box was deployed with
another), and without it `gpuq` is either not on PATH or reading the wrong
queue.

    BOX=<ssh-alias>
    id=$(ssh $BOX '. /workspace/env.sh && gpuq submit --project <name> \
      --commit <sha> --branch <branch> --lane cpu \
      --artifact out/summary.json -- python -m src.analyse')

Push the commit before you submit. The box fetches it from the project's
remote, and a commit that only exists on your machine fails the job with
"push it first".

Then check on it without holding a connection open for the whole run:

    ssh $BOX ". /workspace/env.sh && gpuq wait $id --timeout 600"
    # 0 done, 1 failed, 124 still going after 600s -- run it again later
    ssh $BOX ". /workspace/env.sh && gpuq show $id"        # spec, error, log paths
    ssh $BOX ". /workspace/env.sh && gpuq list --json"     # every job, by state

Use a bounded `--timeout` and repeat, not one `gpuq wait` for hours: a
dropped ssh connection tells you nothing about the job, which carries on
either way. Keep the job id. It is the only handle you have, and `gpuq
wait` on a finished job returns at once.

Results come back two ways:

- Declared with `--artifact`, when the project's config on the box has
  `commit_artifacts`. The runner commits those files, and pushes them if
  the project has `push` or a results repository. You read them from git.
  In a results repository they are at `<project>/<job-id>/<path>`.
- The job's stdout and stderr. `gpuq show $id` prints the paths of both
  logs, and they stay on the box after the job ends. For a small result,
  print it and read the log over ssh.

Nothing else survives. The job's worktree is removed when it finishes, so
do not plan to copy files out of it, and do not look in the directories
beside it, which belong to other agents' jobs.

Do not run the work itself over ssh (`ssh $BOX 'python -m ...'`). It
bypasses `cpu_slots`, so it slows every queued job down, and it dies with
your connection.

## Say how much VRAM you need

    gpuq submit … --lane gpu --vram-mb 900 -- python -m src.train

Without `--vram-mb` your job holds the entire card and nothing else runs
beside it — correct for a big training run, wasteful for a small one. With
it, the queue admits your job alongside anyone else's whose declaration
still fits, so two unrelated small jobs run at once instead of queueing.

The unit is MiB **as `nvidia-smi` reports them** — run the thing once and
read your process's memory out of `nvidia-smi` while it works. That number
includes the ~250 MiB CUDA context and the caching allocator's high-water
mark. `torch.cuda.max_memory_allocated()` counts live tensors only and is
always too small; declare from it and your job gets killed.

A job using more than it declared is killed, and the failure says what it
declared and what it was using. So round up, and leave room for your
largest batch — over-declaring only costs you a wait.

### Do not declare VRAM if your code locks the card

The queue only knows what you declare. It cannot see a lock your own code
takes. Some training code takes an exclusive per-card lock before it starts,
for example an `flock` keyed on the GPU UUID. Submit two such jobs with
`--vram-mb` and the queue will admit both. The second then waits on your
lock without training, and if your lock has a timeout shorter than a run,
it fails with your code's own "GPU busy" error. That failure is not an OOM
and not a queue kill.

Check before submitting, from the root of the repo you are submitting:
`git grep -nE 'flock|lockf|FileLock|gpu_lock|claim_gpu'`. That covers every
tracked file, not only `src/`, and the common ways Python takes a file lock
(`fcntl.flock`, `fcntl.lockf`, the `filelock` package). A match is not proof:
read it and see whether the lock is held around training. No match is not
proof either, if your project locks some other way. If your code takes such
a lock, omit `--vram-mb`. The job then holds the whole
card, and your jobs run one after another instead of blocking each other.
The cost is that nobody else's job shares the card while yours runs.

## Always pin the commit

`--commit "$(git rev-parse HEAD)"` is not optional. The runner checks out
that exact tree, so a number it reports can be traced to a configuration
someone can read back. A branch name alone lets the tree move under a
queued job and produce a result nobody can reproduce.

Declare what you want kept with `--artifact` (repeatable, paths relative to
the repo root). The runner collects those and can commit them; anything
else your job writes dies with the box.

## When something goes wrong

    gpuq list                  # everything, by state
    gpuq show <id>             # the spec, plus paths to its stdout/stderr logs
    gpuq cancel <id>           # only while still pending

A failed job carries its stderr tail in `error`, so `gpuq show` usually
tells you what happened without opening the logs.

A job that fails on CUDA out-of-memory is reported as such and is never
retried: it is a configuration problem, not a transient. Make the model or
the batch smaller.

## A job died with signal 9 and no message

A `SIGKILL` writes no stderr, so an `exit -9` with an empty message is
not evidence of a bug in your own code. The queue's orphan sweep kills
CUDA processes that no live claim accounts for, and that is what it
looks like from the victim's side.

Run `gpuq kills` first. If your pid is there, the queue killed it and
the `cgroup` line names what it killed — check whether the work was
running somewhere your claim did not cover. CUDA inside a container is
the usual case, and the fix is to claim for it:

    gpu-claim --vram-mb <MiB> \
      --scope-pid $(docker inspect -f '{{.State.Pid}}' <container>) \
      -- <your command>

If your pid is *not* there, the queue did not kill it and the failure is
somewhere else.

## Telling the owner something about gpuq

Use `gpuq bug` for anything you want the owner to see about the queue itself.
Not only breakage — a proposal, a measurement, a design observation, a gap in
the CLI all go through the same door:

```bash
gpuq bug "gpuq wait never returns for a cancelled job" --body "$(cat <<'EOF'
What I ran, what I expected, what happened, and the exact output.
EOF
)"
```

It does not fix anything and it does not dispatch a run. It files an issue for
the owner, who decides.

**Do not open the issue yourself with `gh issue create`.** It looks equivalent
and is not. `gpuq bug` applies the `gpuq-reported` label, and that label is the
only thing that routes the issue anywhere: it is what assigns the owner, and an
issue with no label notifies nobody at all. A hand-filed issue sits unread
indefinitely, looking filed. This has already happened once.

If `gpuq bug` says autofix is not configured, you are on a box that cannot
file — likely not the GPU box at all. Then, and only then, use `gh` directly
and apply the label by hand, because nothing else will:

```bash
gh issue create --repo <owner>/gpu-queue-management \
  --label gpuq-reported --title "..." --body "..."
```

Do **not** use either path for your own job failing: a CUDA OOM, a timeout, a
non-zero exit, or a declared artifact your script did not write are your bugs,
and the runner already tells you so. When the runner's own code raises, it
files without being asked.

## Do not

- Run GPU work directly. `gpu-claim -- <cmd>` if you truly must run
  something interactively -- and `export GPU_CLAIM_DIR=/workspace/lock/gpu`
  first, or it does *not* take the same lock the queue does. Your shell does
  not inherit the daemon's environment, so without that the claim lands in
  `/var/lock/gpu`, the runner's ledger never counts it, and it will admit a
  job on top of you. `gpu-claim` warns when it can tell.
- Run git in a checkout the runner owns. It manages those, and a concurrent
  checkout corrupts the tree under a running job.
