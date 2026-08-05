---
name: gpu-jobs
description: Use when running training, evaluation or any GPU work on this box, and when a long CPU job would otherwise block you - submit it to the queue instead of running it directly, then either wait for it or go do something else.
---

# Running work on a shared GPU box

This box has one GPU and several people or agents who want it. Running a
training script directly will either fail on an OOM half an hour in, or make
someone else's run mysteriously slow. Submit it instead.

## Submit

    gpuq submit --project <name> --commit "$(git rev-parse HEAD)" \
      --branch "$(git rev-parse --abbrev-ref HEAD)" \
      --lane gpu --artifact runs/glove/v0/summary.json \
      -- python -m src.train.train_wgan_gp --config configs/glove/v0.yaml

It prints a job id and returns at once. Then choose:

**You need the result now:**

    id=$(gpuq submit … )
    gpuq wait "$id"        # 0 done, 1 failed, 124 you gave up waiting

**You have other work to do:** do it, then come back and `gpuq wait "$id"`
whenever you like. If the job already finished, `wait` returns immediately —
there is no penalty for waiting late, and no need to decide up front.

Six datasets to process? Submit all six, then wait on them one at a time.
The runner will already have been working through them.

## Which lane

- `--lane gpu` — training, anything that calls CUDA. One at a time, box-wide.
- `--lane cpu` — data fetching, profiling, analysis. Several at once.

Putting CPU work in the GPU lane blocks everyone else's training for no
reason. Putting GPU work in the CPU lane means several jobs hit the card at
once, which is the failure this queue exists to prevent.

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

## Do not

- Run GPU work directly. `gpu-claim -- <cmd>` if you truly must run
  something interactively; it takes the same lock the queue does.
- Run git in a checkout the runner owns. It manages those, and a concurrent
  checkout corrupts the tree under a running job.
