# gpu-queue-management

Host-level GPU arbitration and job queueing for shared single-GPU boxes.

One machine, one card, several people or agents who all want to train
something. This provides the two pieces that makes that workable: an advisory
lock so nothing accidentally shares the card, and a queue so work waits its
turn without anyone sitting on an ssh session.

It is deliberately small and deliberately boring. The queue is a directory
tree. The lock is `flock`. There is no database and no network service.

## Why it is not inside the project that uses it

A lock is only correct if every participant derives the same key and uses the
same path. Vendored copies drift, and two copies that disagree hold *different*
locks for the *same* card — isolation that looks correct and is not. One shared
installation makes that impossible rather than merely unlikely.

The queue generalizes for the same reason: the runner executes job specs and
knows nothing about any particular model or dataset, so a second project on the
same box should get scheduling without reimplementing it.

## Components

| | |
|---|---|
| `gpu-claim` | Advisory lock keyed on the GPU UUID, with a preflight that refuses to start when foreign CUDA processes hold the card. Wraps any command. |
| `gpuq` | Submit, list, inspect and cancel jobs. |
| `gpuq-runner` | Supervisor-managed daemon. Admits CPU jobs concurrently and GPU jobs one at a time, reaps dead claims, commits artifacts. |
| `bootstrap.sh` | Takes a bare box to a working runner, idempotently. |

## Status

Implemented. Install with `bootstrap.sh`; see `docs/design.md` for the
architecture and `docs/plans/` for the implementation plan.

Needs Python 3.11+ (stdlib `tomllib`). If the box's `python3` is older,
point `bootstrap.sh` at the right one with `PYTHON=/path/to/python3.11`.

Per-job VRAM limits are not implemented: a job that holds the card holds
all of it. See "Not in scope" in `docs/design.md`.

Originating context: `Daniel-T-S-Adams/wgan-synthetic`, which needs six agents
to research six datasets in parallel against a single RTX 4060.
