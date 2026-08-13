# Ready Cohorts X launch thread

Attach
[`spaces/ready-cohorts/assets/ready-cohorts-social-card.png`](../spaces/ready-cohorts/assets/ready-cohorts-social-card.png)
to post 1. The image is a 1600 by 900 PNG and is generated from the released
CSV values by `scripts/build_public_visuals.py`.

## Post 1

LLM agents wait on models and tools, then execute thousands of tiny control
transitions between them.

When should that control path move to GPU?

Our new paper, Ready Cohorts, maps the boundary and finds that batching and
observation placement are separate gates. 🧵

## Post 2

At the primary trace-replay cell (C=100k, K=256, 50 ms), fixed windows made
30.19% of events eligible. Exact deadline packing reached 43.00%, below a 45.85%
upper bound.

That is +12.81 percentage points without changing the workload.

## Post 3

On four named GPU placements, keeping a binary control decision on device beat
the matched host observation + redispatch bundle in all 36 cells (host/resident
row-median ratios: 1.19–2.39×).

But fixed nested device launch was slower in all 60 control cells.

## Post 4

The result is a boundary, not “GPUs replace CPUs”:

ready events → deadline cohorts → resident GPU decision → CPU/DPU-authorized
effects.

Next: join the gates in a finite-capacity service and measure achieved share A,
CPU time, raw P99, and inference interference.

## Post 5

Paper: https://arxiv.org/abs/2608.12123

Interactive explorer: https://huggingface.co/spaces/josefchen/ready-cohorts

Code + frozen release: https://github.com/josefchen/ready-cohorts

Evidence: https://huggingface.co/datasets/josefchen/ready-cohorts

## Image description

Editorial research graphic titled “When should agent control move to GPU?” A
diagram shows ready events with routes and deadlines flowing to a resident GPU
decision, then to an effect mailbox and a CPU/DPU authority plane. The graphic
reports that exact packing raises eligible share from 30.19% to 43.00% at the
primary replay cell; the host/resident row-median ratio spans 1.19× to 2.39×;
all 36 mechanism cells favor the resident path; and fixed nested launch is
slower in all 60 negative-control cells. A note says this is not an end-to-end
speedup claim.

## Posting checks

- Every post is at most 280 characters as written.
- Attach the image only to post 1 and add the image description as alt text.
- Keep the evidence boundary in post 4; do not replace it with a deployment
  claim.
- X accepts PNG and displays a single image with an aspect ratio between 2:1
  and 3:4 in full. The launch image is 16:9 and below the 5 MB photo limit.
  See [X Help: posting pictures](https://help.x.com/en/using-x/posting-gifs-and-pictures)
  and [X Help: how to post](https://help.x.com/en/using-x/how-to-post).
