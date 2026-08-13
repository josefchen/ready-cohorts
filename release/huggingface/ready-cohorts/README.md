---
pretty_name: Ready Cohorts Processed Evidence
language:
  - en
tags:
  - llm-agents
  - gpu-systems
  - cuda-graphs
  - batching
  - reproducibility
---

# Ready Cohorts: processed evidence

This is the public processed-evidence mirror for *Ready Cohorts: Bounding GPU
Opportunity and Avoiding Host Round Trips in LLM-Agent Control* by Josef Chen,
Independent Researcher.

- Paper: <https://arxiv.org/abs/2608.12123>
- Hugging Face Paper page: <https://huggingface.co/papers/2608.12123>
- Code: <https://github.com/josefchen/ready-cohorts>
- Frozen v1 source and release bundle:
  <https://github.com/josefchen/ready-cohorts/releases/tag/ready-cohorts-arxiv-v1>
- Interactive results explorer: <https://huggingface.co/spaces/josefchen/ready-cohorts>
- This mirror: <https://huggingface.co/datasets/josefchen/ready-cohorts>
- Input trace dataset at the pinned conversion commit:
  <https://huggingface.co/datasets/Exgentic/agent-llm-traces/tree/f7c94012d0bfbf66fe4d6ed627699508bbb555ff>

The input trace bytes are pinned to dataset revision
`70036b93a04e61b0ea2706a68b962f4f26774587` and Parquet conversion revision
`f7c94012d0bfbf66fe4d6ed627699508bbb555ff`. This mirror does not redistribute
prompt text, tool arguments, tool results, API keys, or provider receipts.

## Contents

- `source/`: source provenance, session summaries, and derived span features.
- `trace/`: the 540 cell-seed evaluations and summary surface for exact
  equal-relative-deadline packing.
- `resident/`: four-placement summaries and paired mechanism contrasts for the
  device-resident decision study.
- `native/`: five-placement summaries and paired contrasts for the fixed nested
  device-launch negative control.
- `paper/`: the release PDF, paper-data manifest, and claim-evidence map.
- `figures/`: the three data figures used by the paper.
- `MANIFEST.json`: source and public SHA-256 values for every mirrored file.

The resident and native timing rows are technical repetitions. Placement is the
outer performance unit. The trace replay is conditional on one 851-session
panel, one arrival model, and nine generated swarms. These files do not support
hardware-population significance or an end-to-end agent-speedup claim.

## Headline cells

At the prospectively frozen trace condition with 100,000 target active
sessions, `K=256`, and a 50 ms launch deadline, the fixed-window share is
30.19%, the exact offline share is 43.00%, and the local upper bound is 45.85%.
In the separate mechanism study, the device-resident path is faster than the
matched host-round-trip path in all 36 named-placement cells; within-placement
row-median ratios range from 1.19x to 2.39x.

## Citation

```bibtex
@misc{chen2026readycohortsboundinggpu,
  title         = {Ready Cohorts: Bounding GPU Opportunity and Avoiding Host Round Trips in LLM-Agent Control},
  author        = {Josef Liyanjun Chen},
  year          = {2026},
  eprint        = {2608.12123},
  archivePrefix = {arXiv},
  primaryClass  = {cs.DC},
  url           = {https://arxiv.org/abs/2608.12123}
}
```

No additional license is asserted by this mirror; upstream source data remain
subject to their source repository terms.
