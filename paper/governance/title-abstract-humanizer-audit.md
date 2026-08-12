# Title and abstract language audit

Skill applied: `humanizer`. Numerical content, sampling grain, and claim force
remain locked by the claim-evidence map. This pass changes framing and cadence,
not scientific scope.

## Candidate title

> Ready Cohorts: Bounding GPU Opportunity and Avoiding Host Round Trips in
> LLM-Agent Control

The title names the two measured contributions. "Bounding" refers to
`F`, `P*`, and `U`; "avoiding host round trips" refers to the matched resident
mechanism. It does not claim a completed online runtime, CPU superiority, or
algorithmic priority.

## Draft before the final rewrite

> Between model and tool calls, an LLM-agent runtime routes events and updates
> state. Whether a GPU helps depends on two quantities: how often a grouping
> supplies enough events before their launch deadlines, and how much
> host-observation overhead is paid between device transitions. We call their
> interface the ready-cohort boundary. For zero service time, unlimited
> capacity, and equal relative launch deadlines, a specialized dynamic program
> computes the exact offline maximum P*. The model brackets it between
> fixed-partition eligibility F and a local-overlap upper bound U. [...] These
> studies are conditional and descriptive. They do not instantiate a matched
> online runtime, establish CPU displacement, or estimate a hardware or
> workload population effect.

## What makes the below so obviously AI generated?

- The abstract opens with taxonomy instead of the concrete systems action.
- Nearly every result is followed immediately by a disclaimer, which makes the
  prose read like a rebuttal checklist.
- Repeated phrases such as "separate mechanism test," "separate calibration,"
  and "these studies" expose the document's construction process.
- The final paragraph summarizes what the work did not do instead of stating
  the two positive gates it established.
- The sentences have similar length and clause structure, so the numerical
  evidence lands with little emphasis.

## Final rewrite decisions

1. Open on the deterministic transition and the exact question asked.
2. State the mathematical, trace, positive mechanism, and negative mechanism
   results in that order.
3. Keep route-key identity and joined-runtime scope in one sentence each.
4. End on the positive synthesis: cohort supply and observation placement are
   measurable gates for GPU agent control.
5. Move detailed threats to one scope-of-inference section.

The final abstract is
`paper/arxiv/sections/00_abstract.tex`. It contains no promotional priority
language, no Unicode em dash or en dash, and no population or end-to-end claim.

## Manuscript-wide audit

- Replaced warning-first framing with result-first framing.
- Kept one coined term: ready-cohort boundary.
- Consolidated repeated caveats into `Scope of inference`.
- Preserved the zero-opportunity regimes and failed nested-launch treatment.
- Preserved placement as the outer unit and batch-average timing as the metric
  grain.
- Kept the trace and resident measurements numerically separate.
- Replaced the red warning palette in the evidence map with solid measured
  components and neutral dashed future components.
