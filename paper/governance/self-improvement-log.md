# Self-improvement log

This log records corrections to the research process. It does not claim that an
agent autonomously became more capable.

## SI-001: independent unit correction

- Defect: Early summaries risked treating thousands of timing rows as
  independent performance replications.
- Evidence: The statistical audit reconstructed 18 fresh launches in the
  earlier hardware study and four named placements in resident-policy-001.
- Change: All paper performance claims use placement as the outer unit. The
  current resident result is descriptive and has no population p-value.
- Guard: `AGENTS.md` prohibits row-level population inference; the paper checker
  rejects known inflated-sample phrases.

## SI-002: exact scheduling scope correction

- Defect: Fixed-window eligibility was initially at risk of being called a
  universal deadline ceiling.
- Evidence: A legal cohort can cross an origin-aligned window boundary.
- Change: The paper distinguishes fixed-window share `F`, exact offline optimum
  `P*`, local upper bound `U`, and online achievement `A`.
- Guard: The formalism states `F <= P* <= U` only under the explicit model and
  reserves "ceiling" for `P*` or a proved upper bound.

## SI-003: floating-point exactness correction

- Defect: The first MILP helper used an epsilon and could admit a launch before
  an event release at close timestamps.
- Evidence: An adversarial two-event counterexample reproduced the failure.
- Change: The equal-relative-deadline implementation normalizes to integer
  nanosecond ticks and uses a specialized dynamic program.
- Guard: Brute-force agreement and boundary tests cover integer-clock behavior.

## SI-004: mechanism correction

- Defect: Device-side graph launch was initially treated as a candidate win by
  itself.
- Evidence: The fixed nested device graph was slower in all 60 observed cells,
  with ratios from 1.075 to 1.994.
- Change: The nested graph is a negative ablation. The positive treatment must
  remove a matched host synchronize, copy, and dispatch decision.
- Guard: The claims ledger marks the fixed nested mechanism falsified.

## SI-005: metric-grain correction

- Defect: P95 and P99 summaries could be read as raw invocation tails.
- Evidence: Each recorded row contains an average over many invocations.
- Change: Columns and prose now say "quantiles of batch-average rows."
- Guard: The analyzer and paper checker prohibit raw-tail wording for these
  fields.

## SI-006: follow-up design rejection

- Defect: The first resident-policy-002 draft combined completed-only stopping,
  long method periods, five seed clusters, and a weak nuisance-variance bound.
- Evidence: Independent systems and statistical reviews found outcome-dependent
  sampling, infeasible artifact volume, carryover confounding, and no defined
  intention-to-run failure value.
- Change: The draft is blocked. A bounded instrumentation qualification will
  precede any online-runtime placement study.
- Guard: Fixed scheduled IDs, hard timeouts, block CPU counters, provider/day
  blocking, and assurance simulation are now required.

## SI-007: launch deadline versus completion SLO

- Defect: A follow-up draft reused 50 ms as an invocation-completion SLO even
  though the trace model defines a latest admissible launch.
- Evidence: The formal model and trace preregistration specify launch time.
- Change: The manuscript keeps launch eligibility separate from completion
  latency.
- Guard: The checker flags the prohibited phrase "50 ms completion SLO."

## SI-008: separate evidence is not a joined runtime

- Defect: The shared values `K=256` and `H=32` made it tempting to read the
  trace replay and resident-policy pilot as one end-to-end experiment.
- Evidence: The trace threshold is swept rather than measured for the resident
  source, and the trace model has no 32-epoch observation-free horizon.
- Change: The title names packing and decisions as two components. The abstract,
  results, and conclusion say they are separate feasibility studies.
- Guard: The claim map marks matched `K`, online `A`, and end-to-end benefit as
  unmeasured; the paper checker requires this limitation in the abstract.

## SI-009: unresolved author metadata

- Defect: Prior manuscripts disagree between KAIKAKU.AI and Independent
  Researcher for Josef Chen.
- Evidence: The prior-paper audit verified both forms in source LaTeX, and the
  author explicitly confirmed "Independent Researcher" and a private submitter
  contact address on 12 August 2026.
- Change: The public title block and PDF metadata identify Josef Chen as an
  Independent Researcher. Contact is kept only in the private arXiv handoff.
- Guard: The paper checker requires the public name and affiliation, rejects a
  pending-confirmation placeholder, and rejects an email address in public
  manuscript source.

## SI-010: scheduling-priority and complexity correction

- Defect: The local formalism described the exact evaluator as near-linear and
  linear after sorting, which overstated the frozen implementation and risked
  rediscovering established clustering structure.
- Evidence: An independent algorithms audit connected the problem to line
  `r`-gathering with outliers, microaggregation with suppression, and unit
  interval clique packing. It also found one binary boundary search per event.
- Change: The paper distinguishes an attainable grouped `O(N log N)` evaluator
  from the frozen code's `O(NR + sum_r n_r log n_r)` bound and presents the
  recurrence as a fixed-diameter trace oracle with no priority claim.
- Guard: Related work cites the closest formulation, and the checker rejects
  "linear after sorting" and algorithm-priority phrases.

## SI-011: route labels are not semantic compatibility

- Defect: Early prose treated the trace's exact route key as if it proved that
  events could share one executable transition.
- Evidence: The extractor derives the key only from outcome class and tool
  names. The generic multi-tool key covers 325 spans and 122 distinct tool-name
  encodings; the key omits state-machine, schema, arguments, and policy context.
- Change: Results are now route-key-conditioned opportunity. Semantic fusion is
  explicitly unverified and outside the current trace evidence.
- Guard: RC-015 and the abstract require the proxy limitation; the online
  preregistration requires a verified executable route identity.

## SI-012: source-revision binding

- Defect: The extraction manifest named a dataset revision while its shard URLs
  used a mutable conversion branch.
- Evidence: Independent retrieval resolved conversion commit
  `f7c94012d0bfbf66fe4d6ed627699508bbb555ff`; all 19 local, manifest, and
  commit-resolved remote SHA-256 values match.
- Change: The manifest and extractor use commit-resolved URLs and record both
  dataset and conversion revisions.
- Guard: The paper generator rejects a changed conversion revision or an
  incomplete 19-of-19 source verification record.

## SI-013: online denominator and threshold contract

- Defect: Online share `A` used an ambiguous denominator, and the smallest
  observed winning cohort was treated as a globally monotone threshold.
- Evidence: Dropped or failed events could otherwise make `A > P*`; measured
  performance need not remain profitable at every larger batch size.
- Change: `A` uses the identical fixed event set `E` with failures and fallback
  retained. `K_r` is defined as the start of a measured safe suffix over a
  declared range, with extrapolation labeled as a model assumption.
- Guard: The online preregistration freezes fixed attempts, an ITT denominator,
  safe-suffix qualification, and maximum measured cohort size.

## SI-014: the online comparator must inhabit the oracle's feasible set

- Defect: Using the same event denominator was initially treated as sufficient
  to conclude `A <= P*`.
- Evidence: An online implementation could otherwise count batches that violate
  the oracle's route grouping, threshold, or launch deadline.
- Change: The proposition now requires every batch counted by `A` to satisfy the
  same event-set, route, threshold, and deadline constraints. Finite service and
  capacity may only make the online feasible set stricter.
- Guard: The online qualification freezes executable route identity, measured
  safe-suffix thresholds, launch deadlines, and fixed attempt IDs before `A` is
  computed.

## SI-015: scientific review is not a release freeze

- Defect: A locally passing paper checker could be mistaken for a reproducible
  arXiv release, even though it regenerates artifacts before validation and the
  manuscript is not yet present in an immutable clean checkout.
- Evidence: The final factual audit passed the scientific content but separately
  blocked release on author metadata, public-receipt redaction, bundle-wide
  secret scanning, clean-checkout reconstruction, and immutable source-to-PDF
  binding.
- Change: The state and checklist now say "scientifically reviewed working
  draft; arXiv release blocked." The reviewed PDF hash is recorded separately
  from any future release hash.
- Guard: A release candidate must be rebuilt without mutation from frozen bytes
  and receive a new final review after the public bundle is fixed.

## SI-016: rebuttal language leaked into the narrative

- Defect: Independent audits correctly narrowed the claims, but the manuscript
  repeated those boundaries in the abstract, introduction, interpretation,
  related work, limitations, and conclusion. The result read as an apology for
  a pilot rather than a direct account of what was proved and measured.
- Evidence: The rendered first page ended the abstract with three unsupported
  outcomes, the introduction opened its final paragraph with "No current
  result," and the conclusion repeated the same exclusions. The architecture
  figure also used two red dashed boxes labeled "unmeasured."
- Change: The abstract and conclusion now lead with the exact trace and
  mechanism results, detailed caveats are consolidated in `Scope of inference`,
  and the evidence map uses neutral future boxes. The title names both measured
  contributions.
- Guard: The checker rejects the former "conditional and descriptive" abstract
  sentence. The release checklist requires a fresh voice and factual review
  after this reframing.
