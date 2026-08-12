# Statistical design for the confirmatory research program

Status: design audit and recommendation, 2026-08-11. This document governs
future confirmatory experiments. It does not retroactively turn pilot timing
iterations into independent replications.

## Executive decision

The paper should not optimize for a collection of small `p < 0.05` results.
It should make a few difficult claims that satisfy all four conditions:

1. the uncertainty calculation uses the unit that was independently assigned
   or independently sampled;
2. a familywise confidence interval excludes the null or noninferiority bound;
3. the effect also clears a frozen minimum worthwhile effect;
4. the result reproduces across fresh placements and a sealed workload panel.

The present artifact is strong pilot evidence and an unusually good raw audit
trail. It is not yet a population-level confirmatory study. The 19,440 rows in
pilots 012--029 are 19,440 timing iterations but only 18 fresh GPU/container
launches, three per requested GPU class. The next study must spend compute on
fresh placements, workload instances, and task examples rather than adding
thousands more iterations inside the same process.

The three confirmatory programs should use these default standards:

| Program | Independent population unit | Absolute floor | Target before variance re-estimation |
|---|---|---:|---:|
| Native resident runtime | fresh GPU placement/server launch | 12 per primary hardware stratum | 30 per stratum |
| Shared-vLLM slack harvesting | fresh inference-server placement with a randomized paired baseline/shared trial | 20 in the primary H100 stratum | 30--80, selected from blinded paired-variance simulation |
| Diversity--regularity quality | held-out task/problem | 20,000 total and at least 4,000 per domain | 50,000 or enough for at least 200 reference all-wrong events |
| Diversity--regularity systems | fresh serving placement | 12 per primary hardware stratum | 20 per stratum |

These are floors, not magical guarantees. Final sample size is the smallest
precomputed size giving at least 90% power at the frozen smallest effect of
interest under a pessimistic pilot variance. If that calculation exceeds the
target, run the larger number. Unlimited timing iterations cannot compensate
for too few placements or too few distinct tasks.

## What the current evidence does and does not establish

### Experimental units in the existing artifact

| Evidence layer | Recorded rows | Correct independent unit | Permissible inference now |
|---|---:|---|---|
| Compiler/sub-256 timing pilots | 9 iterations per cell | one host/process launch | descriptive behavior of those launches |
| Hardware confirmation | 30 iterations per cell, 18 launches | fresh placement, within requested GPU class | exact behavior of 18 observed placements; preliminary deployment variation |
| Fixed-window and sliding replay | 3--5 Monte Carlo seeds per cell | replay seed conditional on one fixed 851-session empirical panel | Monte Carlo uncertainty under the frozen panel and arrival model only |
| Correctness check | one deterministic check per shape | workload shape/trajectory | an implementation invariant on checked shapes, not an estimated field failure rate |

Within a compiled timing cell, the repetitions are contiguous. The benchmark
advances persistent state from one repetition to the next, and system noise is
serially correlated. They are neither independent tasks nor fresh deployments.
The current independent percentile bootstrap of CPU and GPU iterations is
therefore correctly labelled *descriptive* in the notebooks, but it must not
be used as the confirmatory population interval.

The confirmatory hardware run did use 18 distinct GPU UUIDs according to the
captured `nvidia-smi` manifests. This supports calling them distinct GPUs.
Provider host identifiers are not captured, however, so shared-host or
same-cluster dependence cannot be ruled out. Future manifests must retain GPU
UUID, provider instance/host ID when exposed, PCI bus ID, region/zone, driver,
actual SKU, clocks/power limit, launch time, container image digest, and git
revision. Spread runs over calendar days and zones; do not launch an entire
confirmatory population in one provider burst.

### Recomputed audit quantities

The following checks were recomputed directly from the processed and raw
ledgers:

- All 18 confirmation launches have distinct GPU UUIDs, zero duplicate
  `(case_id, repetition)` rows, zero execution errors, and zero invalid shapes.
- `255 / 648` timing cells have a conventional within-cell CV above 10%; the
  median is 5.82%, p90 is 32.56%, and the maximum is 63.93%. CV is only a noise
  diagnostic here; latency ratios and quantiles should be analyzed on a log
  scale.
- Across the 72 `(requested GPU, N, H)` strata, the standard deviation of the
  three placement-level log speedups has median 0.055, p75 0.098, p90 0.219,
  and maximum 0.373. Three placements cannot distinguish these variance regimes.
- Three wins in three fresh placements give a one-sided exact sign-test
  `p = 0.125` under a 50% win probability. The two-sided 95% Clopper--Pearson
  lower bound on the underlying win probability is only 0.292. Thus “all three
  observed placements won” is true; “this class generally wins” remains a
  pilot inference.
- The R3 analysis pairs an alternative GPU's placement replicate `j` with
  H100 replicate `j`. Those indices are labels, not randomized matched pairs.
  The scientific comparison must treat placements as independent samples or
  use all-pairs/range reasoning. Fortunately, the observed conclusion is
  robust for T4, L4, A10, and L40S: each class's *maximum* observed fused cost
  is below H100's *minimum* observed fused cost. The next analysis should use
  an independent-sample hierarchical contrast rather than index pairing.
- The six-card price-versus-wall-time Spearman value is `-0.486`. Under the
  standard exact permutation null of no rank association, its two-sided
  `p = 0.356` (`p = 0.178` for a negative one-sided alternative). The frozen
  hypothesis “point estimate below 0.8” was descriptively satisfied, but it is
  not evidence for a population correlation below 0.8. Treat the six selected
  service classes as a dated descriptive comparison.
- The primary CPU reference is the faster of one- and eight-thread medians
  selected using the same timing data. This makes the point comparison
  conservative for the GPU, but ordinary intervals do not account for the
  selection. The native study must freeze the CPU implementation/thread policy
  on pilot data, or reselect it independently inside each training block and
  evaluate on a separate measurement block.
- Cloud manifests have complete software versions and `nvidia-smi` telemetry,
  but most have no git revision. A source tarball or immutable image digest is
  required for every future cloud run.

### Trace-replay uncertainty has two distinct layers

The five Poisson replay seeds quantify simulation noise conditional on the
same 851 sessions. They do not quantify how eligibility would change for a new
set of tasks, domains, agent frameworks, or production arrival processes.
Future trace intervals need an outer, session-cluster bootstrap, stratified by
benchmark/harness, and an inner arrival-process seed. Report separately:

1. Monte Carlo error under the frozen trace panel;
2. empirical-panel sampling sensitivity;
3. between-corpus/domain variation;
4. arrival-model sensitivity, which is a design analysis rather than sampling
   error.

No p-value is needed for mathematical invariants such as `F <= P* <= U`, a
grouping refinement inequality, or exact replay reproduction. Those must pass
for every tested case. Statistical uncertainty applies to how often the
quantities occur in a workload population, not to the proof.

## Common confirmatory model

### Hierarchy and pairing

The default systems hierarchy is:

```text
provider / region / day
    fresh physical GPU or server placement
        randomized implementation period
            independent workload or request-trace seed
                request, event, or timing iteration
```

Treatment is assigned at the implementation-period level, but the population
claim is normally about new deployments. Analyze a paired contrast within
placement, then generalize using the distribution of placement contrasts.
Requests and timing iterations improve the precision of a placement summary;
they do not increase the deployment sample size.

Every placement should run all methods in a balanced randomized order. Use a
Williams or Latin-square order across placements, reset deterministic state
at the start of each block, use common request/trace seeds across methods, and
insert a frozen warm-up and cache-reset policy. Record the exact order. A/B
periods must be long enough that P99 is estimated from at least 10,000 and
preferably 20,000 eligible requests; use more for P99.9.

### Primary estimands

For a positive metric such as latency or cost, define the placement-paired log
ratio

```text
theta = E_placement,workload[ log(Y_new / Y_baseline) ].
```

`exp(theta)` is the geometric-mean ratio over the explicitly defined placement
and workload population. Report its confidence interval and a placement-level
prediction interval. The prediction interval answers the systems question a
mean cannot: how variable might the effect be on the next deployment?

For tail latency, first calculate P95/P99 within each sufficiently long trial,
then compare those trial-level quantiles. Do not pool all requests across
placements; a fast placement must not dominate by emitting more requests.
Use a moving-block bootstrap only to describe uncertainty within a trial. The
confirmatory interval uses paired placement/trial summaries.

For task success and all-wrong events, the unit is the distinct held-out task.
Use paired task outcomes because every method sees the same task. Repeated
samples from one prompt are nested stochastic draws, not new tasks.

### Uncertainty method

Use two complementary analyses, frozen before data collection:

1. **Primary randomization/cluster analysis.** Paired placement log contrasts
   with a studentized randomization test or t interval when the number of
   placements is at least 20. With fewer or visibly heavy-tailed placements,
   use a paired permutation test plus a BCa/percentile-t cluster bootstrap and
   show both. Never bootstrap timing rows as if they were placements.
2. **Hierarchical sensitivity model.** A multilevel model with fixed effects
   for implementation and preregistered hardware/load strata, random effects
   for placement and trace/task block, and implementation-by-placement random
   slopes. This estimates heterogeneity; it must not replace the design-based
   primary analysis.

For crossed placement and workload sampling, use a two-way cluster bootstrap:
resample placements and trace/task seeds independently, retain all nested
events, and recompute the complete statistic. For quality panels spanning
domains, use a stratified task bootstrap and report the aggregate plus every
domain rather than an unweighted average of domain percentages.

Use 95% two-sided intervals for estimation. Directional superiority and
noninferiority decisions use one-sided alpha 0.025, leaving the conventional
confidence statement easy to audit. Report exact adjusted p-values only as a
secondary view; effect sizes and intervals are primary.

### Failures, timeouts, and missingness

The confirmatory estimand is intention-to-run:

- compilation failure, OOM, server crash, or deadline timeout is a system
  outcome, not a timing row to exclude;
- encode a timeout at the frozen censoring/SLO limit and count it as an SLO
  failure; additionally report completed-request latency;
- retry only failures classified in advance as provider provisioning failures
  before the benchmark starts; keep both launch records;
- if instrumentation corrupts a run, invalidate the entire randomized block
  using a treatment-blind rule and replace the block with a fresh placement;
- publish a CONSORT-like launch ledger: requested, provisioned, started,
  completed, invalidated, retried, and analyzed.

## Practical margins and success rules

“Not significant” never means equivalent. Positive claims must clear a
minimum worthwhile effect; no-harm claims require a noninferiority or
equivalence test against a frozen margin.

| Claim | Primary contrast | Frozen practical requirement |
|---|---|---|
| Native runtime performance | strongest resident runtime vs strongest frozen legal baseline | at least 15% throughput improvement or equivalently speedup `>= 1.15`; the adjusted lower confidence bound must exceed 1.0, and the point estimate must exceed 1.15 |
| Native runtime CPU displacement | CPU core-seconds per valid event | at least 25% reduction with adjusted upper ratio bound below 1.0 |
| End-to-end no harm | task-completion-time ratio | noninferior at `+2%` (`new / baseline < 1.02`) |
| Task utility no harm | paired task-success difference | lower bound above `-1.0` percentage point |
| Shared-GPU inference no harm | P99 TTFT and P99 TPOT ratios | both one-sided upper bounds below `1.02`; also freeze an absolute SLO guardrail and use whichever is stricter |
| Shared-GPU SLO attainment | shared minus solo SLO-hit share | lower bound above `-0.5` percentage point |
| Shared-GPU useful work | valid control throughput under the no-harm constraint | lower bound exceeds 10% of the standalone cheap-control-GPU rate, or displaces at least two measured CPU cores; choose one before confirmation |
| Joint ensemble vs quality-only selector | all-wrong probability | noninferior within `min(0.5 pp, 20% of the development-set baseline beta)` |
| Joint ensemble vs quality-only selector | P95 task-completion time | at least 10% improvement |
| Joint ensemble vs systems-only selector | all-wrong probability | at least the frozen development-derived absolute improvement, default 0.5 pp when baseline beta is at least 1% |
| Joint ensemble vs systems-only selector | P95 task-completion time | noninferior within `+5%` |

For the native runtime, the most defensible decision combines statistical and
engineering evidence: adjusted lower speedup bound above 1.0 *and* point
estimate at least 1.15. Requiring a confidence bound above 1.15 is a stronger
claim and may be used if pilot effects are large. State which rule is frozen;
do not switch after seeing results.

Correctness is a gate, not a tradeable endpoint. The production runtime must
have deterministic sequence/commit semantics and per-step equality or an
explicitly proved numeric contract. Any unexplained action/trajectory mismatch
halts that implementation version. A zero-error empirical run can supplement
the proof: with zero failures in `M` independent episodes, the one-sided 95%
exact upper failure-rate bound is `1 - 0.05^(1/M)`, but episodes sharing one
deterministic route bug are not independent.

## Power and sample-size strategy

### Why placements dominate

Using the observed hardware study only as a planning distribution, the p90
placement-level log-speedup standard deviation is 0.219. Approximate
one-sample/paired t-test counts below give 90% power at one-sided alpha 0.025:

| SD of paired log ratio | Detect 10% | Detect 15% | Detect 20% |
|---:|---:|---:|---:|
| 0.10 | 14 | 8 | 6 |
| 0.15 | 29 | 15 | 10 |
| 0.22 | 58 | 29 | 18 |

This makes 30 placements per primary stratum a nominal starting point for the
future **online route-compacting runtime**, not a license to scale the current
mechanism microbenchmark. Under the simple planning approximation it targets a
15% effect near the current p90 variance; it is not a power guarantee, and a
mere 10% effect at that variance would require about 58. Final size must be
recomputed from the paired new-runtime contrast and rounded to complete
provider, day, and schedule blocks.

For a 2% shared-vLLM noninferiority margin when the true paired effect is zero,
the approximate counts for 90% power are:

| SD of paired log-P99 ratio | Placements |
|---:|---:|
| 0.01 | 5 |
| 0.02 | 13 |
| 0.03 | 27 |
| 0.05 | 69 |
| 0.08 | 174 |

Tail noninferiority can therefore require far more runs than a mean-throughput
paper. Pairing the same request trace on the same placement is essential. Do
not weaken the 2% margin merely because the experiment is noisy; improve the
blocking or run more placements.

### Simulation-based planning protocol

Before each confirmation:

1. run at least 12 fresh, blinded nuisance placements per intended primary
   stratum, blocked across the intended providers and days, and exclude them
   from confirmatory inference;
2. estimate only nuisance quantities needed for power: placement random-slope
   variance, within-period block variance, P99 density, binary discordance,
   failure rate, and intraclass correlation;
3. use a 90% or 95% assurance calculation, or integrate nuisance-variance
   uncertainty directly, rather than treating a low-confidence variance bound
   as known;
4. simulate the exact randomized layout, missing/failure mechanism, primary
   statistic, endpoint covariance, provider/day blocks, and multiplicity
   procedure for at least 100,000 synthetic studies;
5. choose the smallest `N` whose lower Monte Carlo confidence bound is at least
   90% power at the SESOI and whose familywise type-I error is at most 5%; add
   a 10% reserve only for pre-start provisioning failures, while retaining
   post-start failures as intention-to-run outcomes;
6. freeze `N`, seeds, configs, image digest, and analysis commit before
   confirmatory outcomes are exposed.

A blinded sample-size re-estimation may pool paired-contrast variance without
revealing the mean effect. It may increase but never decrease the frozen floor.
If effect-dependent early stopping is desired, preregister a group-sequential
alpha-spending design. The default is simpler: no efficacy peeking and no
optional stopping.

### Rare co-failure events

For a paired binary comparison, power is driven by the number of discordant
tasks, not the marginal accuracy alone. Approximate tasks required for 90%
power at two-sided alpha 0.05 are:

| Discordant share | Detect 0.5 pp difference | Detect 1.0 pp | Detect 2.0 pp |
|---:|---:|---:|---:|
| 2% | 8,402 | 2,098 | -- |
| 5% | 21,011 | 5,250 | 1,310 |
| 10% | 42,026 | 10,504 | 2,623 |
| 20% | 84,056 | 21,011 | 5,250 |

These normal approximations must be replaced by a paired Bernoulli simulation
using development-set discordance before freezing the panel. If the reference
all-wrong probability is rare, also target at least 200 reference all-wrong
events; that gives roughly 14% relative 95% precision in the simple binomial
case. At `beta = 0.1%`, this alone requires about 200,000 distinct tasks.

## Experiment 1: native device-resident runtime

### Confirmatory question and population

Can the resident runtime improve deployment-level control performance and CPU
consumption over the strongest legal tuned baseline while preserving exact
semantics? The target population is fresh instances of the two frozen service
classes and the sealed trace/workload mixture, not “all GPUs” or “all agents.”

Use L4 and H100 as the two primary hardware strata. A broad cross-hardware
claim requires both to pass. If only one passes, report a Holm-adjusted
card-specific result and treat the other as the measured boundary; do not
average a win and a loss into a universal claim.

### Frozen primary comparison

- New method: one exact runtime version, chosen before confirmation.
- Baseline: the strongest legal method selected on pilots from optimized
  C++/SIMD/OpenMP CPU, host CUDA Graph, route-bucketed host dispatch, and any
  relevant published runtime. Freeze this choice per stratum.
- Primary workload: one threshold-adjacent exact-route trace condition, where
  scheduling policy can plausibly matter.
- Replication workload: one live tool/coding-agent workload with externally
  scored task utility.
- Primary endpoints: P99 valid control-event latency, throughput, CPU
  core-seconds per event, and achieved accelerated share relative to `P*`.

The primary systems claim is co-primary: performance must improve, CPU use must
fall, task completion must be noninferior, and correctness must pass. Requiring
all components is an intersection-union claim and does not gain success by
cherry-picking one metric.

### Layout

- 30 fresh placements per GPU class, distributed over at least five calendar
  days and, where possible, two zones; no more than six analyzed placements of
  one class per day.
- Every placement executes all methods using a balanced randomized order.
- Use one globally sealed workload panel crossed with every placement and reset
  agent state for every block. Treat a small common seed panel as fixed; if the
  claim generalizes over workload seeds, use and power a materially larger seed
  sample rather than bootstrapping five clusters.
- Freeze exact scheduled attempt IDs, a hard period timeout, block cadence, and
  sample count from an effect-blind technical-precision qualification. Do not
  replenish failed, timed-out, or crash-lost attempts to reach a completed-only
  count, and do not extend a period based on the observed treatment effect.
- Persist raw invocation latency for empirical tails, but measure process,
  thread, context-switch, and cgroup CPU counters at calibrated block
  boundaries so instrumentation does not dominate microsecond invocations.
- Measure compile/cold-start separately; run both steady-state and amortized
  lifetime estimands. Do not hide compilation by excluding it from every
  headline.
- Run the full exactness suite on every placement before timing, and sample
  per-step trajectories during timing.

### Analysis

Compute paired placement log ratios, then use a two-way cluster bootstrap over
placement and trace seed. Report geometric ratio, 95% interval, next-placement
prediction interval, probability of improvement, and the full empirical
placement distribution. Use a max-statistic bootstrap for the two hardware
strata, or Holm-adjust the two card-specific primary contrasts.

The full population/horizon/route atlas remains exploratory. Present confidence
bands or performance profiles, not hundreds of isolated p-values. The
predeclared primary cell carries the inferential claim.

### Native-runtime stop rules

- **Validity stop:** any unexplained semantic mismatch, corrupted ordering, or
  impossible timestamp stops that runtime version. Diagnose, version, and
  preregister again; do not delete the failed launch.
- **Engineering futility after pilots only:** stop building a conditional/device
  graph compiler if no pilot stratum improves either P99 or CPU core-seconds by
  10% over the tuned host graph.
- **Confirmation:** no efficacy stop. Provisioning may stop after frozen `N` or
  after an alpha-spending boundary specified before the first confirmatory run.
- **Interpretation:** failure to show superiority plus a confidence interval
  entirely within `[1/1.05, 1.05]` supports practical equivalence; a wide
  interval is inconclusive, not evidence of no effect.

## Experiment 2: shared-vLLM slack harvesting

### Primary estimand

Do not make “no statistically significant slowdown” the claim. Define a fixed
useful control rate from pilots, then test whether shared execution is
noninferior to solo inference:

```text
D_TTFT = P99_TTFT(shared at frozen control rate) / P99_TTFT(solo)
D_TPOT = P99_TPOT(shared at frozen control rate) / P99_TPOT(solo)
```

The primary benefit is valid control work completed at that rate. The
secondary systems object is `lambda*`, the maximum control arrival rate whose
simultaneous upper confidence bounds keep both ratios below 1.02 and whose SLO
hit-rate loss stays above -0.5 pp.

Use one H100 model/serving configuration and one 85%-of-solo-saturation request
mix as the sole primary cell. L4, a second model, and 50/70/95% load are
replication/generalization strata. This prevents a large load/model grid from
becoming a multiplicity escape hatch.

### Layout

- Six H100 pilot placements estimate the paired P99 log-ratio variance.
- At least 20 and normally 30--80 confirmatory H100 placements, with final `N`
  chosen by the frozen 90%-power simulation above.
- Each placement runs solo and shared conditions in randomized AB/BA order
  using an identical generated request trace and decoding seeds.
- Start each period from the same model/KV/cache state or restart the server;
  freeze warm-up, cache flushing, clock policy, and cooldown.
- Each period contains at least 20,000 completed or timed-out requests and at
  least 200 observations in the empirical top 1%; increase duration when
  autocorrelation reduces the effective count.
- A treatment period that OOMs, crashes, or sheds requests remains a failed
  treatment period and an SLO violation.
- Freeze inference model, tensor parallelism, quantization, scheduler, batch
  limits, prompts/length distribution, prefill/decode mixture, and vLLM/CUDA
  commits.

### Decision rule and multiplicity

Success requires all of the following in the primary cell:

1. one-sided 97.5% upper bound for `D_TTFT < 1.02`;
2. one-sided 97.5% upper bound for `D_TPOT < 1.02`;
3. lower bound on SLO-hit difference above `-0.5 pp`;
4. lower bound on useful control work above the frozen benefit threshold;
5. inference error/timeout rate and task utility pass their guardrails.

Because the advertised claim requires every guardrail, this is an
intersection-union test. For the secondary `lambda*` curve, form simultaneous
max-t cluster-bootstrap bands across tested control rates. Holm-adjust formal
claims across secondary GPU/model strata; otherwise label the grid exploratory.

### Shared-vLLM stop rules

- During engineering pilots, immediately stop a rate after OOM or a sustained
  greater-than-10% P99 regression; record it as outside the feasible frontier.
- Choose the confirmatory rate using pilot placements only. Never lower it
  after viewing confirmatory SLO results.
- Do not stop confirmation when the first noninferiority interval passes.
- If the 2% margin is unattainable because paired SD is high, run the powered
  placement count. A post hoc 5% margin is a new experiment, not a reinterpretation.

## Experiment 3: diversity--regularity frontier

### Avoiding model-subset selection bias

Enumerating subsets on the existing correctness panel is method development,
not confirmation. Divide evidence into three immutable layers:

1. **Development:** existing 67-model panel; invent and tune selectors.
2. **Validation:** new tasks used to choose model subset sizes, constraints,
   and the exact joint objective.
3. **Sealed confirmation:** new task instances and preferably time-separated
   benchmark families, evaluated once after code/config freeze.

No task, paraphrase, template sibling, repository, or conversation may cross
these splits. Group splits by generator/template/repository, not by individual
row. Deduplicate semantically and by hashes. If model training-contamination
risk is material, include newly generated execution-verifiable tasks with a
public creation timestamp and report legacy benchmarks separately.

### Two-axis experimental unit

- **Epistemic/quality axis:** distinct task/problem, clustered by benchmark,
  template, and repository. All methods receive the same tasks.
- **Systems axis:** fresh serving placement. All methods run balanced blocks on
  the same placement and request trace.
- **Stochastic decoding:** generation seed nested within task/model; it does
  not turn one problem into many independent problems.

Use at least four substantively different domains. The aggregate should be
task-weighted according to a frozen target mixture; also report equal-domain
and per-domain sensitivity. An aggregate claim does not imply every-domain
superiority.

### Primary Pareto-movement claim

Freeze five methods before confirmation: best single, homogeneous/self-sample,
quality-only minimum-beta, systems-only affinity, and the proposed joint method
(five methods total). Additional heuristics are exploratory.

The joint method establishes a Pareto movement only if both comparisons pass:

1. versus quality-only, co-failure/task utility is noninferior and P95 task
   completion improves by at least 10%;
2. versus systems-only, co-failure improves by the frozen minimum and P95 task
   completion is noninferior within 5%.

All four component tests are required. This is more convincing than reporting
a post hoc scalarized score or hypervolume whose weights were chosen after the
results.

### Sample size and layout

- Start from a minimum 20,000 sealed tasks, at least 4,000 per domain.
- From validation data, simulate paired discordant outcomes and raise the
  count until every primary quality contrast has at least 90% power after the
  planned procedure. Target at least 200 reference all-wrong events; use up to
  200,000 tasks when beta is near 0.1%.
- For stochastic strategies, use a frozen small number of draws per task
  (recommended four), share seed schedules where meaningful, and cluster all
  draws by task.
- Use 20 fresh serving placements in each primary hardware stratum, every
  method run in balanced randomized order, with common task batches.
- A minimum of 10,000 requests per method/placement is required for P99;
  quality sample size is set by tasks, not requests.

### Analysis

Use paired task bootstrap/randomization for accuracy and beta differences,
report risk differences and risk ratios, and use exact or score intervals when
events are rare. A model-family or task-level cluster bootstrap must preserve
all outputs for a task together. Use a two-way task-by-placement bootstrap for
end-to-end latency/quality claims. Report:

- beta and its interval, not only average accuracy;
- paired discordance table for every primary comparison;
- accuracy, execution-graded utility, calibration if used by the selector, and
  abstention/failure rates;
- P50/P95/P99 task-completion time, GPU-seconds, dollars, model swaps, cache
  reuse, batch size, and queue time;
- a stratified performance profile across tasks/domains in addition to a mean.

### Diversity--regularity stop rules

- All algorithm/subset search stops before the sealed panel is unblinded.
- Validation-stage futility: do not build the new serving layer if the selected
  minimum-beta ensembles show less than 5% residency/cache/latency penalty
  versus systems-affinity ensembles at matched quality.
- Confirmation uses a fixed task count; no stopping when 200 favorable events
  have accumulated. Event-rate sample-size adjustment uses validation data or
  blinded pooled prevalence only.
- If beta is zero, report an exact upper bound; do not claim literal immunity
  to co-failure.

## Multiplicity map

Keep inferential families small and explicit:

| Family | Members | Rule |
|---|---|---|
| Native primary | two hardware-specific runtime-vs-baseline contrasts | broad claim requires both; card-specific claims use Holm or max-t simultaneous intervals |
| Native guardrails | TCT and task utility | all must pass noninferiority; intersection-union |
| Shared primary | TTFT, TPOT, SLO attainment, useful control work | all must pass; intersection-union |
| Shared secondary | extra GPU/model/load strata | Holm across named contrasts or exploratory only |
| Diversity Pareto claim | four directional/NI components | all must pass; intersection-union |
| Per-domain quality claims | four or more domains | Holm or max-t task-cluster bootstrap |

Do not apply false-discovery-rate control to rescue headline systems claims.
The full atlas may contain hundreds of cells, but it should be summarized by
effect surfaces and simultaneous bands, not asterisks. Freeze one or two cells
that carry each confirmatory statement. Use hierarchical gatekeeping: only if
the global/primary family passes may secondary cells receive confirmatory
language.

## Preregistration template

Copy and complete the following before any confirmatory outcome is inspected:

```markdown
# Experiment [ID]: [claim]

## Version and freeze
- Timestamp and immutable preregistration hash:
- Git commit and container-image digest:
- Analysis commit or sealed branch:
- Pilot datasets allowed for design:
- Confirmation data access control:

## Population and units
- Target provider/region/hardware/workload population:
- Treatment-assignment unit:
- Independent replication unit:
- Nested observation units:
- Pairing/blocking variables:
- Explicit non-generalization boundary:

## Methods and baselines
- Frozen treatment implementation:
- Frozen strongest legal baseline and how it was selected:
- Warm-up, reset, cache, compilation, and cooldown policy:
- Randomized block/order generator and seed list:

## Primary estimands
- Exact formula, aggregation order, units, and direction:
- Tail-quantile definition and timeout treatment:
- Target task/domain weighting:
- Deployment and task population over which expectation is taken:

## Hypotheses and practical margins
- Superiority null and SESOI:
- Noninferiority/equivalence null and margin justification:
- Correctness and SLO gates:
- Composite success rule:

## Multiplicity
- Primary family membership:
- Holm/max-t/gatekeeping procedure:
- Secondary/exploratory boundary:

## Sample size and power
- Pilot nuisance estimates and provenance:
- Simulation code/hash and 10,000-study type-I/power results:
- Frozen independent-unit count and provisioning reserve:
- Minimum requests/events/tasks per independent unit:

## Exclusions and failures
- Pre-start provider failure rule:
- Instrumentation-invalid block rule, treatment blind:
- OOM/crash/timeout encoding:
- Retry and replacement rule:
- Intention-to-run launch ledger:

## Stopping
- Engineering safety/validity stops:
- Futility rule, if any:
- Efficacy alpha-spending rule, if any; otherwise no peeking:
- Maximum units and calendar cutoff:

## Analysis
- Primary randomization/cluster interval:
- Hierarchical sensitivity model:
- Prediction interval and heterogeneity statistics:
- Robustness analyses fixed in advance:

## Artifact
- Raw schema, manifests, hashes, prices, and telemetry:
- Reproduction command and bounded-cost subset:
- Negative/failure result publication commitment:
```

## Reporting checklist

Every paper table/figure should make the following recoverable:

- number of placements, days, regions, GPU UUIDs, trace seeds, tasks, requests,
  and timing iterations separately;
- randomized order and whether contrasts were paired;
- point effect, 95% interval, practical margin, adjusted decision, and
  next-placement prediction interval;
- every crash, timeout, invalid block, retry, and semantic mismatch;
- P50/P95/P99 and empirical distributions, not only averages;
- compile/cold-start, steady-state, and amortized-lifetime results;
- per-domain and per-hardware effects alongside aggregates;
- primary, secondary, and exploratory labels in captions;
- complete model/config/provider-price timestamps and immutable source/image
  identifiers;
- negative regimes. A boundary paper is strengthened, not weakened, when a
  predeclared GPU, load, or route distribution loses.

## Methodological basis

The design follows the hierarchical-variation treatment in Kalibera and
Jones, [*Rigorous Benchmarking in Reasonable Time*](https://doi.org/10.1145/2464157.2464160),
and the systems-reporting discipline of Hoefler and Belli,
[*Scientific Benchmarking of Parallel Computing Systems*](https://doi.org/10.1145/2807591.2807644).
The task-suite recommendations use stratified bootstrap, performance profiles,
and robust aggregate reporting motivated by Agarwal et al.,
[*Deep Reinforcement Learning at the Edge of the Statistical Precipice*](https://arxiv.org/abs/2108.13264).
Power is planned prospectively as advocated by Card et al.,
[*With Little Power Comes Great Responsibility*](https://aclanthology.org/2020.emnlp-main.745/),
and variation sources follow Bouthillier et al.,
[*Accounting for Variance in Machine Learning Benchmarks*](https://arxiv.org/abs/2103.03098).
Practical no-harm claims use the TOST/noninferiority logic summarized by
Lakens, [*Equivalence Tests*](https://doi.org/10.1177/1948550617697177),
and small confirmatory families use Holm's
[*Sequentially Rejective Multiple Test Procedure*](https://www.jstor.org/stable/4615733).
