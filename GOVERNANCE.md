# Governance for the Ready-Cohort project

Agents are fallible research workers. Reliability comes from bounded authority,
separation of duties, typed evidence, and deterministic release checks.

## Research contract

`paper/brief.yaml` fixes the question, audience, evidence cutoff, exclusions,
deliverables, and stopping conditions. A thesis change that invalidates that
contract reopens the contract gate and receives a recorded decision.

## Authority and orchestration

- The primary orchestrator owns canonical synthesis, manuscript prose, the
  claim map, and `paper/STATE.yaml`.
- Source scouts locate primary evidence but cannot verify their own claims.
- Quantitative reviewers recompute results but cannot silently rewrite the
  estimand or sampling unit.
- Novelty reviewers search for collisions and report the losing argument.
- Voice reviewers may change cadence and wording but not numbers, confidence,
  causal force, or citation scope.
- Release reviewers may block publication but cannot add last-minute evidence.

Every delegated task has one bounded question and no shared write scope. The
orchestrator records the task path, evidence cutoff, output, and disposition in
`paper/governance/agent-task-ledger.md`.

## Evidence and claims

The evidence hierarchy is:

1. mathematical proof or protocol-level specification for formal claims;
2. immutable raw artifacts and independently reproducible calculations;
3. peer-reviewed papers and official technical documentation;
4. official datasets with pinned revisions and licenses;
5. arXiv manuscripts with inspectable methods;
6. search results only for discovery.

Each headline number must point to a file hash, generating script, and analysis
unit. Contradictory evidence is retained. An unresolved load-bearing conflict
forces the claim to be hedged, demoted, or cut.

The paper uses the following epistemic labels:

- `proved`: follows from the stated mathematical model;
- `observed`: directly recorded on named artifacts or placements;
- `computed`: deterministically derived from pinned inputs;
- `inferred`: a statistical population statement supported by a valid design;
- `proposed`: a hypothesis, architecture, or future experiment.

Only `proved`, `observed`, and `computed` positive claims are currently allowed
in the abstract. The manuscript has no positive provider-population or
workload-population inference yet.

## Experiment governance

The next online-runtime experiment must freeze:

- route semantics and the strongest legal baselines;
- route-, hardware-, visibility-, and horizon-specific crossover measurement;
- fixed scheduled attempt IDs and hard timeouts;
- crashes, OOMs, deadline misses, and missing chunks as intention-to-run
  outcomes;
- a single durable lifecycle ledger and provider cleanup path;
- placement, workload seed, day, provider, and hardware as explicit factors;
- correctness, P99 latency, CPU core-seconds, online accelerated share, task
  utility, and LLM interference guardrails.

No favorable microbenchmark authorizes a larger performance campaign. The
bounded instrumentation qualification precedes nuisance estimation, and pilot
placements remain outside confirmation.

## Self-improvement loop

Every substantive paper pass records four fields in
`paper/governance/self-improvement-log.md`:

1. the defect or reviewer attack;
2. the evidence that exposed it;
3. the change made to the manuscript, analysis, or protocol;
4. a mechanical guard that prevents recurrence.

Examples include replacing row-level significance with placement-level units,
renaming batch quantiles, rejecting a fixed nested graph, and separating launch
deadlines from completion SLOs. Self-improvement means preserving and testing
these corrections, not asserting that an agent improved itself.

## Release blockers

Release is blocked by any of the following:

- a used claim without a source, proof, run, or exact locator;
- a paper number that fails independent recomputation;
- a figure without a source table, generating script, and hash;
- a citation that does not entail the sentence it supports;
- unresolved critical novelty, factual, quantitative, or voice review;
- language that promotes a preliminary result as a population claim;
- missing AI-use, artifact, ethics, or limitations disclosure;
- a failed clean build or clean-checkout reproduction;
- a private token, credential, prompt payload, tool result, or unredacted
  provider receipt in the release bundle;
- a stale review after editing a reviewed artifact.

Advisory scores never override these blockers.
