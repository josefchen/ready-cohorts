# Research and paper agent contract

This repository contains a measurement artifact and an arXiv manuscript about
GPU execution of deterministic LLM-agent control. Read `paper/brief.yaml`,
`paper/STATE.yaml`, `GOVERNANCE.md`, and the relevant preregistration before
changing experiments, claims, or manuscript text.

## Non-negotiable rules

1. Never invent a citation, result, locator, configuration, provider event, or
   statistical conclusion.
2. Classify load-bearing statements as `observed`, `computed`, `proved`,
   `inferred`, or `proposed`. A citation supports its exact premise, not a
   stronger inference.
3. Treat a fresh GPU placement as the performance sampling unit. Timing rows
   and batched invocations are technical repetitions.
4. Do not describe batch-average quantiles as invocation-tail latency.
5. Do not claim population significance, CPU displacement, end-to-end agent
   speedup, energy savings, or shared-LLM safety without the corresponding
   experiment.
6. Raw inputs and provider receipts are immutable. Private RunPod receipts stay
   local because they contain ephemeral artifact tokens.
7. Only the primary orchestrator edits canonical ledgers, paper state, or
   manuscript prose. Parallel agents perform bounded read-only audits or return
   immutable review packets. A scout cannot approve its own evidence.
8. A new performance-bearing experiment requires a frozen source hash,
   preregistration, fixed scheduled outcomes, intention-to-run failure handling,
   and an explicit spend and lifecycle bound.
9. Preserve negative results and boundary regimes. Do not move the primary cell
   or redefine a metric after seeing an outcome.
10. Run the anti-slop audit on title, abstract, introduction, discussion, and
    conclusion. Prefer plain technical prose, concrete subjects, and exact
    limitations. Promotional language is a release blocker.

## Paper gates

Move through:

`contract -> coverage -> sources -> analysis -> claims -> draft -> adversarial_review -> release`

The current gate is recorded in `paper/STATE.yaml`. Do not call the manuscript
release-ready while any critical review is open, the clean build fails, a used
claim lacks evidence, or the repository contains a publishable secret.

## Required checks

```bash
.venv/bin/pytest -q
.venv/bin/ruff check scripts tests src
make -C paper/arxiv clean all
.venv/bin/python scripts/check_arxiv_paper.py
git diff --check
```

The paper build is a working-draft check. Final release also requires a clean
checkout reproduction, public-safe provider provenance, an immutable commit and
tag, and fresh factual, quantitative, novelty, voice, and release reviews.
