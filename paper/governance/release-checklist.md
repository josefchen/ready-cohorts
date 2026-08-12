# Release checklist

Status: arXiv upload candidate. Scientific and source-package gates pass. The
remaining arXiv steps require the author's account: choose a license, inspect
the generated preview, and submit.

## Scientific scope gates

- [x] Every abstract number maps to the claim-evidence file.
- [x] Complete appendix proofs received an adversarial formal review.
- [x] Algorithm-priority search was repeated; the manuscript makes no
      algorithm-priority claim.
- [x] Closest-work citations were checked against primary records and revised
      to avoid novelty infringement.
- [x] The manuscript distinguishes named-placement description from population
      inference.
- [x] The online-runtime claim is explicitly absent from the title, abstract,
      contributions, and conclusion.
- [x] Public author metadata is exactly Josef Chen, Independent Researcher.
      Submitter contact is private and absent from the PDF, source archive,
      repository index, and Hugging Face mirror.
- [x] The three data figures are hash-bound to tables and scripts and were
      visually reviewed in color and grayscale. They use redundant markers and
      line styles, restrained color, honest axes, and explicit data grain.
- [x] The final scientific edit set preserves the prior independent factual,
      quantitative, novelty, and voice findings. Later changes are limited to
      public metadata, repository links, path redaction, and TeX compatibility.

## arXiv source-package gates

- [x] The 23-file source archive is deterministic and contains no auxiliary,
      private, unused, or machine-local files.
- [x] The official arXiv `submission-tools` preflight at commit
      `c494260357a28295d2e95c4cecdfd2b77b0b2e9e` reports one top-level TeX file,
      19 TeX files, three images, a pre-generated bibliography, and zero issues.
- [x] Three-pass isolated compilation succeeds under local TeX Live 2023 with
      network access and shell escape disabled.
- [x] Three-pass isolated compilation succeeds under public TeX Live 2025 image
      digest `sha256:5912d6a33798957f4cd6ff1673819209f59300c378033d25066feb93270b90ce`.
      It produces 14 pages with text exactly matching the canonical build.
- [x] PDF metadata names Josef Chen, all 31 fonts are embedded, JavaScript is
      absent, and the rendered PDF contains no email address.
- [x] The submission metadata is ASCII and the abstract is below arXiv's
      1,920-character limit.
- [x] `SHA256SUMS`, the release manifest, compiler records, and verification
      record bind every handoff file.
- [x] The complete staged public tree contains zero exact matches to configured
      secrets, zero high-confidence credential matches, and zero private-email
      matches.
- [ ] The exact release commit reproduces from a clean checkout.
- [ ] The public GitHub branch and annotated release tag point to that commit.

## Public artifact gates

- [x] GitHub owner and Hugging Face owner were authenticated as `josefchen`.
- [x] The stable project URLs are
      <https://github.com/josefchen/ready-cohorts> and
      <https://huggingface.co/datasets/josefchen/ready-cohorts>.
- [x] The processed-evidence mirror is public at Hugging Face commit
      `41ff8d06578d4ec68934040a8fb925d3c8b01767`; 21 repository files are present.
- [x] Downloaded mirror copies of the card, manifest, and PDF match local
      SHA-256 values exactly.
- [x] Raw provider receipts and native binaries are excluded. Published
      manifests contain repository-relative paths and bind public copies to
      their source SHA-256 values.
- [x] All 19 source shards match their local, manifest, and commit-resolved
      remote SHA-256 values at Exgentic conversion revision
      `f7c94012d0bfbf66fe4d6ed627699508bbb555ff`.
- [ ] Add the arXiv identifier to GitHub, Hugging Face, and `CITATION.cff` after
      announcement.

## Author account actions

- [ ] Choose the arXiv distribution license. This choice is irrevocable and is
      intentionally not made by the release scripts.
- [ ] Upload `release/arxiv/ready-cohorts-arxiv-v1.tar.gz`, paste the frozen
      metadata, inspect every page of arXiv's generated preview, and submit.

## Follow-up research, not arXiv blockers

- [ ] Build the finite-capacity online route service and measure achieved share
      A, raw P99 control latency, CPU core-seconds, cost, and exact effects.
- [ ] Run the powered placement design before making hardware-population or
      provider-population claims.
- [ ] Add the shared-vLLM interference arm before claiming datacenter service
      benefit.
- [ ] Obtain fresh external review for a later peer-reviewed systems submission.
