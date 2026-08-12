# Raw benchmark ledger

Each benchmark invocation writes one append-only CSV and one JSON manifest.
Rows with errors are evidence and must not be deleted. No credentials, account
identifiers, or environment-variable values belong in this directory.

Pilot families:

- 001/002: eager and CUDA-Graph crossover smoke/pilot;
- 003/004: compiler-matched GTX 1660 Ti and Modal L4 atlas;
- 005/006: preregistered sub-256 refinement;
- 007–011: preregistered Modal cross-generation hardware sweep;
- 012–029: preregistered Modal cross-generation confirmation, three fresh
  placements for each of T4, L4, A10, L40S, A100-80GB, and exact H100.

Native mechanism family:

- `native-dispatch-001`: field-exact fixed-graph calibration on one local GTX
  1660 Ti, two fresh Modal L4 placements, one RunPod L4 placement, and one
  Lambda H100 placement. Each placement contains 2,400 measured rows (50
  randomized-order repetitions for every mechanism/shape cell). Provider
  receipts and nested RunPod collection artifacts remain under `data/external/`,
  `data/provider-runpod-launches/`, and the corresponding RunPod raw directory.
  The timing rows are technical repetitions; the five fresh GPU placements are
  the independent units for portability and performance variation.
- `resident-policy-001`: a new decision-bearing CUDA mechanism. The archived
  `resident-policy-dev-smoke` is explicitly disclosed pre-freeze development
  evidence. Completed `resident-policy-001-*` ledgers use the frozen source in
  `preregistration/resident-policy-001.md`; every invocation is checked against
  an independent host-only oracle for both final state fields and the complete
  decision trace. Measured rows contain minimum-duration batches and remain
  technical repetitions; fresh GPU placements are the sampling units. The
  completed first stage contains four placements (local GTX 1660 Ti, Modal L4,
  RunPod L4, and Lambda H100), 3,240 measured rows, and provider-bound cloud
  receipts. RunPod's validated archive is nested below its collection
  directory; the analysis discovers it recursively.

The CSV is the timing ledger. The sibling manifest is required evidence: a CSV
without its manifest must not enter a processed analysis.
