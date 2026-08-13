# Ready Cohorts arXiv manuscript

Published paper: <https://arxiv.org/abs/2608.12123>

Public project: <https://github.com/josefchen/ready-cohorts>

Processed evidence: <https://huggingface.co/datasets/josefchen/ready-cohorts>

This directory contains the governed manuscript sources. Outcome numbers and
primary tables are generated from canonical processed evidence.

Verify the exact arXiv upload archive from the repository root:

```bash
.venv/bin/python scripts/verify_arxiv_release.py
```

Full numerical regeneration also requires the pinned public trace shards. Run
`scripts/fetch_trace_source.py` before `make -C paper/arxiv clean all`.

Do not hand-edit files under `generated/`. The upload bundle and its independent
verification records are under `../../release/arxiv/`; private provider receipts
are excluded from that bundle.
