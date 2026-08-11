# GPU Agent Crossover

Research artifacts for **When Is a GPU a Cheap CPU? The Residency–Regularity
Frontier for AI Agent Swarms**.

The project measures when non-neural agent control work benefits from GPU
execution. It deliberately separates four effects that are often conflated:

1. population-level parallelism;
2. arithmetic work per agent;
3. whether agent state remains device-resident;
4. whether the repeated control graph can be captured and replayed.

The first pilot compares one- and eight-thread CPU execution against eager,
CUDA-Graph, and host-visible GPU execution. A low-end GTX 1660 Ti is the first
hardware anchor; ephemeral datacenter GPUs are added only after the harness
passes correctness and stability checks.

## Reproduce the local pilot

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[dev,cloud]'
.venv/bin/python scripts/run_pilot.py --config configs/pilot-001.toml
```

Raw observations are append-only CSV files under `data/raw/`. Each run also
writes a JSON manifest containing the exact hardware, software, command, seed,
and configuration. The analysis notebook reads those files rather than copied
summary values.

## Research rule

A GPU win is not assumed. If no GPU mode beats the eight-core CPU within the
tested envelope, that is the pilot result and the next study maps the negative
boundary. Accuracy checks, synchronization, initialization exclusions, and
failed cells are retained rather than silently filtered.

