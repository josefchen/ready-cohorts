# Pilot figure contract

## Analytical question

How does synchronized wall-time speedup over the eight-thread CPU baseline vary
with agent population, execution mode, state width, action count, and hardware?

## Supported takeaway

The pilot may show a **framework-level crossover** and the penalty for exposing
each action to the host. It must not be titled or narrated as proof that GPU
cores beat an optimized CPU implementation because the CPU path is eager
PyTorch while one GPU path is captured.

## Form

- Family: ordered relationship / uncertainty and benchmark.
- Variant: marker-line small multiples with a horizontal `1×` reference.
- Grain: median of 7 or 10 fresh timing repetitions per benchmark cell.
- X: agent population, logarithmic scale.
- Y: speedup over the matched eight-thread CPU median, logarithmic scale.
- Facets: state width × action-state count; one figure per hardware environment.
- Series: CUDA Graph resident, eager resident, eager host-visible.
- Invalid correctness cells: omitted from the primary line and marked in the
  quality table; never silently interpolated.

## Visual policy

- CUDA Graph: blue solid line, filled circle.
- Eager resident: orange solid line, square.
- Host-visible: charcoal dashed line, open triangle.
- The `1×` reference is neutral grey.
- Markers and line styles preserve meaning in grayscale.
- Near-white background, quiet grey grid, no gradient or truncated linear axis.

## Output and QA

- Notebook: `notebooks/01_pilot_analysis.ipynb`.
- Figures: `results/figures/*-speedup.png` and matching SVG.
- Processed tables: `data/processed/pilot-cell-summary.csv` and
  `data/processed/pilot-crossovers.csv`.
- Inspect exported PNGs for clipped labels, legend collisions, and visible
  `1×` reference before sharing.

