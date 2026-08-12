# Figure contracts and visual QA

Surface: two-column arXiv PDF. Audience: systems and ML-systems readers. The
figures must support the paper's quantitative claims at print size and remain
legible without color.

## Figure map

| Figure | Question | Form | Fields | Supported claim |
|---|---|---|---|---|
| Evidence map | How do the two studies feed the joined runtime test? | Directed schematic | `K`, `F`, `P*`, `U`, `A`, service outcomes | Workload supply and observation placement are separate measured gates |
| Opportunity frontier | How do `F`, `P*`, and `U` change with launch deadline? | Three-series ordered line plot | deadline, eligible share, estimator | Exact packing recovers fixed-window headroom at the primary deadline |
| Horizon ratios | Does the host/resident direction persist across cohort size and horizon? | Faceted ordered line plot | agents, epochs, placement, ratio | All 36 named placement-cells favor the resident path |
| Primary mechanisms | How large are the three mechanism times at the primary cell? | Connected categorical comparison on a log scale | mechanism, placement, wall time | Resident is below host on every placement; oracle floor remains diagnostic |

## Palette and non-color encoding

- Paper ink: `#262B33`; grid: `#D7DBE0`; background: white.
- Primary blue: `#1769AA`; reference gold: `#B97913`; supporting olive:
  `#687A3A`; supporting pink: `#A64D79`; neutral placement: `#5E6673`.
- The opportunity plot uses one blue root plus neutrals and a gold primary
  deadline. `F`, `P*`, and `U` also differ by square/circle/triangle markers,
  solid/dashed/dotted lines, and open versus filled markers.
- Placement plots use provider-specific colors, marker shapes, and line styles.
  No comparison relies on hue alone.
- Red/green semantics, gradients, colored backgrounds, and library default
  palettes are prohibited.

## Scale and grain checks

- Eligible share uses a zero-based percentage scale and an explicitly labeled
  log-scaled deadline axis.
- The primary mechanism plot uses a log scale because the diagnostic floor is
  roughly an order of magnitude below the legal paths; the axis states this.
- Every ratio is a ratio of within-placement medians of batch-average rows.
  Placement is the outer unit. The plots do not label these values as raw P99
  or population confidence intervals.
- The 50 ms line is a launch deadline, not a completion SLO.

## Final-context gate

After every regeneration, inspect the PNG and the compiled PDF. Reject a figure
for clipped labels, overlapping legends, unreadable footer text, default color
cycles, color-only distinctions, an unlabeled log scale, or a caption whose
metric grain differs from the plotted table.
