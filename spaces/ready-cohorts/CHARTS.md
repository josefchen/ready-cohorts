# Chart contracts

This note fixes the analytical purpose, data grain, encoding, and provenance of
the three interactive views. It is implementation documentation, not an
additional claim surface.

## Ready-cohort opportunity

- Question: how does eligible share change across launch deadlines for one
  selected replay condition?
- Takeaway: cohort supply depends sharply on C, K, grouping, and the launch
  deadline. Fixed windows can miss schedulable events.
- Form: highlighted multi-series line over five ordered deadline values. The
  selected deadline receives a larger marker and a direct P* minus F annotation
  when the gap is nonzero.
- Grain: one summary row per C, deadline, grouping, and K; each is a mean over
  three generated swarms.
- Encodings: F uses a neutral dashed square line; P* uses a solid cobalt circle
  line; U uses a dotted open-triangle cobalt line. P* error bars show the
  descriptive minimum-to-maximum range across the three generated swarms, not
  a confidence interval. Labels, line styles, marker shapes, and direct readouts
  keep the series distinct without color.
- Scale: 0% to 100% on the eligible-share axis; log deadline axis with all five
  measured values labeled.
- Source: `data/trace-summary.csv`, copied byte-for-byte from
  `josefchen/ready-cohorts@ready-cohorts-arxiv-v1:trace/summary.csv`.

## Resident decision comparison

- Question: at one N and H, how does the host round-trip path compare with the
  device-resident path on each named placement?
- Takeaway: every released placement-cell favors the resident path
  descriptively, with magnitude varying by placement and horizon.
- Form: horizontal lollipop and equality-line comparison across four
  categories, with the resident-faster side lightly shaded.
- Grain: placement is the outer unit; each value is a ratio of within-placement
  medians of batch-average technical rows.
- Encodings: neutral stems begin at the equal-time reference of 1; cobalt dots
  and exact direct labels mark ratios.
- Scale: focused ratio axis anchored by an explicit equality line. A zero
  baseline is not meaningful for this comparison.
- Source: `data/resident-contrasts.csv`, copied byte-for-byte from
  `josefchen/ready-cohorts@ready-cohorts-arxiv-v1:resident/contrasts.csv`.

## Fixed nested launch negative control

- Question: does device-side nested graph launch help when it removes no host
  decision?
- Takeaway: the fixed nested path is slower in every released placement-cell.
- Form: horizontal open-diamond lollipop and equality-line comparison across
  five categories, with the slower-than-host side lightly shaded.
- Grain: placement is the outer unit; each value is a ratio of within-placement
  wall-time medians.
- Encodings: neutral stems begin at the equal-time reference of 1; open cobalt
  diamonds and exact direct labels mark ratios.
- Scale: focused ratio axis anchored by an explicit equality line.
- Source: `data/native-contrasts.csv`, copied byte-for-byte from
  `josefchen/ready-cohorts@ready-cohorts-arxiv-v1:native/contrasts.csv`.

## Palette and QA

The interface uses cobalt for computed opportunity, ochre for observation
placement, and cool neutrals for structure. Charts use line style, marker shape,
position, and direct labels in addition to color. The two hues remain distinct
under grayscale through their separate shapes and positions. All charts have
plain-text readouts. The final Space must be inspected at desktop and mobile
widths in light and dark system modes before release.

## Architecture and launch visual

- `assets/ready-cohorts-architecture.svg` is a systems map, not another result
  figure. It keeps the computed workload gate, observed placement gate, and
  proposed joined runtime in distinct bordered regions.
- `assets/ready-cohorts-social-card.svg` and its deterministic 1600 by 900 PNG
  reuse only released headline values and state the evidence boundary inside
  the image.
- `scripts/build_public_visuals.py` reads the bundled CSVs, validates direction
  and counts, renders both source SVGs, validates their XML, and creates the PNG
  with headless Chrome. The assets are SHA-256-bound in `data/manifest.json`.
