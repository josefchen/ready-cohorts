# Chart contracts

This note fixes the analytical purpose, data grain, encoding, and provenance of
the three interactive views. It is implementation documentation, not an
additional claim surface.

## Ready-cohort opportunity

- Question: how does eligible share change across launch deadlines for one
  selected replay condition?
- Takeaway: cohort supply depends sharply on C, K, grouping, and the launch
  deadline. Fixed windows can miss schedulable events.
- Form: highlighted multi-series line over five ordered deadline values.
- Grain: one summary row per C, deadline, grouping, and K; each is a mean over
  three generated swarms.
- Encodings: F uses a neutral dashed square line; P* uses a solid cobalt circle
  line; U uses a dotted open-triangle cobalt line. Labels and line styles keep
  the series distinct without color.
- Scale: 0% to 100% on the eligible-share axis; log deadline axis with all five
  measured values labeled.
- Source: `data/trace-summary.csv`, copied byte-for-byte from
  `josefchen/ready-cohorts@ready-cohorts-arxiv-v1:trace/summary.csv`.

## Resident decision comparison

- Question: at one N and H, how does the host round-trip path compare with the
  device-resident path on each named placement?
- Takeaway: every released placement-cell favors the resident path
  descriptively, with magnitude varying by placement and horizon.
- Form: horizontal dot and reference-line comparison across four categories.
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
- Form: horizontal dot and reference-line comparison across five categories.
- Grain: placement is the outer unit; each value is a ratio of within-placement
  wall-time medians.
- Encodings: neutral stems begin at the equal-time reference of 1; open cobalt
  diamonds and exact direct labels mark ratios.
- Scale: focused ratio axis anchored by an explicit equality line.
- Source: `data/native-contrasts.csv`, copied byte-for-byte from
  `josefchen/ready-cohorts@ready-cohorts-arxiv-v1:native/contrasts.csv`.

## Palette and QA

The interface uses one cobalt root plus cool neutrals. Charts use line style,
marker shape, position, and direct labels in addition to color. All charts have
plain-text readouts. The final Space must be inspected at desktop and mobile
widths in light and dark system modes before release.
