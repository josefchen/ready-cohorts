# Trace replay 002 preregistration: sliding-deadline local upper bound

Status: frozen before computation on 2026-08-11.

## Motivation

Trace replay 001 measures the exact eligible share under an origin-aligned,
non-overlapping microbatch partition. That value is not a universal scheduler
ceiling: events on opposite sides of a frozen boundary may still share a legal
sliding deadline. This follow-up computes a valid per-event local-eligibility
upper bound under equal sliding deadlines, while retaining the same source
panel and stationary replay mechanism.

## Definition

Event `i` is released at `t_i` and must launch by `t_i + delta`. For its exact
group, let `A(tau)` be the number of event intervals containing `tau`. Its
local cohort size is

```text
u_i(delta) = max_{tau in [t_i, t_i + delta]} A(tau).
```

The local upper share at threshold `K` is the fraction with `u_i >= K`. Every
accelerated event in any legal schedule must satisfy this condition, but all
locally eligible events need not be jointly schedulable. It is therefore a
valid upper bound, not an achievability claim.

## Frozen design

- same pinned tau2 feature/session files and stationary Poisson template replay
  as trace replay 001;
- target active sessions: 1,000, 10,000, and 100,000;
- equal per-event deadlines: 10, 25, 50, 100, and 250 ms;
- grouping: pooled, event class, and exact route;
- crossover thresholds: 32, 64, 128, and 256;
- three seeded replay repetitions;
- 60-second retained interval;
- root seed: 20260811, matching replay 001 so overlapping cells can be checked
  for exact deterministic reproduction.

The corresponding fixed-window eligible share is recomputed from the same
events. No event or timing observation is removed.

## Hypotheses and invariants

- **B1:** fixed-window eligibility is no greater than the local upper share in
  every matched cell;
- **B2:** exact-route upper share is no greater than event-class upper share,
  which is no greater than pooled upper share in every matched cell;
- **B3:** at least one matched cell has a strictly positive boundary-alignment
  gap (`local upper - fixed window`);
- **B4:** at `C=100,000`, `K=256`, and 50 or 100 ms, the exact-route local upper
  share is strictly above fixed-window eligibility.

B1 and B2 are implementation-validity invariants. A failure invalidates the
analysis rather than becoming a negative result. B3/B4 are descriptive pilot
hypotheses.

## Limits

The local upper share can overestimate a jointly achievable schedule because
overlapping opportunities may require reusing the same events. The exact
offline packing optimum remains a separate implementation target. The source
arrival, route, and timestamp qualifications from replay 001 still apply.
