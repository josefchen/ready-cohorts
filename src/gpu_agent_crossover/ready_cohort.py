from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ReplayResult:
    event_times_s: np.ndarray
    event_class_ids: np.ndarray
    route_ids: np.ndarray
    mean_active_sessions: float
    arrival_count: int


@dataclass(frozen=True)
class ExactPackingResult:
    """Exact equal-deadline solution to the sliding-deadline batching problem."""

    accelerated_event_count: int
    accelerated_share: float
    batch_count: int
    event_launch_times_s: np.ndarray
    algorithm: str


_NANOSECONDS_PER_SECOND = 1_000_000_000
_NANOSECONDS_PER_MILLISECOND = 1_000_000


def _event_time_ticks(event_times_s: np.ndarray) -> np.ndarray:
    """Normalize floating timestamps onto the experiment's integer-ns clock."""

    scaled = event_times_s * _NANOSECONDS_PER_SECOND
    if not np.isfinite(scaled).all() or scaled.max() > np.iinfo(np.int64).max:
        raise ValueError("event times exceed the signed 64-bit nanosecond clock")
    return np.rint(scaled).astype(np.int64)


def _duration_ticks(duration_ms: float) -> int:
    scaled = duration_ms * _NANOSECONDS_PER_MILLISECOND
    if not np.isfinite(scaled) or scaled > np.iinfo(np.int64).max:
        raise ValueError("duration exceeds the signed 64-bit nanosecond clock")
    ticks = round(scaled)
    if ticks <= 0:
        raise ValueError("duration rounds to zero on the nanosecond clock")
    return ticks


def _positive_session_table(session_frame: pd.DataFrame) -> pd.DataFrame:
    required = {"session_id", "session_duration_s"}
    missing = required.difference(session_frame.columns)
    if missing:
        raise ValueError(f"session frame is missing columns: {sorted(missing)}")
    sessions = session_frame.loc[:, ["session_id", "session_duration_s"]].copy()
    sessions["session_duration_s"] = pd.to_numeric(
        sessions["session_duration_s"], errors="raise"
    )
    if sessions["session_id"].duplicated().any():
        raise ValueError("session_id must be unique in the session frame")
    if (sessions["session_duration_s"] <= 0).any():
        raise ValueError("session durations must be positive")
    return sessions


def simulate_stationary_swarm(
    span_frame: pd.DataFrame,
    session_frame: pd.DataFrame,
    *,
    target_active_sessions: int,
    horizon_s: float,
    rng: np.random.Generator,
) -> ReplayResult:
    """Replay empirical session templates under stationary Poisson arrivals.

    Templates are sampled uniformly. Splitting one Poisson arrival process
    across templates produces independent per-template Poisson processes, which
    lets the implementation stay vectorized even at high concurrency.
    """

    if target_active_sessions <= 0:
        raise ValueError("target_active_sessions must be positive")
    if horizon_s <= 0:
        raise ValueError("horizon_s must be positive")
    required = {"session_id", "ready_offset_s", "event_class", "route_key", "span_index"}
    missing = required.difference(span_frame.columns)
    if missing:
        raise ValueError(f"span frame is missing columns: {sorted(missing)}")

    sessions = _positive_session_table(session_frame)
    span_groups = {
        str(session_id): group.sort_values("span_index")
        for session_id, group in span_frame.groupby("session_id", sort=False)
    }
    if set(sessions["session_id"].astype(str)) != set(span_groups):
        raise ValueError("span and session frames contain different session IDs")

    event_class_labels = sorted(span_frame["event_class"].astype(str).unique())
    route_labels = sorted(span_frame["route_key"].astype(str).unique())
    event_class_lookup = {label: index for index, label in enumerate(event_class_labels)}
    route_lookup = {label: index for index, label in enumerate(route_labels)}

    mean_duration_s = float(sessions["session_duration_s"].mean())
    arrival_rate_hz = target_active_sessions / mean_duration_s
    template_rate_hz = arrival_rate_hz / len(sessions)
    event_times: list[np.ndarray] = []
    event_class_ids: list[np.ndarray] = []
    route_ids: list[np.ndarray] = []
    total_active_time_s = 0.0
    arrival_count = 0

    for row in sessions.itertuples(index=False):
        session_id = str(row.session_id)
        duration_s = float(row.session_duration_s)
        interval_s = horizon_s + duration_s
        count = int(rng.poisson(template_rate_hz * interval_s))
        if count == 0:
            continue
        arrivals = rng.uniform(-duration_s, horizon_s, size=count)
        arrival_count += count
        active_start = np.maximum(arrivals, 0.0)
        active_end = np.minimum(arrivals + duration_s, horizon_s)
        total_active_time_s += float(np.maximum(active_end - active_start, 0.0).sum())

        template = span_groups[session_id]
        ready_offsets = template["ready_offset_s"].to_numpy(dtype=np.float64)
        candidate_times = arrivals[:, None] + ready_offsets[None, :]
        keep = (candidate_times >= 0.0) & (candidate_times < horizon_s)
        if not keep.any():
            continue
        event_times.append(candidate_times[keep])

        template_event_ids = np.array(
            [event_class_lookup[str(value)] for value in template["event_class"]],
            dtype=np.int32,
        )
        template_route_ids = np.array(
            [route_lookup[str(value)] for value in template["route_key"]],
            dtype=np.int32,
        )
        repeated_event_ids = np.broadcast_to(template_event_ids, candidate_times.shape)
        repeated_route_ids = np.broadcast_to(template_route_ids, candidate_times.shape)
        event_class_ids.append(repeated_event_ids[keep])
        route_ids.append(repeated_route_ids[keep])

    if not event_times:
        raise RuntimeError("stationary replay retained no events")
    return ReplayResult(
        event_times_s=np.concatenate(event_times),
        event_class_ids=np.concatenate(event_class_ids),
        route_ids=np.concatenate(route_ids),
        mean_active_sessions=total_active_time_s / horizon_s,
        arrival_count=arrival_count,
    )


def cohort_metrics(
    event_times_s: np.ndarray,
    group_ids: np.ndarray,
    *,
    window_ms: float,
    thresholds: Iterable[int],
) -> dict[str, float | int]:
    """Compute event-weighted fixed-window cohort statistics."""

    times = np.asarray(event_times_s, dtype=np.float64)
    groups = np.asarray(group_ids, dtype=np.int64)
    if times.ndim != 1 or groups.ndim != 1 or len(times) != len(groups):
        raise ValueError("event times and group IDs must be aligned one-dimensional arrays")
    if len(times) == 0:
        raise ValueError("at least one event is required")
    if window_ms <= 0:
        raise ValueError("window_ms must be positive")
    if not np.isfinite(times).all() or (times < 0).any() or (groups < 0).any():
        raise ValueError("event times and group IDs must be finite and nonnegative")

    time_ticks = _event_time_ticks(times)
    window_ticks = _duration_ticks(window_ms)
    window_ids = time_ticks // window_ticks
    group_stride = int(groups.max()) + 1
    cohort_keys = window_ids * group_stride + groups
    _, inverse, cohort_counts = np.unique(
        cohort_keys,
        return_inverse=True,
        return_counts=True,
    )
    event_cohort_sizes = cohort_counts[inverse]
    wait_ms = (
        ((window_ids + 1) * window_ticks - time_ticks)
        / _NANOSECONDS_PER_MILLISECOND
    )
    output: dict[str, float | int] = {
        "event_count": len(times),
        "cohort_count": len(cohort_counts),
        "cohort_size_event_mean": float(event_cohort_sizes.mean()),
        "cohort_size_event_p50": float(np.quantile(event_cohort_sizes, 0.50)),
        "cohort_size_event_p90": float(np.quantile(event_cohort_sizes, 0.90)),
        "cohort_size_event_p99": float(np.quantile(event_cohort_sizes, 0.99)),
        "cohort_size_max": int(cohort_counts.max()),
        "wait_ms_mean": float(wait_ms.mean()),
        "wait_ms_p95": float(np.quantile(wait_ms, 0.95)),
    }
    for threshold in thresholds:
        if threshold <= 0:
            raise ValueError("thresholds must be positive")
        output[f"eligible_share_k{threshold}"] = float(
            np.mean(event_cohort_sizes >= threshold)
        )
    return output


def _range_maximum_queries(
    values: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    """Return inclusive range maxima with a vectorized segment tree."""

    values = np.asarray(values)
    left = np.asarray(left, dtype=np.int64)
    right = np.asarray(right, dtype=np.int64)
    if values.ndim != 1 or left.shape != values.shape or right.shape != values.shape:
        raise ValueError("values, left, and right must be aligned one-dimensional arrays")
    if len(values) == 0:
        return np.empty(0, dtype=values.dtype)
    if (left < 0).any() or (right < left).any() or (right >= len(values)).any():
        raise ValueError("range bounds are invalid")

    leaf_count = 1 << (len(values) - 1).bit_length()
    tree = np.zeros(2 * leaf_count, dtype=values.dtype)
    tree[leaf_count : leaf_count + len(values)] = values
    width = leaf_count // 2
    while width:
        children = tree[2 * width : 4 * width]
        tree[width : 2 * width] = np.maximum(children[0::2], children[1::2])
        width //= 2

    query_left = left + leaf_count
    query_right = right + leaf_count + 1
    result = np.zeros(len(values), dtype=values.dtype)
    while np.any(query_left < query_right):
        take_left = (query_left & 1).astype(bool) & (query_left < query_right)
        if take_left.any():
            result[take_left] = np.maximum(
                result[take_left], tree[query_left[take_left]]
            )
            query_left[take_left] += 1

        take_right = (query_right & 1).astype(bool) & (query_left < query_right)
        if take_right.any():
            query_right[take_right] -= 1
            result[take_right] = np.maximum(
                result[take_right], tree[query_right[take_right]]
            )
        query_left //= 2
        query_right //= 2
    return result


def sliding_deadline_local_cohort_sizes(
    event_times_s: np.ndarray,
    group_ids: np.ndarray,
    *,
    deadline_ms: float,
) -> np.ndarray:
    """Compute a per-event upper-bound cohort size for equal sliding deadlines.

    For event ``i`` with release ``t_i`` and deadline ``t_i + delta``, the
    returned value is the largest number of same-group event intervals sharing
    a candidate launch time inside event ``i``'s interval. Any accelerated
    event must belong to a cohort no larger than this value. The resulting
    threshold share is therefore a valid local-eligibility upper bound, but it
    need not be jointly achievable because overlapping opportunities can reuse
    the same events.
    """

    times = np.asarray(event_times_s, dtype=np.float64)
    groups = np.asarray(group_ids, dtype=np.int64)
    if times.ndim != 1 or groups.ndim != 1 or len(times) != len(groups):
        raise ValueError("event times and group IDs must be aligned one-dimensional arrays")
    if len(times) == 0:
        raise ValueError("at least one event is required")
    if deadline_ms <= 0:
        raise ValueError("deadline_ms must be positive")
    if not np.isfinite(times).all() or (times < 0).any() or (groups < 0).any():
        raise ValueError("event times and group IDs must be finite and nonnegative")

    time_ticks = _event_time_ticks(times)
    deadline_ticks = _duration_ticks(deadline_ms)
    if int(time_ticks.max()) > np.iinfo(np.int64).max - deadline_ticks:
        raise ValueError("event deadlines exceed the signed 64-bit nanosecond clock")
    output = np.empty(len(times), dtype=np.int64)
    for group_id in np.unique(groups):
        original_indices = np.flatnonzero(groups == group_id)
        order = np.argsort(time_ticks[original_indices], kind="stable")
        sorted_indices = original_indices[order]
        starts = time_ticks[sorted_indices]

        active_left = np.searchsorted(starts, starts - deadline_ticks, side="left")
        active_at_start = (
            np.arange(len(starts), dtype=np.int64) - active_left + 1
        )
        last_candidate = (
            np.searchsorted(starts, starts + deadline_ticks, side="right")
            - 1
        )
        local_sizes = _range_maximum_queries(
            active_at_start,
            np.arange(len(starts), dtype=np.int64),
            last_candidate,
        )
        output[sorted_indices] = local_sizes
    return output


def sliding_deadline_local_metrics(
    event_times_s: np.ndarray,
    group_ids: np.ndarray,
    *,
    deadline_ms: float,
    thresholds: Iterable[int],
) -> dict[str, float | int]:
    """Summarize the equal-deadline local-eligibility upper bound."""

    local_sizes = sliding_deadline_local_cohort_sizes(
        event_times_s,
        group_ids,
        deadline_ms=deadline_ms,
    )
    output: dict[str, float | int] = {
        "event_count": len(local_sizes),
        "local_cohort_size_event_mean": float(local_sizes.mean()),
        "local_cohort_size_event_p50": float(np.quantile(local_sizes, 0.50)),
        "local_cohort_size_event_p90": float(np.quantile(local_sizes, 0.90)),
        "local_cohort_size_event_p99": float(np.quantile(local_sizes, 0.99)),
        "local_cohort_size_max": int(local_sizes.max()),
    }
    for threshold in thresholds:
        if threshold <= 0:
            raise ValueError("thresholds must be positive")
        output[f"local_upper_share_k{threshold}"] = float(
            np.mean(local_sizes >= threshold)
        )
    return output


def exact_sliding_deadline_packing(
    event_times_s: np.ndarray,
    group_ids: np.ndarray,
    *,
    deadline_ms: float,
    threshold: int | Mapping[int, int],
) -> ExactPackingResult:
    """Solve equal-relative-deadline packing exactly in ``O(N log N)``.

    The model has zero service time, unlimited simultaneous GPU capacity, and
    same-group batches with a minimum cardinality quota. Floating release times
    and the deadline are first normalized to the nearest nanosecond; all
    feasibility comparisons are then exact integer comparisons on that clock.

    Within one group, an optimum can be uncrossed into disjoint contiguous
    blocks after sorting by release. The dynamic program chooses whether to
    skip the newest event or end such a block there. A monotone deque evaluates
    its sliding range maximum in linear time after sorting. Groups are
    independent and may have distinct quotas through a mapping.
    """

    times = np.asarray(event_times_s, dtype=np.float64)
    groups = np.asarray(group_ids, dtype=np.int64)
    if times.ndim != 1 or groups.ndim != 1 or len(times) != len(groups):
        raise ValueError("event times and group IDs must be aligned one-dimensional arrays")
    if len(times) == 0:
        raise ValueError("at least one event is required")
    if deadline_ms <= 0:
        raise ValueError("deadline_ms must be positive")
    if not np.isfinite(times).all() or (times < 0).any() or (groups < 0).any():
        raise ValueError("event times and group IDs must be finite and nonnegative")

    if isinstance(threshold, Mapping):
        group_thresholds = {int(key): int(value) for key, value in threshold.items()}
        if any(value <= 0 for value in group_thresholds.values()):
            raise ValueError("thresholds must be positive")
    else:
        scalar_threshold = int(threshold)
        if scalar_threshold <= 0:
            raise ValueError("threshold must be positive")
        group_thresholds = {
            int(group_id): scalar_threshold for group_id in np.unique(groups)
        }

    time_ticks = _event_time_ticks(times)
    deadline_ticks = _duration_ticks(deadline_ms)
    if int(time_ticks.max()) > np.iinfo(np.int64).max - deadline_ticks:
        raise ValueError("event deadlines exceed the signed 64-bit nanosecond clock")
    launch_times = np.full(len(times), np.nan, dtype=np.float64)
    total_batches = 0

    for group_id in np.unique(groups):
        quota = group_thresholds.get(int(group_id))
        if quota is None:
            raise ValueError(f"threshold mapping is missing group {int(group_id)}")
        original_indices = np.flatnonzero(groups == group_id)
        order = np.argsort(time_ticks[original_indices], kind="stable")
        sorted_indices = original_indices[order]
        starts = time_ticks[sorted_indices]
        event_count = len(starts)

        optimum = np.zeros(event_count + 1, dtype=np.int64)
        block_start = np.full(event_count + 1, -1, dtype=np.int64)
        candidate_starts: deque[int] = deque()

        for prefix_size in range(1, event_count + 1):
            newest_candidate = prefix_size - quota
            if newest_candidate >= 0:
                newest_score = int(optimum[newest_candidate]) - newest_candidate
                while candidate_starts:
                    last = candidate_starts[-1]
                    if int(optimum[last]) - last >= newest_score:
                        break
                    candidate_starts.pop()
                candidate_starts.append(newest_candidate)

            minimum_start = int(
                np.searchsorted(
                    starts,
                    starts[prefix_size - 1] - deadline_ticks,
                    side="left",
                )
            )
            while candidate_starts and candidate_starts[0] < minimum_start:
                candidate_starts.popleft()

            optimum[prefix_size] = optimum[prefix_size - 1]
            if candidate_starts:
                start = candidate_starts[0]
                take_value = prefix_size + int(optimum[start]) - start
                if take_value > optimum[prefix_size]:
                    optimum[prefix_size] = take_value
                    block_start[prefix_size] = start

        cursor = event_count
        while cursor > 0:
            start = int(block_start[cursor])
            if start < 0:
                cursor -= 1
                continue
            selected = sorted_indices[start:cursor]
            launch_tick = int(starts[start]) + deadline_ticks
            launch_times[selected] = launch_tick / _NANOSECONDS_PER_SECOND
            total_batches += 1
            cursor = start

    accelerated_count = int(np.isfinite(launch_times).sum())
    return ExactPackingResult(
        accelerated_event_count=accelerated_count,
        accelerated_share=accelerated_count / len(times),
        batch_count=total_batches,
        event_launch_times_s=launch_times,
        algorithm="unit_interval_monotone_dp",
    )
