from __future__ import annotations

import numpy as np
import pytest
import torch

from gpu_agent_crossover.benchmark import AgentTransition, Case, make_cases
from gpu_agent_crossover.compiled_benchmark import (
    CompiledCase,
    make_rollout,
)
from gpu_agent_crossover.compiled_benchmark import (
    make_cases as make_compiled_cases,
)
from gpu_agent_crossover.ready_cohort import (
    cohort_metrics,
    exact_sliding_deadline_packing,
    sliding_deadline_local_cohort_sizes,
    sliding_deadline_local_metrics,
)
from gpu_agent_crossover.trace_features import extract_session_features, parse_json_list


def test_transition_is_seed_deterministic() -> None:
    left = AgentTransition(128, 8, 4, torch.device("cpu"), seed=17)
    right = AgentTransition(128, 8, 4, torch.device("cpu"), seed=17)
    for _ in range(3):
        left_actions = left.step()
        right_actions = right.step()
    assert torch.equal(left_actions, right_actions)
    assert torch.equal(left.state, right.state)
    assert torch.equal(left.budget, right.budget)


def test_case_id_is_stable_and_sensitive() -> None:
    first = Case(256, 8, 1, "cpu", 1)
    same = Case(256, 8, 1, "cpu", 1)
    changed = Case(256, 8, 1, "cpu", 8)
    assert first.case_id == same.case_id
    assert first.case_id != changed.case_id


def test_make_cases_builds_full_factorial() -> None:
    config = {
        "experiment": {
            "agent_counts": [1, 2],
            "state_widths": [4],
            "action_counts": [1, 2],
            "cpu_threads": [1, 2],
            "gpu_modes": ["eager-resident"],
            "seed": 3,
            "randomize_case_order": False,
        }
    }
    assert len(make_cases(config)) == 2 * 1 * 2 * (2 + 1)


def test_compiled_rollout_matches_repeated_transition() -> None:
    transition = AgentTransition(128, 8, 4, torch.device("cpu"), seed=19)
    compiled_inputs = (
        transition.state.clone(),
        transition.budget.clone(),
        transition.weights,
        transition.action_deltas,
        transition.action_costs,
        transition.thresholds,
    )
    eager = AgentTransition(128, 8, 4, torch.device("cpu"), seed=19)
    rollout = make_rollout(4)
    compiled_state, compiled_budget, compiled_actions = rollout(*compiled_inputs)
    for _ in range(4):
        eager_actions = eager.step()
    assert torch.equal(compiled_actions, eager_actions)
    assert torch.allclose(compiled_state, eager.state)
    assert torch.equal(compiled_budget, eager.budget)


def test_make_compiled_cases_builds_full_factorial() -> None:
    config = {
        "experiment": {
            "agent_counts": [1, 2],
            "state_widths": [4],
            "action_counts": [1, 2],
            "observation_horizons": [1, 4],
            "cpu_threads": [1, 2],
            "gpu_visibility": ["resident", "host-visible"],
            "seed": 3,
            "randomize_case_order": False,
        }
    }
    cases = make_compiled_cases(config)
    assert len(cases) == 2 * 1 * 2 * 2 * (2 + 2)
    assert all(isinstance(case, CompiledCase) for case in cases)


def test_trace_extraction_emits_metadata_without_message_content() -> None:
    row = {
        "harness": "tool_calling",
        "benchmark": "tau2_retail",
        "models": ["test-model"],
        "session_id": "session-1",
        "spans": [
            {
                "span_id": "span-1",
                "type": "llm_call",
                "start_time": "2026-01-01T00:00:00+00:00",
                "end_time": "2026-01-01T00:00:02+00:00",
                "attributes": {
                    "gen_ai.usage.input_tokens": 10,
                    "gen_ai.usage.output_tokens": 5,
                    "gen_ai.input.messages": '[{"role":"user","parts":[]}]',
                    "gen_ai.output.messages": (
                        '[{"role":"assistant","parts":[{"type":"tool_call",'
                        '"name":"lookup","arguments":{"secret":"not-emitted"}}]}]'
                    ),
                    "gen_ai.tool.definitions": '[{"name":"lookup"}]',
                    "gen_ai.response.finish_reasons": ["tool_calls"],
                },
                "status": {"code": 1, "message": ""},
            }
        ],
    }
    spans, session = extract_session_features(row, source_file="sample.parquet", source_row=0)
    assert spans[0]["route_key"] == "tool:lookup"
    assert spans[0]["tool_call_count"] == 1
    assert spans[0]["input_chars"] > 0
    assert spans[0]["output_chars"] > 0
    assert "secret" not in str(spans[0])
    assert session["tool_call_count"] == 1
    assert session["effective_route_count"] == 1.0


def test_parse_json_list_rejects_non_list_and_invalid_json() -> None:
    assert parse_json_list("[]") == ([], True)
    assert parse_json_list("{}") == ([], False)
    assert parse_json_list("not-json") == ([], False)


def test_cohort_metrics_are_event_weighted() -> None:
    metrics = cohort_metrics(
        np.array([0.001, 0.002, 0.011]),
        np.zeros(3, dtype=np.int32),
        window_ms=10,
        thresholds=[2, 3],
    )

    assert metrics["cohort_count"] == 2
    assert metrics["cohort_size_event_mean"] == pytest.approx(5 / 3)
    assert metrics["eligible_share_k2"] == pytest.approx(2 / 3)
    assert metrics["eligible_share_k3"] == 0


def test_finer_grouping_cannot_increase_threshold_eligibility() -> None:
    times = np.array([0.001, 0.002, 0.003, 0.011])
    pooled = cohort_metrics(
        times,
        np.zeros(4, dtype=np.int32),
        window_ms=10,
        thresholds=[2],
    )
    split = cohort_metrics(
        times,
        np.array([0, 1, 1, 0], dtype=np.int32),
        window_ms=10,
        thresholds=[2],
    )

    assert pooled["eligible_share_k2"] >= split["eligible_share_k2"]


def _brute_local_sizes(times: np.ndarray, groups: np.ndarray, deadline_s: float) -> np.ndarray:
    output = []
    for index, (release, group) in enumerate(zip(times, groups, strict=True)):
        candidates = times[
            (times >= release)
            & (times <= release + deadline_s)
            & (groups == group)
        ]
        best = 0
        for launch in candidates:
            active = (
                (groups == group)
                & (times <= launch)
                & (times + deadline_s >= launch)
            )
            best = max(best, int(active.sum()))
        assert best >= 1, index
        output.append(best)
    return np.asarray(output)


def test_sliding_deadline_local_sizes_match_brute_force() -> None:
    rng = np.random.default_rng(20260811)
    for _ in range(30):
        times = np.round(rng.uniform(0, 0.2, size=25), decimals=3)
        groups = rng.integers(0, 4, size=len(times), dtype=np.int64)
        observed = sliding_deadline_local_cohort_sizes(
            times,
            groups,
            deadline_ms=35,
        )
        expected = _brute_local_sizes(times, groups, 0.035)
        np.testing.assert_array_equal(observed, expected)


def test_fixed_window_eligibility_is_bounded_by_local_eligibility() -> None:
    times = np.array([0.001, 0.009, 0.011, 0.019, 0.021, 0.029])
    groups = np.zeros(len(times), dtype=np.int64)
    fixed = cohort_metrics(times, groups, window_ms=10, thresholds=[2, 3])
    local = sliding_deadline_local_metrics(
        times,
        groups,
        deadline_ms=10,
        thresholds=[2, 3],
    )
    assert fixed["eligible_share_k2"] <= local["local_upper_share_k2"]
    assert fixed["eligible_share_k3"] <= local["local_upper_share_k3"]


def _brute_exact_packing_count(
    times: np.ndarray,
    groups: np.ndarray,
    deadline_s: float,
    threshold: int,
) -> int:
    tick_rate = 1_000_000_000
    time_ticks = np.rint(times * tick_rate).astype(np.int64)
    deadline_ticks = round(deadline_s * tick_rate)
    deadlines = time_ticks + deadline_ticks
    candidates = np.unique(deadlines)
    choices = []
    for release, deadline in zip(time_ticks, deadlines, strict=True):
        feasible = [
            index
            for index, launch in enumerate(candidates)
            if release <= launch <= deadline
        ]
        choices.append([-1, *feasible])

    best = 0

    def search(index: int, assignments: list[int]) -> None:
        nonlocal best
        if index == len(times):
            assigned = 0
            for candidate_index in range(len(candidates)):
                for group in np.unique(groups):
                    members = [
                        event_index
                        for event_index, assignment in enumerate(assignments)
                        if assignment == candidate_index and groups[event_index] == group
                    ]
                    if members and len(members) < threshold:
                        return
                    assigned += len(members)
            best = max(best, assigned)
            return
        for choice in choices[index]:
            assignments.append(choice)
            search(index + 1, assignments)
            assignments.pop()

    search(0, [])
    return best


def test_exact_sliding_packing_matches_brute_force() -> None:
    rng = np.random.default_rng(20260812)
    for _ in range(12):
        times = np.round(rng.uniform(0, 0.04, size=6), decimals=3)
        groups = rng.integers(0, 2, size=len(times), dtype=np.int64)
        observed = exact_sliding_deadline_packing(
            times,
            groups,
            deadline_ms=12,
            threshold=2,
        )
        expected = _brute_exact_packing_count(times, groups, 0.012, 2)
        assert observed.accelerated_event_count == expected
        assigned = np.isfinite(observed.event_launch_times_s)
        normalized_times = np.rint(times * 1e9) / 1e9
        assert np.all(
            observed.event_launch_times_s[assigned] >= normalized_times[assigned]
        )
        assert np.all(
            observed.event_launch_times_s[assigned]
            <= normalized_times[assigned] + 0.012
        )


def test_exact_packing_lies_between_fixed_and_local_bounds() -> None:
    times = np.array([0.009, 0.011, 0.019])
    groups = np.zeros(len(times), dtype=np.int64)
    fixed = cohort_metrics(times, groups, window_ms=10, thresholds=[2])
    exact = exact_sliding_deadline_packing(
        times,
        groups,
        deadline_ms=10,
        threshold=2,
    )
    local = sliding_deadline_local_metrics(
        times,
        groups,
        deadline_ms=10,
        thresholds=[2],
    )
    assert fixed["eligible_share_k2"] <= exact.accelerated_share
    assert exact.accelerated_share <= local["local_upper_share_k2"]


def test_local_bound_can_be_strictly_above_exact_packing() -> None:
    times = np.array([0.000, 0.009, 0.018])
    groups = np.zeros(len(times), dtype=np.int64)
    exact = exact_sliding_deadline_packing(
        times,
        groups,
        deadline_ms=10,
        threshold=2,
    )
    local = sliding_deadline_local_metrics(
        times,
        groups,
        deadline_ms=10,
        thresholds=[2],
    )
    assert exact.accelerated_share == pytest.approx(2 / 3)
    assert local["local_upper_share_k2"] == 1.0


def test_exact_packing_uses_integer_clock_without_epsilon_edges() -> None:
    times = np.array([0.0, 0.0010000006])
    groups = np.zeros(2, dtype=np.int64)
    exact = exact_sliding_deadline_packing(
        times,
        groups,
        deadline_ms=1,
        threshold=2,
    )
    assert exact.accelerated_event_count == 0


def test_exact_packing_supports_route_specific_thresholds() -> None:
    times = np.array([0.0, 0.001, 0.0, 0.001])
    groups = np.array([0, 0, 1, 1], dtype=np.int64)
    exact = exact_sliding_deadline_packing(
        times,
        groups,
        deadline_ms=2,
        threshold={0: 2, 1: 3},
    )
    assert exact.accelerated_event_count == 2
    assert np.isfinite(exact.event_launch_times_s[:2]).all()
    assert np.isnan(exact.event_launch_times_s[2:]).all()


@pytest.mark.parametrize(
    "times_ms",
    [
        [0, 8, 9, 10, 18, 19],
        [0, 1, 2, 10, 11, 12, 21],
    ],
)
def test_exact_packing_avoids_plausible_greedy_failures(
    times_ms: list[int],
) -> None:
    times = np.asarray(times_ms, dtype=np.float64) / 1_000.0
    exact = exact_sliding_deadline_packing(
        times,
        np.zeros(len(times), dtype=np.int64),
        deadline_ms=10,
        threshold=3,
    )
    assert exact.accelerated_event_count == len(times)
