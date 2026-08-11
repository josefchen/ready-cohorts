from __future__ import annotations

import torch

from gpu_agent_crossover.benchmark import AgentTransition, Case, make_cases
from gpu_agent_crossover.compiled_benchmark import (
    CompiledCase,
    make_rollout,
)
from gpu_agent_crossover.compiled_benchmark import (
    make_cases as make_compiled_cases,
)


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
