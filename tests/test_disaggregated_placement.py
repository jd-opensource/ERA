"""Validation tests for fully disaggregated env/rollout topology."""

from __future__ import annotations

import pytest

from joyrlinf.core.launcher import Placement, create_process_groups


def test_disaggregated_requires_one_env_rank_per_rollout_rank():
    placement = Placement(
        env_ranks=[0, 1, 2, 3],
        rollout_ranks=[4, 5],
        actor_ranks=[6, 7],
    )
    with pytest.raises(ValueError, match="1:1 rank mapping"):
        create_process_groups(
            placement,
            comm_mode="v1",
            pipeline_mode="async",
        )
