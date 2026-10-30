"""Per-role profiler step selection (async wall-clock alignment)."""
from joyrlinf.profiler.config import parse_profiler_config
from joyrlinf.profiler.manager import ProfilerManager


def _cfg(**profiler_block):
    return {"profiler": profiler_block}


class TestParseRoleSteps:
    def test_role_steps_parsed_and_normalized(self):
        cfg = parse_profiler_config(
            _cfg(steps=[3], role_steps={"Rollout": [3], "ACTOR": ["2"]}),
            output_dir="/tmp/out",
        )
        assert cfg.role_steps == {"rollout": [3], "actor": [2]}
        assert cfg.steps == [3]

    def test_role_steps_default_empty(self):
        cfg = parse_profiler_config(_cfg(steps=[2]), output_dir="/tmp/out")
        assert cfg.role_steps == {}


class TestManagerRoleSteps:
    def _manager(self, role: str, **profiler_block) -> ProfilerManager:
        config = parse_profiler_config(_cfg(**profiler_block), output_dir="/tmp/out")
        return ProfilerManager(rank=0, role=role, config=config)

    def test_role_override_wins_over_shared_steps(self):
        m = self._manager("actor", steps=[3], role_steps={"actor": [2]})
        assert m._steps == [2]
        assert m._should_start_step(2)
        assert not m._should_start_step(3)

    def test_unlisted_role_falls_back_to_steps(self):
        m = self._manager("rollout", steps=[3], role_steps={"actor": [2]})
        assert m._steps == [3]
        assert m._should_start_step(3)
        assert not m._should_start_step(2)

    def test_async_pair_covers_same_wall_window(self):
        """rollout [N] / actor [N-1] is the aligned pair on the async runner."""
        rollout = self._manager(
            "rollout", steps=[3], role_steps={"rollout": [3], "actor": [2]},
        )
        actor = self._manager(
            "actor", steps=[3], role_steps={"rollout": [3], "actor": [2]},
        )
        assert rollout._steps == [3]
        assert actor._steps == [2]
        assert actor._should_stop_after_step(2)
        assert rollout._should_stop_after_step(3)
