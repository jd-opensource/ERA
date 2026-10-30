"""Step-scoped profiler manager for embodied training loops."""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Iterator

from joyrlinf.profiler.config import ProfilerConfig, parse_profiler_config
from joyrlinf.profiler.torch_profiler import TorchProfilerBackend

logger = logging.getLogger(__name__)


class ProfilerManager:
    """Drive profiler start/stop around global training steps."""

    def __init__(
        self,
        *,
        rank: int,
        role: str,
        config: ProfilerConfig,
    ) -> None:
        self.rank = rank
        self.role = role.lower().strip()
        self.config = config
        self._backend: TorchProfilerBackend | None = None
        self._profile_window_active = False
        self._this_step = False
        # Role-specific steps win over the shared list (async pipeline: actor
        # step N wall-overlaps rollout step N+1, so pair rollout [N]/actor [N-1]
        # to trace the same wall-clock window).
        self._steps: list[int] = list(
            (config.role_steps or {}).get(self.role) or config.steps or []
        )

        if not config.enable or not config.tool:
            return
        if config.roles and self.role not in config.roles:
            logger.info(
                "[Profiler] disabled for rank=%d role=%s (roles filter=%s)",
                rank, self.role, config.roles,
            )
            return
        if not self._rank_selected(rank, config):
            logger.info(
                "[Profiler] disabled for rank=%d (ranks=%s all_ranks=%s)",
                rank, config.ranks, config.all_ranks,
            )
            return
        if not self._steps:
            logger.warning("[Profiler] enable=True but steps is empty; profiler disabled")
            return

        tool = str(config.tool).lower().strip()
        if tool == "torch":
            self._backend = TorchProfilerBackend(
                rank=rank,
                config=config,
                tool_config=config.torch_config,
            )
        else:
            raise ValueError(
                f"Unsupported profiler.tool={config.tool!r}; supported: torch"
            )

        logger.info(
            "[Profiler] ready rank=%d role=%s tool=%s steps=%s save_path=%s",
            rank, self.role, tool, self._steps, config.save_path,
        )

    @classmethod
    def from_cfg(
        cls,
        cfg: Any,
        *,
        rank: int,
        role: str,
        output_dir: str,
    ) -> "ProfilerManager":
        return cls(
            rank=rank,
            role=role,
            config=parse_profiler_config(cfg, output_dir=output_dir),
        )

    @property
    def enabled(self) -> bool:
        return self._backend is not None

    @property
    def discrete(self) -> bool:
        return self.enabled and self.config.torch_config.discrete

    @staticmethod
    def _rank_selected(rank: int, config: ProfilerConfig) -> bool:
        if config.all_ranks:
            return True
        if config.ranks:
            return rank in config.ranks
        return rank == 0

    def _should_start_step(self, step: int) -> bool:
        if not self._steps:
            return False
        if self.config.profile_continuous_steps:
            return min(self._steps) <= step <= max(self._steps)
        return step in self._steps

    def _should_stop_after_step(self, step: int) -> bool:
        if not self._steps:
            return False
        if self.config.profile_continuous_steps:
            return step >= max(self._steps)
        return step in self._steps

    def on_step_begin(self, step: int) -> None:
        if not self.enabled or self._backend is None:
            return
        if self.discrete:
            self._this_step = self._should_start_step(step)
            return
        if self._should_start_step(step) and not self._profile_window_active:
            self._backend.start(step=step, role=self.role)
            self._profile_window_active = True
        self._this_step = self._profile_window_active

    def on_step_end(self, step: int) -> None:
        if not self.enabled or self._backend is None:
            return
        if self.discrete:
            self._this_step = False
            return
        if self._profile_window_active and self._should_stop_after_step(step):
            self._backend.stop()
            self._profile_window_active = False
        self._this_step = self._profile_window_active

    def step(self) -> None:
        if self.enabled and self._backend is not None and self._profile_window_active:
            self._backend.step()

    @contextmanager
    def range(self, tag: str, step: int) -> Iterator[None]:
        """Profile a tagged region (discrete mode) or annotate (continuous mode)."""
        if not self.enabled or not self._this_step:
            yield
            return
        if self.discrete and self._backend is not None:
            with self._backend.discrete_range(step=step, role=self.role, tag=tag):
                yield
            return
        if self._backend is not None:
            with self._backend.record(tag):
                yield
            return
        yield

    def finalize(self) -> None:
        if self._backend is not None and self._backend.is_active():
            self._backend.stop()
            self._profile_window_active = False
