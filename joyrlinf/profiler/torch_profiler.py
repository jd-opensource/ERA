"""PyTorch profiler backend — exports Chrome trace JSON."""
from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator, Optional

import torch

from joyrlinf.profiler.config import ProfilerConfig, TorchProfilerToolConfig

logger = logging.getLogger(__name__)


def _build_torch_profile(
    *,
    contents: list[str],
    save_path: str,
    rank: int,
    step: int,
    role: str,
    compress: bool,
) -> torch.profiler.profile:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    contents_set = set(contents or [])
    activities: list[torch.profiler.ProfilerActivity] = []
    if not contents_set or "cpu" in contents_set:
        activities.append(torch.profiler.ProfilerActivity.CPU)
    if not contents_set or "cuda" in contents_set:
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    def _trace_handler(prof: torch.profiler.profile) -> None:
        logger.info("[Profiler] Saving trace to %s", save_path)
        prof.export_chrome_trace(save_path)

    return torch.profiler.profile(
        activities=activities,
        with_stack="stack" in contents_set,
        record_shapes="shapes" in contents_set,
        profile_memory="memory" in contents_set,
        on_trace_ready=_trace_handler,
    )


def _trace_filename(
    *,
    save_dir: str,
    rank: int,
    step: int,
    role: str,
    tag: str | None,
    compress: bool,
) -> str:
    ts = datetime.now(tz=timezone.utc).astimezone().strftime("%Y%m%d%H%M%S%f")[:-3]
    pid = os.getpid()
    suffix = "json.gz" if compress else "json"
    parts = [f"step-{step}", f"rank-{rank}", f"role-{role}"]
    if tag:
        parts.append(f"tag-{tag}")
    parts.append(f"pid-{pid}")
    parts.append(ts)
    name = "_".join(parts) + f".trace.{suffix}"
    return os.path.join(save_dir, name)


class TorchProfilerBackend:
    """Wrap ``torch.profiler.profile`` with step/role scoped trace export."""

    def __init__(
        self,
        rank: int,
        config: ProfilerConfig,
        tool_config: TorchProfilerToolConfig,
    ) -> None:
        self.rank = rank
        self.config = config
        self.tool_config = tool_config
        self._prof: torch.profiler.profile | None = None
        self._active_step: int | None = None
        self._active_role: str | None = None

    @property
    def save_dir(self) -> str:
        role_dir = self._active_role or "default"
        return os.path.join(self.config.save_path or ".", role_dir)

    def start(self, *, step: int, role: str, tag: str | None = None) -> None:
        if self._prof is not None:
            return
        self._active_step = step
        self._active_role = role
        trace_path = _trace_filename(
            save_dir=self.save_dir,
            rank=self.rank,
            step=step,
            role=role,
            tag=tag,
            compress=self.tool_config.compress,
        )
        self._prof = _build_torch_profile(
            contents=self.tool_config.contents,
            save_path=trace_path,
            rank=self.rank,
            step=step,
            role=role,
            compress=self.tool_config.compress,
        )
        self._prof.start()
        logger.info(
            "[Profiler] started rank=%d step=%d role=%s -> %s",
            self.rank, step, role, trace_path,
        )

    def step(self) -> None:
        if self._prof is not None:
            self._prof.step()

    def stop(self) -> None:
        if self._prof is None:
            return
        try:
            self._prof.step()
            self._prof.stop()
            logger.info(
                "[Profiler] stopped rank=%d step=%s role=%s",
                self.rank, self._active_step, self._active_role,
            )
        finally:
            self._prof = None
            self._active_step = None
            self._active_role = None

    def is_active(self) -> bool:
        return self._prof is not None

    @contextmanager
    def discrete_range(
        self,
        *,
        step: int,
        role: str,
        tag: str,
    ) -> Iterator[None]:
        """Profile a single scoped region and write one trace file."""
        trace_path = _trace_filename(
            save_dir=os.path.join(self.config.save_path or ".", role),
            rank=self.rank,
            step=step,
            role=role,
            tag=tag,
            compress=self.tool_config.compress,
        )
        prof = _build_torch_profile(
            contents=self.tool_config.contents,
            save_path=trace_path,
            rank=self.rank,
            step=step,
            role=role,
            compress=self.tool_config.compress,
        )
        prof.start()
        try:
            with torch.profiler.record_function(tag):
                yield
        finally:
            prof.step()
            prof.stop()
            logger.info(
                "[Profiler] discrete trace rank=%d step=%d tag=%s -> %s",
                self.rank, step, tag, trace_path,
            )

    @contextmanager
    def record(self, name: str) -> Iterator[None]:
        if not self.is_active():
            yield
            return
        with torch.profiler.record_function(name):
            yield
