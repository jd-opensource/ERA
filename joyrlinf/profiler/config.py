"""Profiler configuration for joyrlinf."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TorchProfilerToolConfig:
    """Torch profiler tool options."""

    contents: list[str] = field(default_factory=lambda: ["cuda", "cpu"])
    discrete: bool = False
    compress: bool = False


@dataclass
class ProfilerConfig:
    """Global profiler settings (YAML ``profiler`` section)."""

    enable: bool = False
    tool: str | None = None
    steps: list[int] | None = None
    # Per-role step override, e.g. {"rollout": [3], "actor": [2]}. Roles not
    # listed fall back to ``steps``. Needed on the async pipeline to capture
    # the same wall-clock window: actor step N trains the previous epoch's
    # batch, so it wall-overlaps rollout step N+1.
    role_steps: dict[str, list[int]] = field(default_factory=dict)
    profile_continuous_steps: bool = False
    save_path: str | None = None
    all_ranks: bool = False
    ranks: list[int] = field(default_factory=list)
    roles: list[str] = field(default_factory=list)
    tool_config: dict[str, Any] = field(default_factory=dict)

    @property
    def torch_config(self) -> TorchProfilerToolConfig:
        raw = self.tool_config.get("torch", {}) if self.tool_config else {}
        if hasattr(raw, "items"):
            items = dict(raw)
        else:
            items = {}
        contents = items.get("contents", ["cuda", "cpu"])
        if contents is None:
            contents = ["cuda", "cpu"]
        return TorchProfilerToolConfig(
            contents=list(contents),
            discrete=bool(items.get("discrete", False)),
            compress=bool(items.get("compress", False)),
        )


def parse_profiler_config(cfg: Any, *, output_dir: str) -> ProfilerConfig:
    """Parse ``profiler`` block from an OmegaConf / dict config."""
    block = cfg.get("profiler", {}) if hasattr(cfg, "get") else {}
    if block is None:
        block = {}

    def _get(key: str, default=None):
        if hasattr(block, "get"):
            return block.get(key, default)
        return getattr(block, key, default)

    steps = _get("steps", None)
    if steps is not None:
        steps = [int(s) for s in list(steps)]

    role_steps_raw = _get("role_steps", {}) or {}
    role_steps = {
        str(role).lower().strip(): [int(s) for s in list(role_list)]
        for role, role_list in dict(role_steps_raw).items()
        if role_list is not None
    }

    ranks = _get("ranks", [])
    if ranks is None:
        ranks = []
    ranks = [int(r) for r in list(ranks)]

    roles = _get("roles", [])
    if roles is None:
        roles = []
    roles = [str(r).lower().strip() for r in list(roles)]

    save_path = _get("save_path", None)
    if save_path is None:
        save_path = f"{output_dir.rstrip('/')}/profiler"

    tool_config = _get("tool_config", {})
    if tool_config is None:
        tool_config = {}
    if hasattr(tool_config, "items"):
        tool_config = dict(tool_config)

    return ProfilerConfig(
        enable=bool(_get("enable", False)),
        tool=_get("tool", None),
        steps=steps,
        role_steps=role_steps,
        profile_continuous_steps=bool(_get("profile_continuous_steps", False)),
        save_path=str(save_path),
        all_ranks=bool(_get("all_ranks", False)),
        ranks=ranks,
        roles=roles,
        tool_config=tool_config,
    )
