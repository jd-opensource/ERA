"""Regression tests for benchmark environment selection."""

from __future__ import annotations

import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def test_all_gr00t_manifests_select_gr00t_venv():
    for name in (
        "manifest_gr00t.json",
        "manifest_gr00t_maniskill.json",
        "manifest_gr00t_optimized.json",
    ):
        rows = json.loads((REPO / "bench" / name).read_text(encoding="utf-8"))
        assert rows, name
        assert all(row.get("venv") == "gr00t" for row in rows), name
