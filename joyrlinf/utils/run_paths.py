"""Resolve per-run artifact directory under ``results/``."""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path


def resolve_run_output_dir(cfg) -> str:
    """Return the directory for TB, ``config.yaml``, and ``checkpoints/``.

    - **续训**（``resume_dir``）：沿用 checkpoint 所在 run 目录。
    - **从头训**：``{log_path}/{experiment_name}/runs/{JOYRLINF_RUN_ID}/``；
      ``JOYRLINF_RUN_ID`` 由 ``train.py`` 在多机间广播对齐。
    """
    resume_dir = cfg.runner.get("resume_dir", None)
    if resume_dir:
        p = Path(str(resume_dir)).expanduser().resolve()
        if p.name.startswith("global_step_") and p.parent.name == "checkpoints":
            return str(p.parent.parent)
        if p.name == "actor" and p.parent.name.startswith("global_step_"):
            return str(p.parent.parent.parent)
        return str(p.parent)

    log_path = cfg.runner.logger.get("log_path", "./results")
    exp = cfg.runner.logger.get("experiment_name", "default")
    base = os.path.join(log_path, exp)

    run_id = os.environ.get("JOYRLINF_RUN_ID")
    if not run_id:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(base, "runs", run_id)
