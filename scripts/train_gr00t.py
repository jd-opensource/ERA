"""joyrlinf GR00T training entry point."""
import os
import sys

local_rank = int(os.environ.get("LOCAL_RANK", 0))
os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)

import argparse
import logging
from datetime import datetime

import torch
import torch.distributed as dist
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from joyrlinf.runners.embodied_runner import EmbodiedRunner
from joyrlinf.utils.logging_setup import (
    GLOBAL_LOG_LEADER,
    current_rank,
    format_config_summary,
    log_at_rank,
    setup_logging,
    sync_run_timestamp,
)


def _torchrun_launched() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)

    if _torchrun_launched() and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = current_rank()

    if not cfg.runner.get("resume_dir", None) and "JOYRLINF_RUN_ID" not in os.environ:
        if dist.is_initialized():
            os.environ["JOYRLINF_RUN_ID"] = sync_run_timestamp()
        else:
            os.environ["JOYRLINF_RUN_ID"] = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_file = setup_logging(rank, args.config)

    from joyrlinf.core.gpu_memory_guard import build_alloc_conf, apply_memory_budget_from_cfg

    mem_cfg = cfg.get("memory", {})
    alloc_conf = build_alloc_conf(
        gc_threshold=mem_cfg.get("gc_threshold", 0.8),
        max_split_size_mb=mem_cfg.get("max_split_size_mb", 128),
        expandable_segments=mem_cfg.get("expandable_segments", True),
    )
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", alloc_conf)

    logger = logging.getLogger(__name__)
    from joyrlinf.utils.run_paths import resolve_run_output_dir

    output_dir = resolve_run_output_dir(cfg)
    log_at_rank(
        logger, rank, GLOBAL_LOG_LEADER, logging.INFO,
        "train_gr00t.py start: config=%s output_dir=%s log_file=%s node_rank=%s cwd=%s "
        "local_rank=%s visible_devices=%s CUDA_ALLOC_CONF=%s",
        args.config, output_dir, log_file, os.environ.get("NODE_RANK", "0"),
        os.getcwd(), os.environ.get("LOCAL_RANK", "0"),
        os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "<unset>"),
    )
    log_at_rank(
        logger, rank, GLOBAL_LOG_LEADER, logging.INFO,
        "Config summary: %s", format_config_summary(cfg),
    )

    gpu_guard = apply_memory_budget_from_cfg(cfg, device=0)

    runner = EmbodiedRunner(cfg, gpu_guard=gpu_guard)
    log_at_rank(
        logger, rank, GLOBAL_LOG_LEADER, logging.INFO,
        "EmbodiedRunner created, entering run loop",
    )
    runner.run()
    log_at_rank(
        logger, rank, GLOBAL_LOG_LEADER, logging.INFO,
        "Run loop finished, destroying process group",
    )
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
