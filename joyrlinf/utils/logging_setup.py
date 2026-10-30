"""Central logging setup and rank-aware helpers for joyrlinf training."""
from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

try:
    import fcntl
except ImportError:
    fcntl = None

# Global torchrun rank used for one-shot startup messages (config, placement, etc.).
GLOBAL_LOG_LEADER = 0

_THIRD_PARTY_LOGGERS = (
    "datasets",
    "httpx",
    "filelock",
    "urllib3",
    "PIL",
)


class LockedFileHandler(logging.FileHandler):
    """Multi-process safe file handler (POSIX fcntl)."""

    def emit(self, record):
        if self.stream is None:
            self.stream = self._open()
        if fcntl is None:
            return super().emit(record)
        try:
            fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX)
            try:
                super().emit(record)
            finally:
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
        except Exception:
            self.handleError(record)


class TqdmLoggingHandler(logging.StreamHandler):
    """Console handler that emits records via ``tqdm.write``.

    Plain ``StreamHandler`` output collides with an active tqdm progress bar:
    the bar redraws with carriage returns and overwrites (or is overwritten by)
    log lines, so messages appear to vanish. Routing every record through
    ``tqdm.write`` clears the bars, prints the line, then redraws the bars,
    letting logging and progress bars coexist cleanly.
    """

    def emit(self, record):
        try:
            msg = self.format(record)
        except Exception:
            self.handleError(record)
            return
        try:
            from tqdm import tqdm

            tqdm.write(msg, file=self.stream, end=self.terminator)
        except Exception:
            try:
                self.stream.write(msg + self.terminator)
                self.flush()
            except Exception:
                self.handleError(record)


def current_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def is_debug_mode() -> bool:
    return os.environ.get("JOYRLINF_DEBUG", "0") == "1"


def is_verbose_comm() -> bool:
    """When set, comm/barrier hot-path logs stay at INFO instead of DEBUG."""
    return is_debug_mode() or os.environ.get("JOYRLINF_VERBOSE_COMM", "0") == "1"


def log_at_rank(
    logger: logging.Logger,
    rank: int,
    leader_rank: int,
    level: int,
    msg: str,
    *args,
    **kwargs,
) -> None:
    if rank == leader_rank:
        logger.log(level, msg, *args, **kwargs)


def log_at_global_leader(
    logger: logging.Logger,
    level: int,
    msg: str,
    *args,
    **kwargs,
) -> None:
    log_at_rank(logger, current_rank(), GLOBAL_LOG_LEADER, level, msg, *args, **kwargs)


def configure_third_party_loggers(level: int = logging.WARNING) -> None:
    for name in _THIRD_PARTY_LOGGERS:
        logging.getLogger(name).setLevel(level)


def format_config_summary(cfg: Any) -> str:
    algo = cfg.get("algorithm", {})
    env_train = cfg.get("env", {}).get("train", {})
    actor = cfg.get("actor", {})
    cluster = cfg.get("cluster", {})
    placement = cluster.get("component_placement", {})
    runner = cfg.get("runner", {})
    return (
        f"pipeline={algo.get('pipeline_mode', 'sync')} "
        f"comm_mode={algo.get('comm_mode', 'v1')} "
        f"adv={algo.get('adv_type', '?')} "
        f"loss={algo.get('loss_type', '?')} "
        f"envs={env_train.get('total_num_envs', '?')} "
        f"ga={actor.get('gradient_accumulation_steps', '?')} "
        f"mbs={actor.get('micro_batch_size', '?')} "
        f"max_steps={runner.get('max_steps', '?')} "
        f"placement(env={placement.get('env', '?')} "
        f"rollout={placement.get('rollout', '?')} "
        f"actor={placement.get('actor', '?')})"
    )


def _torchrun_launched() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def _is_multi_node_job() -> bool:
    world = dist.get_world_size()
    local_ws = int(os.environ.get("LOCAL_WORLD_SIZE", str(world)))
    return world > local_ws


def sync_run_timestamp() -> str:
    """One second-precision timestamp for all ranks (multi-node safe)."""
    if not dist.is_initialized():
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    rank = dist.get_rank()
    world = dist.get_world_size()
    ts_epoch = int(datetime.now().timestamp())
    if _is_multi_node_job():
        gloo_group = dist.new_group(ranks=list(range(world)), backend="gloo")
        ts_cpu = torch.zeros(1, dtype=torch.int64)
        if rank == 0:
            ts_cpu.fill_(ts_epoch)
        dist.broadcast(ts_cpu, src=0, group=gloo_group)
        return datetime.fromtimestamp(int(ts_cpu.item())).strftime("%Y%m%d_%H%M%S")
    torch.cuda.set_device(0)
    ts_tensor = torch.zeros(1, dtype=torch.int64, device="cuda:0")
    if rank == 0:
        ts_tensor.fill_(ts_epoch)
    dist.broadcast(ts_tensor, src=0)
    return datetime.fromtimestamp(int(ts_tensor.item())).strftime("%Y%m%d_%H%M%S")


# Remembered args from the last setup_logging call so we can rebuild the same
# handlers after a third-party lib hijacks the root logger (see reassert_logging).
_LOG_STATE: dict | None = None


def _install_root_handlers(
    rank: int, log_file: Path, console_level: int, root_level: int
) -> None:
    root = logging.getLogger()
    root.setLevel(root_level)
    root.handlers.clear()

    console_handler = TqdmLoggingHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(logging.Formatter(f"[rank {rank}] %(message)s"))
    root.addHandler(console_handler)

    fh = LockedFileHandler(log_file, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        f"%(asctime)s [rank {rank}] [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root.addHandler(fh)

    configure_third_party_loggers()


def setup_logging(rank: int, config_path: str, log_dir: str = "logs") -> Path:
    """Attach console + locked file handlers for this process."""
    is_debug = is_debug_mode()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if is_debug:
        console_level = logging.DEBUG
    elif local_rank == 0:
        console_level = logging.INFO
    else:
        console_level = logging.WARNING

    config_name = Path(config_path).stem
    base = Path(log_dir)
    base.mkdir(parents=True, exist_ok=True)
    ts = os.environ.get("JOYRLINF_RUN_ID")
    if not ts:
        ts = sync_run_timestamp()
    node_rank = int(os.environ.get("NODE_RANK", "0"))
    log_file = base / f"{ts}_{config_name}_node{node_rank}.log"

    root_level = logging.DEBUG if is_debug else logging.INFO
    _install_root_handlers(rank, log_file, console_level, root_level)

    global _LOG_STATE
    _LOG_STATE = dict(
        rank=rank, log_file=log_file,
        console_level=console_level, root_level=root_level,
    )
    return log_file


def reassert_logging() -> None:
    """Reinstall joyrlinf's root handlers, undoing third-party logging hijacks.

    Some dependencies (e.g. ``prismatic.overwatch``) run
    ``logging.config.dictConfig`` at import time, which clears our console + file
    handlers and installs a RichHandler with its own ``| >>`` format. Call this
    after importing/loading such a library to restore clean ``[rank N]`` console
    output and resume writing to the same run log file. No-op if setup_logging
    has not run yet in this process.
    """
    if _LOG_STATE is None:
        return
    _install_root_handlers(**_LOG_STATE)
