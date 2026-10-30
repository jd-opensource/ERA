"""基于 NVIDIA MPS 的单机/多机全共置异步训练启动器。

执行流程：
1. 外层 ``torchrun`` 按物理 GPU 数启动 N 个 EnvRollout 父进程。
2. 每个父进程根据 ``LOCAL_RANK`` 和启动时的数字型 ``CUDA_VISIBLE_DEVICES``
   选择物理 GPU，创建独立的 MPS pipe/log 目录，并启动该 GPU 的 MPS daemon。
3. 父进程在导入任何 CUDA 代码前启动一个独立 reaper。reaper 不导入
   PyTorch，只等待父进程退出，然后向对应 MPS daemon 发送 ``quit``，避免
   “父进程等待 daemon、daemon 等待父进程释放 CUDA context”的退出死锁。
4. 每个 EnvRollout 父进程再派生一个 Actor 子进程，因此每张物理 GPU 上有
   EnvRollout 和 Actor 两个 MPS 客户端。两类进程的逻辑 rank 分别为：

       EnvRollout: 0 .. N-1
       Actor:      N .. 2N-1

5. 2N 个逻辑 rank 使用 Gloo 默认进程组完成控制与 CPU batch 通信；N 个
   Actor rank 另外创建跨节点 NCCL 组用于 FSDP/梯度通信。同卡
   两个进程不能共用 NCCL 默认组，因此默认组必须是 Gloo。
6. EnvRollout 持续采样并把 batch 发送给同卡 Actor；Actor 独立训练并广播
   新权重，实现 rollout N+1 与 actor train N 的重叠。
7. 正常结束、异常结束或 Ctrl+C 时，父进程先终止/等待 Actor。父进程真正
   退出并释放 CUDA context 后，独立 reaper 再关闭 MPS daemon。

注意：所有可能间接导入 CUDA 的模块都必须在 MPS 环境变量配置完成后再导入；
此版本只接受数字 GPU id，不支持 UUID 或 MIG selector。
"""

from __future__ import annotations

import argparse
import logging
import os
import select
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
REAPER_LOG_NAME = "joyrlinf_reaper.log"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--topology-smoke",
        action="store_true",
        help="Validate expanded Gloo/NCCL MPS topology without building env/model",
    )
    parser.add_argument(
        "--mps-role",
        choices=("env_rollout", "actor"),
        default="env_rollout",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--physical-rank",
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--physical-world-size",
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--physical-local-world-size",
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--mps-reaper-parent-pid",
        type=int,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def _configure_process_env(
    *,
    role: str,
    physical_rank: int,
    physical_world_size: int,
    physical_local_world_size: int,
    cfg,
) -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", physical_rank))
    physical_device = os.environ.get("JOYRLINF_PHYSICAL_DEVICE_ID")
    if physical_device is None:
        physical_device = _resolve_physical_device_id(local_rank)
        os.environ["JOYRLINF_PHYSICAL_DEVICE_ID"] = physical_device

    # 父进程先用数字物理 GPU id 启动单卡 daemon；Actor 作为 CUDA/MPS 客户端
    # 只能看到重映射后的逻辑设备 0。robosuite 的导入检查也要求
    # MUJOCO_EGL_DEVICE_ID 出现在 CUDA_VISIBLE_DEVICES 里，所以父进程保持 0。
    # LIBERO EGL 子进程必须在 spawn 时改绑到 JOYRLINF_PHYSICAL_DEVICE_ID
    # （见 joyrlinf/envs/libero/venv.py），否则全部渲到物理 GPU 0。
    os.environ["CUDA_VISIBLE_DEVICES"] = physical_device if role == "env_rollout" else "0"
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"

    # 向放置和通信代码呈现包含两类 MPS 客户端的逻辑 world。
    os.environ["WORLD_SIZE"] = str(2 * physical_world_size)
    os.environ["LOCAL_WORLD_SIZE"] = str(2 * physical_local_world_size)
    expanded_rank = physical_rank if role == "env_rollout" else physical_world_size + physical_rank
    os.environ["RANK"] = str(expanded_rank)
    os.environ["JOYRLINF_MPS_ROLE"] = role
    node_rank = physical_rank // physical_local_world_size
    os.environ["JOYRLINF_PHYSICAL_RANK"] = str(physical_rank)
    os.environ["JOYRLINF_PHYSICAL_WORLD_SIZE"] = str(physical_world_size)
    os.environ["JOYRLINF_PHYSICAL_LOCAL_WORLD_SIZE"] = str(
        physical_local_world_size
    )
    os.environ["JOYRLINF_NODE_RANK"] = str(node_rank)

    mps_cfg = cfg.get("mps", {})
    pct_key = (
        "env_rollout_active_thread_percentage"
        if role == "env_rollout"
        else "actor_active_thread_percentage"
    )
    percentage = int(mps_cfg.get(pct_key, 100))
    if not 1 <= percentage <= 100:
        raise ValueError(f"mps.{pct_key} must be in [1, 100], got {percentage}")
    os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(percentage)

    session_id = os.environ.get("JOYRLINF_MPS_SESSION_ID")
    if not session_id:
        run_id = os.environ.get("TORCHELASTIC_RUN_ID", "run").replace("/", "_")
        session_id = f"{run_id}-{os.environ.get('MASTER_PORT', 'default')}-p{os.getpid()}"
        os.environ["JOYRLINF_MPS_SESSION_ID"] = session_id

    base = Path(mps_cfg.get("pipe_root", "/tmp/joyrlinf-mps"))
    pair_dir = base / f"{session_id}-node{node_rank}-gpu{local_rank}"
    pipe_dir = pair_dir / "pipe"
    log_dir = pair_dir / "log"
    pipe_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_MPS_PIPE_DIRECTORY"] = str(pipe_dir)
    os.environ["CUDA_MPS_LOG_DIRECTORY"] = str(log_dir)

    if os.environ.get("JOYRLINF_SUPPRESS_LIBERO_WARNINGS", "1") != "0":
        os.environ.setdefault(
            "PYTHONWARNINGS",
            "ignore::DeprecationWarning,ignore::FutureWarning,ignore::UserWarning",
        )

    _configure_allocator_env(cfg)


def _configure_allocator_env(cfg) -> None:
    mem_cfg = cfg.get("memory", {})
    alloc_parts = []
    if bool(mem_cfg.get("expandable_segments", True)):
        # max_split_size_mb must NOT be combined with expandable_segments:
        # oversize (>max_split) blocks inside expandable segments are never
        # reused, so every rollout predict maps ~20GB of fresh segments and
        # pt_rsv balloons until the colocated peer process OOMs.
        alloc_parts.append("expandable_segments:True")
    else:
        alloc_parts.append(
            f"max_split_size_mb:{int(mem_cfg.get('max_split_size_mb', 128))}"
        )
    gc_threshold = float(mem_cfg.get("gc_threshold", 0.8))
    if 0 < gc_threshold < 1:
        alloc_parts.append(f"garbage_collection_threshold:{gc_threshold}")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", ",".join(alloc_parts))


def _activate_mps_client_device() -> None:
    if "JOYRLINF_PHYSICAL_DEVICE_ID" not in os.environ:
        raise RuntimeError(
            "MPS physical GPU id was not resolved before client activation"
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"


def _resolve_physical_device_id(local_rank: int) -> str:
    """用 local rank 从启动时的数字可见设备列表中选择物理 GPU。"""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible:
        return str(local_rank)

    selectors = [item.strip() for item in visible.split(",") if item.strip()]
    if local_rank >= len(selectors):
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} is outside CUDA_VISIBLE_DEVICES={visible!r}"
        )
    physical_device = selectors[local_rank]
    if not physical_device.isdecimal():
        raise RuntimeError(
            "Rank-only MPS launch requires numeric CUDA_VISIBLE_DEVICES, got "
            f"selector {physical_device!r} from {visible!r}"
        )
    return physical_device


def _parse_rank_spec(spec: str) -> list[int]:
    ranks: list[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = (int(value) for value in part.split("-", 1))
            ranks.extend(range(lo, hi + 1))
        else:
            ranks.append(int(part))
    return ranks


def _validate_launch_shape(cfg, physical_world: int) -> None:
    placement = cfg.cluster.component_placement
    expected_env = list(range(physical_world))
    expected_actor = list(range(physical_world, 2 * physical_world))
    env_ranks = _parse_rank_spec(placement.env)
    rollout_ranks = _parse_rank_spec(placement.rollout)
    actor_ranks = _parse_rank_spec(placement.actor)
    if env_ranks != expected_env or rollout_ranks != expected_env:
        raise ValueError(
            "MPS launch requires env == rollout == 0..N-1; "
            f"N={physical_world}, env={env_ranks}, rollout={rollout_ranks}"
        )
    if actor_ranks != expected_actor:
        raise ValueError(
            "MPS launch requires actor == N..2N-1; "
            f"N={physical_world}, actor={actor_ranks}"
        )


def _validate_physical_topology(
    cfg,
    *,
    physical_world: int,
    physical_local_world: int,
) -> None:
    """Validate the outer torchrun topology before expanding logical ranks."""
    if physical_world <= 0 or physical_local_world <= 0:
        raise ValueError(
            "physical world sizes must be positive, got "
            f"world={physical_world}, local_world={physical_local_world}"
        )
    if physical_world % physical_local_world != 0:
        raise ValueError(
            f"physical WORLD_SIZE={physical_world} must be divisible by "
            f"LOCAL_WORLD_SIZE={physical_local_world}"
        )
    inferred_nodes = physical_world // physical_local_world
    configured_nodes = int(cfg.cluster.get("num_nodes", 1))
    if configured_nodes != inferred_nodes:
        raise ValueError(
            f"cluster.num_nodes={configured_nodes} does not match torchrun "
            f"topology {physical_world}/{physical_local_world}={inferred_nodes}"
        )
    configured_gpus = int(
        cfg.cluster.get("gpus_per_node", physical_local_world)
    )
    if configured_gpus != physical_local_world:
        raise ValueError(
            f"cluster.gpus_per_node={configured_gpus} does not match torchrun "
            f"LOCAL_WORLD_SIZE={physical_local_world}"
        )


def _start_mps_daemon() -> None:
    executable = shutil.which("nvidia-cuda-mps-control")
    if executable is None:
        raise RuntimeError(
            "mps.manage_daemon=true but nvidia-cuda-mps-control is not on PATH"
        )
    proc = subprocess.run(
        [executable, "-d"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "Failed to start NVIDIA MPS daemon for "
            f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}: "
            f"{proc.stderr.strip() or proc.stdout.strip()}"
        )


def _quit_mps_daemon(timeout_s: float = 30.0) -> int:
    executable = shutil.which("nvidia-cuda-mps-control")
    if executable is None:
        return 127
    try:
        proc = subprocess.run(
            [executable],
            input="quit\n",
            text=True,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return 124
    return proc.returncode


def _write_mps_reaper_status(message: str) -> None:
    log_dir = os.environ.get("CUDA_MPS_LOG_DIRECTORY")
    if not log_dir:
        return
    try:
        path = Path(log_dir) / REAPER_LOG_NAME
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"{timestamp} {message}\n")
    except OSError:
        pass


def _wait_for_process_exit(parent_pid: int) -> None:
    if parent_pid <= 1:
        raise ValueError(f"Invalid MPS reaper parent PID: {parent_pid}")

    pidfd_open = getattr(os, "pidfd_open", None)
    if pidfd_open is not None:
        try:
            pidfd = pidfd_open(parent_pid)
        except ProcessLookupError:
            return
        try:
            poller = select.poll()
            poller.register(pidfd, select.POLLIN)
            while True:
                try:
                    if poller.poll():
                        return
                except InterruptedError:
                    continue
        finally:
            os.close(pidfd)

    # 无 pidfd 时退化为轮询。reaper 由父进程立即创建，短时间内 PID 被复用的
    # 概率极低。
    while True:
        try:
            os.kill(parent_pid, 0)
        except ProcessLookupError:
            return
        except PermissionError:
            pass
        time.sleep(0.2)


def _run_mps_reaper(parent_pid: int) -> None:
    """等待持有 CUDA context 的父进程退出后再关闭 MPS。

    此进程不能导入 torch 或初始化 CUDA，并使用独立 session，避免发送给
    torchrun 的 Ctrl+C 同时中断清理过程。
    """
    pipe_dir = os.environ.get("CUDA_MPS_PIPE_DIRECTORY", "<unset>")
    _write_mps_reaper_status(f"waiting for parent pid={parent_pid} pipe={pipe_dir}")
    _wait_for_process_exit(parent_pid)
    _write_mps_reaper_status(f"parent pid={parent_pid} exited; sending quit")
    returncode = _quit_mps_daemon()
    _write_mps_reaper_status(f"MPS quit completed returncode={returncode}")
    if returncode == 0 and pipe_dir != "<unset>":
        shutil.rmtree(pipe_dir, ignore_errors=True)


def _start_mps_reaper(config_path: str) -> subprocess.Popen:
    command = [
        sys.executable,
        str(SCRIPT_PATH),
        "--config",
        config_path,
        "--mps-reaper-parent-pid",
        str(os.getpid()),
    ]
    return subprocess.Popen(
        command,
        env=os.environ.copy(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
    )


def _start_managed_mps(cfg, config_path: str) -> subprocess.Popen | None:
    if not bool(cfg.get("mps", {}).get("manage_daemon", True)):
        return None

    # 屏蔽 daemon 已启动但 reaper 尚未创建这一小段窗口；完成责任交接后，
    # 所有待处理信号会立即恢复投递。
    blocked_signals = {signal.SIGINT, signal.SIGTERM}
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, blocked_signals)
    try:
        _start_mps_daemon()
        try:
            return _start_mps_reaper(config_path)
        except BaseException:
            # 此时尚未激活 CUDA 客户端，因此可以安全地同步关闭 daemon。
            _quit_mps_daemon()
            raise
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)


def _run_training(args, cfg) -> None:
    # 可能初始化 CUDA 的依赖只能在客户端 MPS 环境变量就绪后导入。
    import torch.distributed as dist

    sys.path.insert(0, str(PROJECT_ROOT))
    from joyrlinf.envs.libero.quiet import suppress_libero_startup_noise

    suppress_libero_startup_noise()

    from joyrlinf.core.gpu_memory_guard import apply_memory_budget_from_cfg
    from joyrlinf.runners.embodied_runner_async_mps import EmbodiedRunner
    from joyrlinf.utils.logging_setup import (
        GLOBAL_LOG_LEADER,
        current_rank,
        format_config_summary,
        log_at_rank,
        setup_logging,
    )

    if not dist.is_initialized():
        dist.init_process_group(backend="gloo")
    if args.topology_smoke:
        _run_topology_smoke(dist, args)
        dist.destroy_process_group()
        return
    rank = current_rank()
    if not cfg.runner.get("resume_dir", None) and "JOYRLINF_RUN_ID" not in os.environ:
        run_id = [datetime.now().strftime("%Y%m%d_%H%M%S") if rank == 0 else None]
        dist.broadcast_object_list(run_id, src=0)
        os.environ["JOYRLINF_RUN_ID"] = run_id[0]

    log_file = setup_logging(rank, args.config)
    logger = logging.getLogger(__name__)
    from joyrlinf.utils.run_paths import resolve_run_output_dir

    output_dir = resolve_run_output_dir(cfg)
    log_at_rank(
        logger,
        rank,
        GLOBAL_LOG_LEADER,
        logging.INFO,
        "train_async_mps start: role=%s rank=%d config=%s output_dir=%s "
        "log_file=%s physical_gpu_id=%s MPS_thread_pct=%s",
        args.mps_role,
        rank,
        args.config,
        output_dir,
        log_file,
        os.environ.get("JOYRLINF_PHYSICAL_DEVICE_ID"),
        os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"),
    )
    log_at_rank(
        logger,
        rank,
        GLOBAL_LOG_LEADER,
        logging.INFO,
        "Config summary: %s",
        format_config_summary(cfg),
    )
    gpu_guard = apply_memory_budget_from_cfg(cfg, device=0)
    runner = EmbodiedRunner(cfg, gpu_guard=gpu_guard)
    runner.run()
    if dist.is_initialized():
        dist.destroy_process_group()


def _run_topology_smoke(dist, args) -> None:
    """Exercise cross-node Gloo pairs and the global Actor NCCL group."""
    import torch

    rank = dist.get_rank()
    physical_world = int(args.physical_world_size or (dist.get_world_size() // 2))
    actor_ranks = list(range(physical_world, 2 * physical_world))
    actor_group = dist.new_group(ranks=actor_ranks, backend="nccl")
    pair_groups = [
        dist.new_group(ranks=[rollout, physical_world + rollout], backend="gloo")
        for rollout in range(physical_world)
    ]

    if rank < physical_world:
        value = torch.tensor([rank], dtype=torch.int64)
        dist.send(
            value,
            dst=physical_world + rank,
            group=pair_groups[rank],
        )
    else:
        physical_rank = rank - physical_world
        value = torch.empty(1, dtype=torch.int64)
        dist.recv(value, src=physical_rank, group=pair_groups[physical_rank])
        if int(value.item()) != physical_rank:
            raise RuntimeError(
                f"Gloo pair mismatch rank={rank}: got {value.item()}, "
                f"expected {physical_rank}"
            )

        cuda_value = torch.tensor([rank + 1.0], device="cuda")
        dist.all_reduce(cuda_value, group=actor_group)
        expected = sum(actor_rank + 1.0 for actor_rank in actor_ranks)
        if float(cuda_value.item()) != expected:
            raise RuntimeError(
                f"Actor NCCL all-reduce mismatch: got {cuda_value.item()}, "
                f"expected {expected}"
            )

    dist.barrier()
    if rank == 0:
        print(
            f"MPS topology smoke OK: physical_world={physical_world} "
            f"logical_world={dist.get_world_size()}",
            flush=True,
        )


def _run_env_rollout_parent(
    args: argparse.Namespace,
    cfg,
    physical_rank: int,
    physical_world: int,
    physical_local_world: int,
) -> None:
    """托管本卡 MPS、Actor 子进程和 EnvRollout 训练主循环。"""
    # torchrun 在任一 rank 失败后会向其他 rank 发送 SIGTERM。将其转成异常，
    # 以便 finally 终止 Actor；MPS daemon 的关闭由独立 reaper 负责。
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def _raise_on_sigterm(signum, _frame):
        raise KeyboardInterrupt(f"received signal {signum}")

    signal.signal(signal.SIGTERM, _raise_on_sigterm)

    mps_reaper: subprocess.Popen | None = None
    actor: subprocess.Popen | None = None
    env_error: BaseException | None = None
    try:
        mps_reaper = _start_managed_mps(cfg, args.config)
        _activate_mps_client_device()
        actor_command = [
            sys.executable,
            str(SCRIPT_PATH),
            "--config",
            args.config,
            "--mps-role",
            "actor",
            "--physical-rank",
            str(physical_rank),
            "--physical-world-size",
            str(physical_world),
            "--physical-local-world-size",
            str(physical_local_world),
        ]
        if args.topology_smoke:
            actor_command.append("--topology-smoke")
        actor = subprocess.Popen(
            actor_command,
            env=os.environ.copy(),
        )
        _run_training(args, cfg)
    except BaseException as exc:
        env_error = exc
        if actor is not None and actor.poll() is None:
            actor.terminate()
        raise
    finally:
        actor_rc = None
        if actor is not None:
            try:
                actor_rc = actor.wait(timeout=30 if env_error is not None else None)
            except subprocess.TimeoutExpired:
                actor.kill()
                actor_rc = actor.wait()
        reaper_failed = mps_reaper is not None and mps_reaper.poll() is not None
        signal.signal(signal.SIGTERM, previous_sigterm)

        # 此处不能同步发送 MPS quit：本进程退出前仍持有 CUDA context，同步
        # 等待会造成循环等待。reaper 会在本进程真正退出后完成关闭。
        if env_error is None and reaper_failed:
            log_dir = os.environ.get("CUDA_MPS_LOG_DIRECTORY")
            raise RuntimeError(
                "MPS reaper exited before its CUDA client parent; see "
                f"{log_dir}/{REAPER_LOG_NAME}"
            )
        if env_error is None and actor_rc not in (None, 0):
            raise RuntimeError(f"Actor MPS subprocess exited with code {actor_rc}")


def main() -> bool:
    """Run one logical role; return whether the MPS parent needs a hard exit."""
    args = _parse_args()

    # reaper 必须在 OmegaConf、torch、distributed 或环境包可能初始化 CUDA
    # 之前进入独立分支。
    # reaper 来清理 MPS daemon，避免父进程和 daemon 互相等待释放 CUDA context。
    if args.mps_reaper_parent_pid is not None:
        _run_mps_reaper(args.mps_reaper_parent_pid)
        return False

    from omegaconf import OmegaConf

    cfg = OmegaConf.load(args.config)

    # --------------------- Rank 和 world_size 的物理/逻辑映射 ---------------------
    original_rank = int(os.environ.get("RANK", 0))
    original_world = int(os.environ.get("WORLD_SIZE", 1))
    original_local_world = int(os.environ.get("LOCAL_WORLD_SIZE", original_world))
    physical_rank = original_rank if args.physical_rank is None else args.physical_rank
    physical_world = (
        original_world if args.physical_world_size is None else args.physical_world_size
    )
    physical_local_world = (
        original_local_world
        if args.physical_local_world_size is None
        else args.physical_local_world_size
    )
    # 逻辑 rank：进程的 RANK，包含 EnvRollout 和 Actor 两类 MPS 客户端。
    # 物理 rank：物理 GPU 的索引，EnvRollout 和 Actor 共享同一张物理 GPU。
    # ---------------------------------------------------------------------------

    _validate_physical_topology(
        cfg,
        physical_world=physical_world,
        physical_local_world=physical_local_world,
    )

    # --------------------- 检查 launch 配置是否符合 MPS 约束 ---------------------
    _validate_launch_shape(cfg, physical_world)
    # --------------------------------------------------------------------------

    # --------------------- 配置 MPS 环境变量 ---------------------
    _configure_process_env(
        role=args.mps_role,
        physical_rank=physical_rank,
        physical_world_size=physical_world,
        physical_local_world_size=physical_local_world,
        cfg=cfg,
    )
    # ------------------------------------------------------------

    if args.mps_role == "actor":
        _activate_mps_client_device()
        _run_training(args, cfg)
        return True

    _run_env_rollout_parent(
        args,
        cfg,
        physical_rank,
        physical_world,
        physical_local_world,
    )
    return True


if __name__ == "__main__":
    hard_exit = main()
    if hard_exit:
        # PyTorch/CUDA, simulation engines, and the MPS client all register
        # native finalizers.  After explicit runner/process-group/child cleanup,
        # their interpreter-shutdown ordering can still segfault.  Let the OS
        # release the completed client's resources atomically.  For parent
        # clients, the independent reaper observes this exit and then owns
        # daemon shutdown.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
