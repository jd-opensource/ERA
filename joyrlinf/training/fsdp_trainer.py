"""FSDP training utilities for actor workers."""
from __future__ import annotations
import functools
import logging
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision
from torch.distributed.fsdp.wrap import (
    _module_wrap_policy,
    _or_policy,
    transformer_auto_wrap_policy,
)
from transformers.trainer_pt_utils import get_module_class_from_name

logger = logging.getLogger(__name__)

SHARDING_MAP = {
    "full_shard": ShardingStrategy.FULL_SHARD,
    "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
    "no_shard": ShardingStrategy.NO_SHARD,
    "hybrid_shard": ShardingStrategy.HYBRID_SHARD,
}


def _get_mixed_precision(cfg):
    mp_cfg = cfg.get("mixed_precision", None)
    if mp_cfg is None:
        return None
    # Match RLinf config spelling.  Experiment configs commonly use the short
    # names (bf16/fp16/fp32), while torch APIs need concrete dtype objects.
    dtype_map = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        None: None,
    }
    pd = dtype_map.get(str(mp_cfg.get("param_dtype")))
    rd = dtype_map.get(str(mp_cfg.get("reduce_dtype")))
    bd = dtype_map.get(str(mp_cfg.get("buffer_dtype")))
    if all(d is None for d in [pd, rd, bd]):
        return None
    return MixedPrecision(
        param_dtype=pd,
        reduce_dtype=rd,
        buffer_dtype=bd,
        keep_low_precision_grads=bool(mp_cfg.get("keep_low_precision_grads", True)),
    )


def compute_grad_norm_fp32(
    parameters,
    norm_type: float = 2.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Global grad norm in fp32 (matches RLinf mixed-precision reporting)."""
    params = [p for p in parameters if p.grad is not None]
    if not params:
        dev = device if device is not None else torch.device("cpu")
        return torch.zeros((), device=dev, dtype=torch.float32)
    grads = [p.grad.detach().float() for p in params]
    if float(norm_type) == float("inf"):
        return torch.stack([g.abs().max() for g in grads]).max()
    return torch.linalg.vector_norm(
        torch.stack([torch.linalg.vector_norm(g, norm_type) for g in grads]),
        norm_type,
    )


def _optimizer_params_with_grad(optimizer) -> list[torch.nn.Parameter]:
    params: list[torch.nn.Parameter] = []
    for group in optimizer.param_groups:
        for param in group.get("params", []):
            if param is not None and param.grad is not None:
                params.append(param)
    return params


@torch.no_grad()
def diagnose_grad_norm(
    model,
    optimizer,
    device: torch.device | None = None,
    norm_type: float = 2.0,
) -> dict[str, float | int | str]:
    """Compare grad-norm methods without clipping (debug / RLinf alignment)."""
    model_params = [p for p in model.parameters() if p.grad is not None]
    opt_params = _optimizer_params_with_grad(optimizer)

    def _count_dtypes(params: list[torch.nn.Parameter]) -> str:
        counts: dict[str, int] = {}
        for param in params:
            key = str(param.grad.dtype)
            counts[key] = counts.get(key, 0) + 1
        return ",".join(f"{k}:{v}" for k, v in sorted(counts.items()))

    def _total_numel(params: list[torch.nn.Parameter]) -> int:
        return int(sum(p.grad.numel() for p in params))

    report: dict[str, float | int | str] = {
        "model_param_tensors": len(model_params),
        "opt_param_tensors": len(opt_params),
        "model_grad_numel": _total_numel(model_params),
        "opt_grad_numel": _total_numel(opt_params),
        "model_grad_dtypes": _count_dtypes(model_params),
        "opt_grad_dtypes": _count_dtypes(opt_params),
    }
    if not model_params:
        report["norm_model_fp32"] = 0.0
        report["norm_model_torch"] = 0.0
        report["norm_opt_fp32"] = 0.0
        return report

    report["norm_model_fp32"] = float(
        compute_grad_norm_fp32(model_params, norm_type=norm_type, device=device).item()
    )
    report["norm_model_torch"] = float(
        torch.nn.utils.get_total_norm(
            [p.grad.detach() for p in model_params], norm_type=norm_type
        ).item()
    )
    if opt_params:
        report["norm_opt_fp32"] = float(
            compute_grad_norm_fp32(opt_params, norm_type=norm_type, device=device).item()
        )
    else:
        report["norm_opt_fp32"] = 0.0
    return report


def _get_auto_wrap_policy(model, fsdp_cfg):
    """Build an FSDP auto_wrap_policy close to rlinf behavior.

    Only wraps transformer decoder layers (from ``_no_split_modules``) and
    an optional ``value_head``.  No LoRA leaf wrap — ``use_orig_params=True``
    already handles mixed ``requires_grad`` correctly with ``no_shard``.
    """
    wrap_cfg = fsdp_cfg.get("wrap_policy", {})
    if wrap_cfg.get("disable", False):
        logger.info("FSDP auto-wrap disabled by config")
        return None

    # OpenVLAOFTForRLActionPrediction wraps the HF model at ``.model``
    wrap_root = model
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        wrap_root = model.model

    cls_names = wrap_cfg.get("transformer_layer_cls_to_wrap", None)
    if cls_names is None:
        if hasattr(wrap_root, "language_model"):
            cls_names = getattr(wrap_root.language_model, "_no_split_modules", None)
        else:
            cls_names = getattr(wrap_root, "_no_split_modules", None)

    if cls_names is None:
        logger.warning("No transformer layer class names found for auto-wrap")
        cls_names = []
    if isinstance(cls_names, str):
        cls_names = [cls_names]

    policies = []
    transformer_cls_to_wrap = set()
    for layer_name in cls_names:
        layer_cls = get_module_class_from_name(wrap_root, layer_name)
        if layer_cls is None:
            logger.warning("Auto-wrap class not found in model: %s", layer_name)
            continue
        transformer_cls_to_wrap.add(layer_cls)

    if transformer_cls_to_wrap:
        policies.append(
            functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls=transformer_cls_to_wrap,
            )
        )

    if getattr(model, "value_head", None) is not None:
        policies.append(
            functools.partial(
                _module_wrap_policy,
                module_classes={model.value_head.__class__},
            )
        )

    if len(policies) == 0:
        logger.warning("FSDP auto-wrap fallback: no wrap policies built")
        return None
    if len(policies) == 1:
        return policies[0]
    return functools.partial(_or_policy, policies=policies)


def setup_fsdp(model, fsdp_cfg, process_group, device):
    sharding = SHARDING_MAP.get(
        fsdp_cfg.get("sharding_strategy", "no_shard"),
        ShardingStrategy.NO_SHARD,
    )
    mp = _get_mixed_precision(fsdp_cfg)
    model = model.to(device)
    auto_wrap_policy = _get_auto_wrap_policy(model, fsdp_cfg)
    logger.info("FSDP auto-wrap policy enabled=%s", auto_wrap_policy is not None)

    wrapped = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=sharding,
        mixed_precision=mp,
        process_group=process_group,
        device_id=device,
        use_orig_params=fsdp_cfg.get("use_orig_params", False),
        limit_all_gathers=fsdp_cfg.get("limit_all_gathers", False),
        forward_prefetch=fsdp_cfg.get("forward_prefetch", False),
    )
    logger.info("FSDP wrapped: sharding=%s mp=%s", sharding, mp is not None)
    return wrapped


def setup_optimizer(model, optim_cfg):
    trainable = [p for p in model.parameters() if p.requires_grad]
    total = sum(p.numel() for p in model.parameters())
    train_numel = sum(p.numel() for p in trainable)
    logger.info(
        "Optimizer: %d trainable param tensors (%.1fM / %.1fM total). "
        "Optimizer will use ~%.1f GB for states.",
        len(trainable), train_numel / 1e6, total / 1e6,
        train_numel * 4 * 2 / 1e9,
    )
    param_dtypes = {p.dtype for p in trainable}
    use_foreach = len(param_dtypes) == 1 and torch.float32 in param_dtypes
    logger.info(
        "Optimizer foreach=%s (param dtypes: %s)",
        use_foreach, sorted(str(d) for d in param_dtypes),
    )
    # PPO critic head gets its own lr (optim.value_lr); group 0 stays the
    # policy group so lr logging (param_groups[0]) is unchanged.
    value_lr = optim_cfg.get("value_lr", None)
    value_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and "value_head" in n
    ]
    if value_lr is not None and value_params:
        policy_params = [
            p for n, p in model.named_parameters()
            if p.requires_grad and "value_head" not in n
        ]
        params = [
            {"params": policy_params},
            {"params": value_params, "lr": float(value_lr)},
        ]
        logger.info(
            "Optimizer param groups: policy=%d tensors lr=%s, value_head=%d "
            "tensors lr=%s",
            len(policy_params), optim_cfg.lr, len(value_params), value_lr,
        )
    else:
        params = trainable
    return torch.optim.AdamW(
        params,
        lr=optim_cfg.lr,
        betas=(optim_cfg.get("adam_beta1", 0.9), optim_cfg.get("adam_beta2", 0.95)),
        eps=optim_cfg.get("adam_eps", 1e-8),
        weight_decay=optim_cfg.get("weight_decay", 0.0),
        foreach=use_foreach,
    )


def setup_scheduler(optimizer, optim_cfg) -> torch.optim.lr_scheduler.LambdaLR:
    """LR scheduler; default ``constant`` (matches RLinf embodied PPO/GRPO)."""
    lr_scheduler = str(optim_cfg.get("lr_scheduler", "constant")).lower()
    warmup_steps = int(optim_cfg.get("lr_warmup_steps", 0))

    if lr_scheduler != "constant":
        raise ValueError(
            f"Unsupported lr_scheduler={lr_scheduler!r}; only 'constant' is supported"
        )

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step) / float(max(warmup_steps, 1))
        return 1.0

    sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    logger.info(
        "LR scheduler: constant (warmup_steps=%d, base_lr=%s)",
        warmup_steps,
        [g.get("lr") for g in optimizer.param_groups],
    )
    return sched


def _clear_memory(sync: bool = True):
    """Release GPU cached memory."""
    if sync:
        torch.cuda.synchronize()
    torch.cuda.empty_cache()


def _iter_fsdp_handles(model: FSDP):
    """Yield non-None FSDP handles for all nested FSDP modules."""
    for fsdp_mod in FSDP.fsdp_modules(model):
        handle = getattr(fsdp_mod, "_handle", None)
        if handle is not None:
            yield handle


def _refresh_handle_views(handle) -> None:
    """Re-sync orig-param views after manual flat_param device moves."""
    if handle._use_orig_params:
        handle._use_sharded_views()


def _refresh_orig_param_views(model: FSDP) -> None:
    for handle in _iter_fsdp_handles(model):
        _refresh_handle_views(handle)


@torch.no_grad()
def offload_params_and_grads(model: FSDP, offload_grad: bool = True) -> None:
    """Move FSDP model parameters (and optionally gradients) to CPU.

    With ``use_orig_params=True``, only move flat params then refresh views.
    Assigning ``param.data`` directly breaks FSDP writeback on the next forward.
    Callers should skip this path when ``use_orig_params=True``
    (see ``ActorWorker._skip_cpu_park``).
    """
    seen_flat: set[int] = set()
    for handle in _iter_fsdp_handles(model):
        fp = handle.flat_param
        fp_id = id(fp)
        if fp_id in seen_flat:
            continue
        seen_flat.add(fp_id)

        if hasattr(fp, "_local_shard") and fp._local_shard is not None:
            fp._local_shard = fp._local_shard.to("cpu", non_blocking=True)
        if fp.data is not None:
            fp.data = fp.data.to("cpu", non_blocking=True)
        if hasattr(fp, "_local_shard") and fp.data is not None:
            fp._local_shard = fp.data
        if offload_grad and fp.grad is not None:
            fp.grad = fp.grad.to("cpu", non_blocking=True)
        _refresh_handle_views(handle)

    for _, param in model.named_parameters():
        if param.data is not None:
            param.data = param.data.to("cpu", non_blocking=True)
        if offload_grad and param.grad is not None:
            param.grad = param.grad.to("cpu", non_blocking=True)
    _clear_memory()
    logger.debug("Offloaded model params (and grads=%s) to CPU", offload_grad)


@torch.no_grad()
def onload_params_and_grads(
    model: FSDP, device: torch.device, onload_grad: bool = False,
) -> None:
    """Reload FSDP model parameters (and optionally gradients) to *device*."""
    seen_flat: set[int] = set()
    for handle in _iter_fsdp_handles(model):
        fp = handle.flat_param
        fp_id = id(fp)
        if fp_id in seen_flat:
            continue
        seen_flat.add(fp_id)

        if hasattr(fp, "_local_shard") and fp._local_shard is not None:
            fp._local_shard = fp._local_shard.to(device, non_blocking=True)
        if fp.data is not None:
            fp.data = fp.data.to(device, non_blocking=True)
        if hasattr(fp, "_local_shard") and fp.data is not None:
            fp._local_shard = fp.data
        if onload_grad and fp.grad is not None:
            fp.grad = fp.grad.to(device, non_blocking=True)
        _refresh_handle_views(handle)

    for _, param in model.named_parameters():
        if param.data is not None:
            param.data = param.data.to(device, non_blocking=True)
        if onload_grad and param.grad is not None:
            param.grad = param.grad.to(device, non_blocking=True)
    _clear_memory()
    logger.debug("Onloaded model params to %s", device)


@torch.no_grad()
def offload_optimizer(optimizer: torch.optim.Optimizer) -> None:
    """Move all optimizer state tensors to CPU."""
    if not optimizer.state:
        return
    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            state = optimizer.state.get(param)
            if state is None:
                continue
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to("cpu", non_blocking=True)
    _clear_memory()
    logger.debug("Offloaded optimizer states to CPU")


@torch.no_grad()
def onload_optimizer(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    """Reload all optimizer state tensors to *device*."""
    if not optimizer.state:
        return
    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            state = optimizer.state.get(param)
            if state is None:
                continue
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device, non_blocking=True)
    _clear_memory()
    logger.debug("Onloaded optimizer states to %s", device)
