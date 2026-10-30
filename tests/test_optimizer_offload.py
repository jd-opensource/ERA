"""Tests for optimizer-only CPU offload helpers."""

from __future__ import annotations

import torch

from joyrlinf.training.fsdp_trainer import offload_optimizer, onload_optimizer


def _state_tensors(opt: torch.optim.Optimizer):
    for group in opt.param_groups:
        for p in group["params"]:
            st = opt.state.get(p)
            if not st:
                continue
            for key, value in st.items():
                if key == "step":
                    continue
                if isinstance(value, torch.Tensor) and value.is_floating_point():
                    yield key, value


def test_offload_onload_optimizer_roundtrip():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    param = torch.nn.Parameter(torch.randn(8, 8, device=device))
    opt = torch.optim.AdamW([param], lr=1e-3)
    (param ** 2).sum().backward()
    opt.step()
    opt.zero_grad(set_to_none=True)

    for _, v in _state_tensors(opt):
        assert v.device.type == device.type

    offload_optimizer(opt)
    for _, v in _state_tensors(opt):
        assert v.device.type == "cpu"

    onload_optimizer(opt, device)
    for _, v in _state_tensors(opt):
        assert v.device.type == device.type
