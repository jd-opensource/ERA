"""Minimal 2-node NCCL connectivity test."""
import torch
import torch.distributed as dist

dist.init_process_group("nccl")
rank = dist.get_rank()
world = dist.get_world_size()
t = torch.ones(1, device="cuda")
dist.all_reduce(t)
print(f"rank {rank}/{world}  all_reduce sum = {t.item()}  PASS" if t.item() == world else f"rank {rank} FAIL got {t.item()}")
dist.destroy_process_group()

