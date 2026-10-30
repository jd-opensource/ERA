import torch


def recursive_to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, list):
        return [recursive_to_device(elem, device) for elem in obj]
    elif isinstance(obj, tuple):
        return tuple(recursive_to_device(elem, device) for elem in obj)
    elif isinstance(obj, dict):
        return {k: recursive_to_device(v, device) for k, v in obj.items()}
    else:
        return obj