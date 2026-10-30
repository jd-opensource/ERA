"""Shared env helpers (tensor conversion, batching)."""

from __future__ import annotations

from typing import Any, Union

import numpy as np
import torch


def to_tensor(
    array: Union[dict, torch.Tensor, np.ndarray, list, Any],
    device: str | torch.device = "cpu",
) -> Union[dict, torch.Tensor, list, None]:
    if array is None:
        return None
    if isinstance(array, dict):
        return {k: to_tensor(v, device=device) for k, v in array.items()}
    if isinstance(array, torch.Tensor):
        return array.to(device)
    if isinstance(array, np.ndarray):
        if array.dtype == object:
            return [to_tensor(x, device=device) for x in array]
        if array.dtype == np.uint16:
            array = array.astype(np.int32)
        elif array.dtype == np.uint32:
            array = array.astype(np.int64)
        ret = torch.tensor(array).to(device)
    else:
        if isinstance(array, list) and any(x is None for x in array):
            return [to_tensor(x, device=device) for x in array]
        if (
            isinstance(array, list)
            and len(array) > 0
            and isinstance(array[0], np.ndarray)
        ):
            array = np.array(array)
        if isinstance(array, np.ndarray) and array.dtype == object:
            return [to_tensor(x, device=device) for x in array]
        ret = torch.tensor(array, device=device)
    if ret.dtype == torch.float64:
        ret = ret.to(torch.float32)
    return ret


def list_of_dict_to_dict_of_list(
    list_of_dict: list[dict[str, Any]] | np.ndarray,
) -> dict[str, list[Any]]:
    if isinstance(list_of_dict, np.ndarray):
        list_of_dict = list_of_dict.tolist()
    if len(list_of_dict) == 0:
        return {}
    keys = list_of_dict[0].keys()
    output = {key: [] for key in keys}
    for data in list_of_dict:
        for key, item in data.items():
            output[key].append(item)
    return output
