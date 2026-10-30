# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Union

import numpy as np
import torch
import torch.nn as nn


def replace_dropout_with_identity(model):
    """Replace all dropout layers with nn.Identity()."""
    for name, module in list(model.named_modules()):
        if isinstance(
            module,
            (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout),
        ):
            if "." in name:
                parent_name = name.rsplit(".", 1)[0]
                child_name = name.rsplit(".", 1)[1]
                parent_module = dict(model.named_modules())[parent_name]
            else:
                parent_module = model
                child_name = name
            setattr(parent_module, child_name, nn.Identity())


def unsqueeze_dict_values(data: dict[str, Any]) -> dict[str, Any]:
    unsqueezed_data = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            unsqueezed_data[k] = np.expand_dims(v, axis=0)
        elif isinstance(v, list):
            unsqueezed_data[k] = np.expand_dims(np.array(v), axis=0)
        elif isinstance(v, torch.Tensor):
            unsqueezed_data[k] = v.unsqueeze(0)
        else:
            unsqueezed_data[k] = v
    return unsqueezed_data


def squeeze_dict_values(data: dict[str, Any]) -> dict[str, Any]:
    squeezed_data = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            squeezed_data[k] = np.squeeze(v, axis=0)
        elif isinstance(v, torch.Tensor):
            squeezed_data[k] = v.squeeze(0)
        else:
            squeezed_data[k] = v
    return squeezed_data
