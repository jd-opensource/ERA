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

from enum import Enum


class EmbodimentTag(Enum):
    GR1 = "gr1"
    OXE_DROID = "oxe_droid"
    AGIBOT_GENIE1 = "agibot_genie1"
    NEW_EMBODIMENT = "new_embodiment"
    LIBERO_FRANKA = "libero_franka"
    MANISKILL_WIDOWX = "maniskill_widowx"
    MANISKILL_PANDA = "maniskill_panda"
    ISAACLAB_FRANKA = "isaaclab_franka"


EMBODIMENT_TAG_MAPPING = {
    EmbodimentTag.NEW_EMBODIMENT.value: 31,
    EmbodimentTag.LIBERO_FRANKA.value: 31,
    EmbodimentTag.OXE_DROID.value: 17,
    EmbodimentTag.AGIBOT_GENIE1.value: 26,
    EmbodimentTag.GR1.value: 24,
    EmbodimentTag.MANISKILL_WIDOWX.value: 30,
    EmbodimentTag.MANISKILL_PANDA.value: 31,
    EmbodimentTag.ISAACLAB_FRANKA.value: 31,
}
