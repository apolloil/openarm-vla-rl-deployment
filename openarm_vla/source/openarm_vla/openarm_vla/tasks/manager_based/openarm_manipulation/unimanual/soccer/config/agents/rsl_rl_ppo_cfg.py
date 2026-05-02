# Copyright 2025 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Local PPO config for the VLA Soccer task.

Subclasses the base ``OpenArmSoccerPPORunnerCfg`` and bumps two
hyperparameters to mirror the Lift-VLA recipe:

* ``max_iterations``: 2000 -> 4000
* ``algorithm.entropy_coef``: 0.006 -> 0.012
"""

from __future__ import annotations

from isaaclab.utils import configclass

from openarm.tasks.manager_based.openarm_manipulation.unimanual.soccer.config.agents.rsl_rl_ppo_cfg import (
    OpenArmSoccerPPORunnerCfg as _BasePPORunnerCfg,
)


@configclass
class OpenArmSoccerVlaPPORunnerCfg(_BasePPORunnerCfg):
    max_iterations = 4000

    def __post_init__(self):
        self.algorithm.entropy_coef = 0.012
