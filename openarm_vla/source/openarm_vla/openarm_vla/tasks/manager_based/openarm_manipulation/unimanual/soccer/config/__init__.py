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

##
# VLA (DiffIK) variants — Unimanual Soccer
##

import gymnasium as gym

_LOCAL_AGENTS = f"{__name__}.agents"

gym.register(
    id="Isaac-VLA-Soccer-OpenArm-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.vla_env_cfg:OpenArmSoccerVlaEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_LOCAL_AGENTS}.rsl_rl_ppo_cfg:OpenArmSoccerVlaPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-VLA-Soccer-OpenArm-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.vla_env_cfg:OpenArmSoccerVlaEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{_LOCAL_AGENTS}.rsl_rl_ppo_cfg:OpenArmSoccerVlaPPORunnerCfg",
    },
    disable_env_checker=True,
)
