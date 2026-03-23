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
# VLA (DiffIK) variants — Unimanual Lift-Cube
##

import gymnasium as gym

# Agent configs live in the base openarm package
_AGENTS = "openarm.tasks.manager_based.openarm_manipulation.unimanual.lift.config.agents"

gym.register(
    id="Isaac-VLA-Lift-Cube-OpenArm-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.vla_env_cfg:OpenArmLiftVlaEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_AGENTS}.rsl_rl_ppo_cfg:OpenArmLiftCubePPORunnerCfg",
        "skrl_cfg_entry_point": f"{_AGENTS}:skrl_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{_AGENTS}:rl_games_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-VLA-Lift-Cube-OpenArm-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.vla_env_cfg:OpenArmLiftVlaEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{_AGENTS}.rsl_rl_ppo_cfg:OpenArmLiftCubePPORunnerCfg",
        "skrl_cfg_entry_point": f"{_AGENTS}:skrl_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{_AGENTS}:rl_games_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)
