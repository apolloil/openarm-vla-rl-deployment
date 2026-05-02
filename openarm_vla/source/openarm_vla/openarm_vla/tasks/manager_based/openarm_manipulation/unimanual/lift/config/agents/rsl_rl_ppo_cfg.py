"""Local PPO config for the VLA Lift-Cube task.

Subclasses the base ``OpenArmLiftCubePPORunnerCfg`` from ``openarm`` and
bumps two hyperparameters:

* ``max_iterations``: 2000 -> 4000 (let PPO converge more fully).
* ``algorithm.entropy_coef``: 0.006 -> 0.012 (keep some policy-level
  stochasticity so rollouts retain diversity when collecting a VLA dataset).

No environment-level randomization is added; all diversity comes from the
policy's own Gaussian.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from openarm.tasks.manager_based.openarm_manipulation.unimanual.lift.config.agents.rsl_rl_ppo_cfg import (
    OpenArmLiftCubePPORunnerCfg as _BasePPORunnerCfg,
)


@configclass
class OpenArmLiftVlaPPORunnerCfg(_BasePPORunnerCfg):
    max_iterations = 4000

    def __post_init__(self):
        self.algorithm.entropy_coef = 0.012
