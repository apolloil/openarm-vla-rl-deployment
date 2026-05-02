"""Train PPO agent with 4D EE action space on the Soccer task.

The policy outputs [Δx, Δy, Δz, grip] (4-dim continuous); rotation deltas
to the DiffIK backend are always zero, but the gripper is policy-controlled
so the agent can open/close as needed (consistent with MetaWorld soccer demos).
The initial downward grasp posture is baked into ``init_state.joint_pos`` of
the VLA Soccer env, so every reset (manual + Isaac Lab auto-reset) starts
from the same wrist pose used by Lift VLA.

Differences from ``train_lift.py``:
  1. Default ``--task`` is ``Isaac-VLA-Soccer-OpenArm-v0``.
  2. Uses ``Push3DEEActionWrapper`` (4-dim [dx,dy,dz,grip], zero rotation).
  3. ``EulerEEActionWrapper`` is not imported.

Usage::

    # Default: soccer task, 4096 envs
    conda activate env_isaaclab
    cd /home/lcw/workspace/openarm
    python openarm_vla/Data_Collector/train_soccer.py
  

    # Custom
    python openarm_vla/Data_Collector/train_soccer.py \\
        --task Isaac-VLA-Soccer-OpenArm-v0 \\
        --num_envs 2048 --max_iterations 4000 --seed 42

    # With video recording
    python openarm_vla/Data_Collector/train_soccer.py --video --video_interval 500
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

parser = argparse.ArgumentParser(
    description="Train PPO agent with push-only 3D EE action space via RSL-RL."
)
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument(
    "--task", type=str, default="Isaac-VLA-Soccer-OpenArm-v0",
    help="Gym task ID (default: Isaac-VLA-Soccer-OpenArm-v0).",
)
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point",
    help="Name of the RL agent configuration entry point.",
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False,
    help="Run training with multiple GPUs or nodes.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import openarm.tasks  # noqa: F401
import openarm_vla.tasks  # noqa: F401  — registers Isaac-VLA-* Gym IDs

from soccer_ee_action_wrapper import Push3DEEActionWrapper
from policy_action_vecenv_wrapper import PolicyActionVecEnvWrapper

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

OPENARM_REPO_ROOT = "/home/lcw/workspace/openarm"


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train PPO with RSL-RL on push-only 3D EE action space."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # State-based RL does not need the camera sensor that the VLA env ships with.
    # Removing it avoids --enable_cameras overhead (shader compilation, extra VRAM).
    if hasattr(env_cfg.scene, "scene_rgb_cam"):
        env_cfg.scene.scene_rgb_cam = None

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # logging directories
    log_root_path = os.path.join(OPENARM_REPO_ROOT, "logs", "rsl_rl", agent_cfg.experiment_name)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = False
    env_cfg.log_dir = log_dir

    # ── Create environment ──────────────────────────────────────────────────
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # Expose the 3D push-only policy interface: [dx, dy, dz], gripper always closed.
    env = Push3DEEActionWrapper(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # ── RSL-RL wrapper (must be last) ───────────────────────────────────────
    env = PolicyActionVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # ── Create runner ───────────────────────────────────────────────────────
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)

    if agent_cfg.resume:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # ── Train ───────────────────────────────────────────────────────────────
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
