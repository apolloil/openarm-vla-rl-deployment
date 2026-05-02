"""Collect an expert VLA dataset using the trained 4-D PPO policy.

Rolls the PPO expert in the VLA ``Isaac-VLA-Lift-Cube-OpenArm-Play-v0`` env,
captures RGB frames + EE/finger state + raw 4-D action, and writes a
LeRobotDataset (v0.5.1) ready for SmolVLA / π0.5 fine-tuning.

Highlights:

- Goal is **frozen per-episode** (``resampling_time_range=(1e6, 1e6)``) and
  visualised by a violet sphere (``target_marker``) so the VLA sees the target
  location in the RGB frame itself — no extra numeric channel required.
- **Early stop**: as soon as the cube sits within ``SUCCESS_DIST_THRESHOLD`` of
  the goal (and above ``MIN_OBJECT_Z_FOR_SUCCESS``) for ``STABLE_STEPS`` in a
  row, we append ``TAIL_BUFFER_STEPS`` more frames then stop appending for
  that env. Cuts the long static tail of typical reach-and-hold rollouts.
- **Failed-episode discard**: only episodes that tripped the success
  criterion before ``EPISODE_MAX_STEPS`` are flushed to disk. Everything else
  (drop, stall, time-out without reach) is thrown away outright.
- Uses ``num_envs=1`` because ``scene_rgb_cam`` lives at ``/World/`` (global
  prim). Multi-env collection would need a per-``{ENV_REGEX_NS}`` camera
  clone, which is a separate change.

    conda activate env_isaaclab
    cd /home/lcw/workspace/openarm
    python openarm_vla/Data_Collector/collect_dataset.py
"""

from __future__ import annotations

# =============================================================================
# Collection configuration — edit here or override via env vars.
# =============================================================================
CHECKPOINT_PATH = "/home/lcw/workspace/openarm/logs/rsl_rl/openarm_lift/2026-04-21_16-57-06/model_3999.pt"

COLLECT_TASK = "Isaac-VLA-Lift-Cube-OpenArm-Play-v0"
RSL_RL_AGENT_ENTRY = "rsl_rl_cfg_entry_point"
COLLECT_DEVICE: str | None = None  # None → hydra default (cuda)
COLLECT_GUI = False

# Output dataset root (parent dir will be created). One flat dataset per run.
DATASET_ROOT = "/home/lcw/workspace/openarm/openarm_vla/Datasets/lift_cube_expert_v0"
DATASET_REPO_ID = "openarm/lift_cube_expert_v0"
ROBOT_TYPE = "openarm_unimanual"

# Scene preset (0..9, see play_scene_presets.py). Set via env OPENARM_COLLECT_SCENE_PRESET.
COLLECT_SCENE_PRESET_ID = 0

# How many *successful* episodes to append under this invocation. The shell
# driver typically cycles through presets, calling this script once per preset
# with e.g. 20 successes, so the aggregated dataset ends up with ~10×20=200.
# Overridden by env OPENARM_COLLECT_NUM_SUCCESS.
NUM_SUCCESSFUL_EPISODES = 20

EPISODE_MAX_STEPS = 440              # 9 s / 0.02 s  −  10 step cushion before auto-reset.
STABLE_STEPS = 5
SUCCESS_DIST_THRESHOLD = 0.05        # m, object-to-goal
MIN_OBJECT_Z_FOR_SUCCESS = 0.06      # m, world frame (above tabletop)
TAIL_BUFFER_STEPS = 3

VIDEO_CODEC = "h264"                 # vs. LeRobot default "libsvtav1"
TASK_STRING = "pick up the red cube and move it to the violet marker"

OPENARM_REPO_ROOT = "/home/lcw/workspace/openarm"
# =============================================================================

import argparse
import os
import sys
import time

os.environ.setdefault("CARB_LOGGING_LEVEL", "error")

# Hydra-gated scripts must not see stray argv.
sys.argv = [sys.argv[0]]

# ── Env-var overrides (before AppLauncher) ────────────────────────────────
def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key, "").strip()
    return int(raw) if raw else default


COLLECT_SCENE_PRESET_ID = _env_int("OPENARM_COLLECT_SCENE_PRESET", COLLECT_SCENE_PRESET_ID)
NUM_SUCCESSFUL_EPISODES = _env_int("OPENARM_COLLECT_NUM_SUCCESS", NUM_SUCCESSFUL_EPISODES)
_ds_root_env = os.environ.get("OPENARM_COLLECT_DATASET_ROOT", "").strip()
if _ds_root_env:
    DATASET_ROOT = _ds_root_env
_ds_repo_env = os.environ.get("OPENARM_COLLECT_DATASET_REPO_ID", "").strip()
if _ds_repo_env:
    DATASET_REPO_ID = _ds_repo_env

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

_parser = argparse.ArgumentParser(description="Collect VLA dataset (config via globals / env vars).")
_parser.add_argument("--video", action="store_true", default=False, help=argparse.SUPPRESS)
_parser.add_argument("--video_length", type=int, default=500, help=argparse.SUPPRESS)
_parser.add_argument("--video_interval", type=int, default=2000, help=argparse.SUPPRESS)
_parser.add_argument("--num_envs", type=int, default=None, help=argparse.SUPPRESS)
_parser.add_argument("--task", type=str, default=COLLECT_TASK, help=argparse.SUPPRESS)
_parser.add_argument("--agent", type=str, default=RSL_RL_AGENT_ENTRY, help=argparse.SUPPRESS)
_parser.add_argument("--seed", type=int, default=None, help=argparse.SUPPRESS)
_parser.add_argument("--max_iterations", type=int, default=None, help=argparse.SUPPRESS)
_parser.add_argument("--distributed", action="store_true", default=False, help=argparse.SUPPRESS)
cli_args.add_rsl_rl_args(_parser)
AppLauncher.add_app_launcher_args(_parser)
args_cli = _parser.parse_args([])

args_cli.task = COLLECT_TASK
args_cli.agent = RSL_RL_AGENT_ENTRY
args_cli.video = True  # required to get scene_rgb_cam outputs with enable_cameras
args_cli.enable_cameras = True
if COLLECT_DEVICE is not None:
    args_cli.device = COLLECT_DEVICE
if COLLECT_GUI:
    os.environ["HEADLESS"] = "0"
    args_cli.headless = False
else:
    args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectRLEnvCfg,
    DirectMARLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.math import combine_frame_transforms
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import openarm.tasks  # noqa: F401
import openarm_vla.tasks  # noqa: F401

from lift_ee_action_wrapper import EulerEEActionWrapper
from play_scene_presets import NUM_PLAY_SCENE_PRESETS, apply_play_scene_materials
from policy_action_vecenv_wrapper import PolicyActionVecEnvWrapper

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from openarm.tasks.manager_based.openarm_manipulation.unimanual.lift.mdp import (
    ee_pose_in_robot_root_frame as _base_mdp_ee_pose,
)


# ── Dataset schema ─────────────────────────────────────────────────────────
def _build_features() -> dict:
    return {
        "observation.images.scene": {
            "dtype": "video",
            "shape": (720, 1280, 3),
            "names": ["height", "width", "channels"],
        },
        # [x, y, z, finger_width]; robot root frame; finger_width in meters.
        "observation.state.ee_pose": {
            "dtype": "float32",
            "shape": (4,),
            "names": {"axes": ["x", "y", "z", "finger_width"]},
        },
        # Goal position in WORLD frame (marker lives in world).
        "observation.state.goal": {
            "dtype": "float32",
            "shape": (3,),
            "names": {"axes": ["x", "y", "z"]},
        },
        # Raw policy output BEFORE binarize-to-±1 on the gripper logit.
        "action": {
            "dtype": "float32",
            "shape": (4,),
            "names": {"axes": ["dx", "dy", "dz", "grip_logit"]},
        },
        "is_episode_successful": {
            "dtype": "bool",
            "shape": (1,),
            "names": None,
        },
    }


def _rgb_to_uint8(rgb) -> np.ndarray:
    """Convert an rgb tensor/array of shape ``(H,W,C)`` / ``(1,H,W,C)`` / ``(N,H,W,C)`` to HWC uint8.

    Mirrors ``play_lift._frame_from_render`` semantics for the num_envs=1
    camera we use during collection.
    """
    if isinstance(rgb, torch.Tensor):
        arr = rgb.detach().cpu().numpy()
    else:
        arr = np.asarray(rgb)

    if arr.ndim == 4:
        arr = arr[0]

    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return arr


def _find_single_joint_index(robot, expr: str) -> int:
    joint_ids, _ = robot.find_joints(expr)
    if not joint_ids:
        raise RuntimeError(f"Robot has no joint matching '{expr}'.")
    return int(joint_ids[0])


_SCENE_SEED_MOD = 2**31


@hydra_task_config(COLLECT_TASK, RSL_RL_AGENT_ENTRY)
def main(
    env_cfg: "ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg",
    agent_cfg: RslRlBaseRunnerCfg,
):
    # ── Seed: mix PID + time_ns so parallel preset shards don't collide. ────
    scene_seed = (int(time.time_ns()) + os.getpid()) % _SCENE_SEED_MOD
    print(f"[INFO] scene_seed={scene_seed}")

    rsl_ns = argparse.Namespace(
        seed=scene_seed,
        resume=False,
        load_run=None,
        checkpoint=None,
        run_name=None,
        logger=None,
        log_project_name=None,
    )
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, rsl_ns)

    # num_envs=1: global /World camera = single viewport. Multi-env rollout
    # would need a per-env scene_rgb_cam clone (deferred; see module docstring).
    env_cfg.scene.num_envs = 1
    env_cfg.seed = agent_cfg.seed

    if COLLECT_DEVICE is not None:
        env_cfg.sim.device = COLLECT_DEVICE
        agent_cfg.device = COLLECT_DEVICE

    log_root_path = os.path.join(OPENARM_REPO_ROOT, "logs", "rsl_rl", agent_cfg.experiment_name)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    resume_path = retrieve_file_path(CHECKPOINT_PATH)
    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    if COLLECT_SCENE_PRESET_ID < 0 or COLLECT_SCENE_PRESET_ID >= NUM_PLAY_SCENE_PRESETS:
        raise ValueError(
            f"scene preset {COLLECT_SCENE_PRESET_ID} out of range "
            f"[0, {NUM_PLAY_SCENE_PRESETS - 1}]."
        )
    _preset = apply_play_scene_materials(env_cfg, COLLECT_SCENE_PRESET_ID)
    print(f"[INFO] Scene preset {_preset.id} ({_preset.name}).")

    env = gym.make(COLLECT_TASK, cfg=env_cfg, render_mode="rgb_array")

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = EulerEEActionWrapper(env)
    env = PolicyActionVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # ── Load PPO actor ────────────────────────────────────────────────────
    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    normalizer = getattr(policy_nn, "actor_obs_normalizer", None)
    if normalizer is None:
        normalizer = getattr(policy_nn, "student_obs_normalizer", None)

    # Reuse play's export convention so a parallel eval + collect run share artifacts.
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    # ── Handles into the scene ─────────────────────────────────────────────
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    object_asset = unwrapped.scene["object"]
    finger_joint_idx = _find_single_joint_index(robot, "openarm_finger_joint1")

    step_dt = float(unwrapped.step_dt)
    fps = int(round(1.0 / step_dt))
    print(f"[INFO] env step_dt={step_dt:.4f}s → fps={fps}")

    # ── LeRobot dataset: one shard per invocation ──────────────────────────
    # LeRobot v0.5.1's ``create`` does ``mkdir(exist_ok=False)``, so each run
    # must target a fresh directory. We place shards under
    # ``DATASET_ROOT/preset<NN>_<name>_<timestamp>`` so the shell driver can
    # collect many shards side-by-side without clobbering each other, and
    # downstream VLA training can load + concatenate shards at load-time.
    features = _build_features()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    shard_root = (
        Path(DATASET_ROOT)
        / f"preset{COLLECT_SCENE_PRESET_ID:02d}_{_preset.name}_{timestamp}"
    )
    if shard_root.exists():
        raise RuntimeError(
            f"Shard root {shard_root} already exists. "
            "This should never happen (timestamp collision?)."
        )
    shard_root.parent.mkdir(parents=True, exist_ok=True)
    dataset = LeRobotDataset.create(
        repo_id=DATASET_REPO_ID,
        fps=fps,
        features=features,
        root=shard_root,
        robot_type=ROBOT_TYPE,
        use_videos=True,
        vcodec=VIDEO_CODEC,
    )
    print(f"[INFO] Writing shard to {shard_root} (vcodec={VIDEO_CODEC}).")

    # ── Camera warm-up ────────────────────────────────────────────────────
    # Isaac Lab's viewport cam only has populated content after at least one
    # physics sub-step. ``env.render()`` called straight after the first
    # ``env.reset()`` of a session returns an all-zero (black) frame even
    # though ``rerender_on_reset=True``. Subsequent episodes don't have this
    # problem because the buffer is inherited from the previous episode.
    # We do one throw-away reset + policy-step + render here so that when
    # the main loop starts and calls reset() again, the first real captured
    # frame of episode 0 is already a valid render.
    print("[INFO] Priming camera with a warm-up step (one reset + one policy action)...")
    _warmup_obs, _ = env.reset()
    with torch.inference_mode():
        _warmup_action = policy(_warmup_obs)
    env.step(_warmup_action)
    _ = unwrapped.render()
    print("[INFO] Warm-up complete; entering main collection loop.")

    # ── Rollout / capture loop ─────────────────────────────────────────────
    num_successful = 0
    num_failed = 0
    batch_idx = 0
    rollout_start = time.time()

    while num_successful < NUM_SUCCESSFUL_EPISODES:
        batch_idx += 1

        obs_td, _ = env.reset()
        obs = obs_td

        # Per-env buffers (num_envs=1 so this is trivial, but written as a
        # list so upgrading to multi-env later needs only a camera-clone patch).
        num_envs = unwrapped.num_envs
        eps_buffers: list[list[dict]] = [[] for _ in range(num_envs)]
        eps_success = [False] * num_envs
        eps_stable = [0] * num_envs
        eps_done = [False] * num_envs
        eps_tail_remaining = [0] * num_envs

        for step in range(EPISODE_MAX_STEPS):
            # --- Capture state(t) BEFORE stepping -------------------------
            # Camera data is the last-rendered frame (from the previous
            # physics step, or from scene init on step 0).
            # ``env.render()`` reads the viewport cam (=/World/scene_rgb_cam)
            # via the ManagerBasedRLEnv viewport API, which is populated on
            # ``reset`` (thanks to ``rerender_on_reset=True``) and after every
            # physics sub-step. It is robust even on step 0 where the raw
            # Camera sensor buffer may still be uninitialized.
            rgb_frame = _rgb_to_uint8(unwrapped.render())

            # We intentionally skip ``observation_manager.compute_group`` and
            # call the helper directly so we get the unflattened 7D EE pose
            # (policy obs are concatenated into a single 1D vector).
            ee_pose_7d = _base_mdp_ee_pose(unwrapped)  # (num_envs, 7)
            ee_xyz = ee_pose_7d[:, :3]

            finger_width = robot.data.joint_pos[:, finger_joint_idx : finger_joint_idx + 1]

            cmd = unwrapped.command_manager.get_command("object_pose")  # (num_envs, ≥3) root frame
            goal_w, _ = combine_frame_transforms(
                robot.data.root_pos_w, robot.data.root_quat_w, cmd[:, :3]
            )

            obj_pos_w = object_asset.data.root_pos_w  # (num_envs, 3)

            # --- Compute action(t) ---------------------------------------
            with torch.inference_mode():
                action_raw = policy(obs)  # (num_envs, 4)

            # --- Buffer frame(t) for live envs ---------------------------
            for i in range(num_envs):
                if eps_done[i]:
                    continue

                eps_buffers[i].append(
                    {
                        "observation.images.scene": rgb_frame,
                        "observation.state.ee_pose": np.concatenate(
                            [
                                ee_xyz[i].detach().cpu().numpy().astype(np.float32),
                                finger_width[i].detach().cpu().numpy().astype(np.float32),
                            ]
                        ),
                        "observation.state.goal": goal_w[i].detach().cpu().numpy().astype(np.float32),
                        "action": action_raw[i].detach().cpu().numpy().astype(np.float32),
                    }
                )

            # --- Step env -----------------------------------------------
            obs_td, _, dones, _ = env.step(action_raw)
            obs = obs_td

            # --- Per-env success logic -----------------------------------
            for i in range(num_envs):
                if eps_done[i]:
                    continue

                dist = float(torch.linalg.vector_norm(obj_pos_w[i] - goal_w[i]))
                obj_z = float(obj_pos_w[i, 2])
                stable_cond = dist < SUCCESS_DIST_THRESHOLD and obj_z > MIN_OBJECT_Z_FOR_SUCCESS

                if stable_cond:
                    eps_stable[i] += 1
                else:
                    eps_stable[i] = 0

                if not eps_success[i] and eps_stable[i] >= STABLE_STEPS:
                    eps_success[i] = True
                    eps_tail_remaining[i] = TAIL_BUFFER_STEPS
                elif eps_success[i]:
                    eps_tail_remaining[i] -= 1
                    if eps_tail_remaining[i] <= 0:
                        eps_done[i] = True

                if bool(dones[i]):
                    # Env auto-reset inside env.step (should be rare at
                    # EPISODE_MAX_STEPS=440 but can fire if the cube falls).
                    eps_done[i] = True

            if all(eps_done):
                break

        # --- Flush successful episodes; discard the rest ------------------
        for i in range(num_envs):
            buf = eps_buffers[i]
            if eps_success[i] and len(buf) > 0:
                for frame in buf:
                    frame_out = dict(frame)
                    frame_out["is_episode_successful"] = np.array([True])
                    frame_out["task"] = TASK_STRING
                    dataset.add_frame(frame_out)
                dataset.save_episode()
                num_successful += 1
                print(
                    f"[INFO] batch {batch_idx} env{i}: saved episode "
                    f"({len(buf)} frames, ok={num_successful}/{NUM_SUCCESSFUL_EPISODES})"
                )
            else:
                num_failed += 1
                print(
                    f"[WARN] batch {batch_idx} env{i}: DISCARD "
                    f"({len(buf)} frames, success={eps_success[i]}, stable={eps_stable[i]}) "
                    f"[failed={num_failed}]"
                )

        if num_failed >= 20 and num_successful == 0:
            print(
                "[ERROR] 20 consecutive failures with zero successes; aborting "
                "before writing a useless dataset. Check checkpoint / env setup."
            )
            break

    dataset.finalize()

    elapsed = time.time() - rollout_start
    print(
        f"[INFO] Done. successful={num_successful} failed={num_failed} "
        f"batches={batch_idx} elapsed={elapsed:.1f}s"
    )
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
