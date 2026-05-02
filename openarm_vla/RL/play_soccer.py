"""Play (evaluate) a trained PPO checkpoint for the 4D EE soccer policy.

Headless server workflow: always records an MP4. Edit the configuration block below
(no command-line arguments).

Loads a checkpoint trained by ``train_soccer.py``, runs the policy in the Play
variant of the VLA Soccer environment, exports JIT/ONNX next to the checkpoint, and
writes a timestamped video under ``<checkpoint_dir>/videos/play/``.

Differences from ``play_lift.py``:
  1. Default ``PLAY_TASK`` is ``Isaac-VLA-Soccer-OpenArm-Play-v0``.
  2. Uses ``Push3DEEActionWrapper`` (4-dim [dx,dy,dz,grip], zero rotation).
  3. ``EulerEEActionWrapper`` is not imported.

    conda activate env_isaaclab
    cd /home/lcw/workspace/openarm
    python openarm_vla/RL/play_soccer.py
"""

# =============================================================================
# Play configuration — edit here only (no CLI).
# =============================================================================
CHECKPOINT_PATH = "/home/lcw/workspace/openarm/logs/rsl_rl/openarm_soccer/2026-05-02_21-23-10/model_1200.pt"
# Evaluate one full training-length episode by default.  Soccer is aligned to
# MetaWorld's 500 decision-step horizon: 500 * Isaac step_dt(0.02 s) = 10 s.
PLAY_EPISODE_LENGTH_S = 10.0
VIDEO_SECONDS = PLAY_EPISODE_LENGTH_S

PLAY_TASK = "Isaac-VLA-Soccer-OpenArm-Play-v0"
RSL_RL_AGENT_ENTRY = "rsl_rl_cfg_entry_point"
# Mixed into the env seed as (PLAY_SEED + time_ns) % 2**31 so every run differs.
PLAY_SEED: int | None = None
# None → keep Hydra/env defaults for sim and agent device.
PLAY_DEVICE: str | None = None
# False → headless + off-screen camera (server). True → local GUI viewport.
PLAY_GUI = False
# Visual presets (see ``play_scene_presets.py``). Valid: 0 … 9.
PLAY_SCENE_PRESET_ID = 1
# If set (e.g. by batch script), MP4s go here instead of ``<run>/videos/play/``.
PLAY_VIDEO_OUTPUT_DIR: str | None = None

# Phase-switch tolerance: require the ball centre to align with Pre_Goal in XY.
# Z is intentionally ignored because the policy may lift/carry the ball.
PRE_GOAL_XY_AXIS_TOL = 0.025
PRE_GOAL_SPEED_XY_TOL = 0.05

# Temporary diagnostics for debugging fixed-orientation DiffIK behavior.
DEBUG_PLAY = False
DEBUG_PRINT_EVERY_STEPS = 10

OPENARM_REPO_ROOT = "/home/lcw/workspace/openarm"
# =============================================================================

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

os.environ.setdefault("CARB_LOGGING_LEVEL", "error")
os.environ.setdefault("OPENARM_PLAY_EPISODE_LENGTH_S", str(PLAY_EPISODE_LENGTH_S))

# Hydra must not see stray argv.
sys.argv = [sys.argv[0]]

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

_parser = argparse.ArgumentParser(description="Play PPO-Push (config via globals at top of file).")
_parser.add_argument("--video", action="store_true", default=False, help=argparse.SUPPRESS)
_parser.add_argument("--video_length", type=int, default=500, help=argparse.SUPPRESS)
_parser.add_argument("--video_interval", type=int, default=2000, help=argparse.SUPPRESS)
_parser.add_argument("--num_envs", type=int, default=None, help=argparse.SUPPRESS)
_parser.add_argument("--task", type=str, default=PLAY_TASK, help=argparse.SUPPRESS)
_parser.add_argument("--agent", type=str, default=RSL_RL_AGENT_ENTRY, help=argparse.SUPPRESS)
_parser.add_argument("--seed", type=int, default=None, help=argparse.SUPPRESS)
_parser.add_argument("--max_iterations", type=int, default=None, help=argparse.SUPPRESS)
_parser.add_argument(
    "--distributed", action="store_true", default=False, help=argparse.SUPPRESS,
)
cli_args.add_rsl_rl_args(_parser)
AppLauncher.add_app_launcher_args(_parser)
args_cli = _parser.parse_args([])

args_cli.task = PLAY_TASK
args_cli.agent = RSL_RL_AGENT_ENTRY
args_cli.video = True
args_cli.enable_cameras = True
if PLAY_DEVICE is not None:
    args_cli.device = PLAY_DEVICE
if PLAY_GUI:
    os.environ["HEADLESS"] = "0"
    args_cli.headless = False
else:
    args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import time
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from rsl_rl.runners import OnPolicyRunner

import isaaclab.sim as sim_utils
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import openarm.tasks  # noqa: F401
import openarm_vla.tasks  # noqa: F401

from soccer_ee_action_wrapper import Push3DEEActionWrapper
from play_scene_presets import NUM_PLAY_SCENE_PRESETS, LiftPlayScenePreset, get_play_scene_preset
from policy_action_vecenv_wrapper import PolicyActionVecEnvWrapper
from soccer_targets import compute_soccer_targets, update_soccer_target_markers
from openarm.tasks.manager_based.openarm_manipulation.assets.local_soccer import (
    SOCCER_BALL_RADIUS,
    SOCCER_FIELD_TOP_Z,
)


BALL_RELEASE_Z_THRESHOLD = 0.040
BALL_TOP_CLEARANCE = 0.015
ARM_LEAVE_BALL_Z = SOCCER_FIELD_TOP_Z + 2.0 * SOCCER_BALL_RADIUS + BALL_TOP_CLEARANCE
ARM_FIELD_TOUCH_Z = SOCCER_FIELD_TOP_Z
ARM_Z_AXIS_TOL = 0.006
ARM_PKICK_XY_TOL = 0.020
HEURISTIC_VERTICAL_SPEED = 6.0
HEURISTIC_LATERAL_SPEED = 0.8


def _apply_soccer_scene_materials(env_cfg: object, preset_index: int) -> LiftPlayScenePreset:
    """Soccer-specific variant of apply_play_scene_materials.

    Soccer scene has no ``object`` or ``table`` attributes.  We only recolour
    the procedural ``plane`` (ground) and the ``soccer_field`` cuboid so the
    visual preset still controls the floor colour.
    """
    preset = get_play_scene_preset(preset_index)
    scene = env_cfg.scene

    # Ground plane (always present as a CuboidCfg in soccer).
    if hasattr(scene, "plane") and scene.plane is not None:
        pl = scene.plane
        scene.plane = pl.replace(
            spawn=pl.spawn.replace(
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=preset.floor_rgb,
                    roughness=preset.floor_roughness,
                )
            )
        )

    # Soccer field cuboid (grass-coloured by default; override to preset table colour
    # so the field still looks distinct from the floor).
    if hasattr(scene, "soccer_field") and scene.soccer_field is not None:
        fld = scene.soccer_field
        scene.soccer_field = fld.replace(
            spawn=fld.spawn.replace(
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=preset.table_rgb,
                    roughness=preset.table_roughness,
                )
            )
        )

    return preset


def _frame_from_render(env) -> np.ndarray:
    """Convert env.render() output to a single HxWx3 uint8 RGB frame."""
    rgb = env.render()
    if rgb is None:
        raise RuntimeError("env.render() returned None; check render_mode='rgb_array' and enable_cameras.")

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


def _write_mp4(frames: list[np.ndarray], path: Path, fps: float) -> None:
    """Write RGB frames to an MP4, preferring imageio and falling back to OpenCV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio

        imageio.mimsave(str(path), frames, fps=fps, codec="libx264")
    except Exception:
        import cv2

        height, width = frames[0].shape[:2]
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(fps),
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"cv2.VideoWriter could not open {path}")

        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()


def _default_video_path(
    log_dir: str,
    resume_path: str,
    *,
    output_dir_override: str | None = None,
    preset_index: int | None = None,
    preset_name: str | None = None,
) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    checkpoint_name = Path(resume_path).stem
    stem = f"{checkpoint_name}_{timestamp}"
    if preset_index is not None and preset_name:
        stem = f"preset{preset_index:02d}_{preset_name}_{stem}"
    if output_dir_override:
        out = Path(output_dir_override)
        out.mkdir(parents=True, exist_ok=True)
        return out / f"{stem}.mp4"
    return Path(log_dir) / "videos" / "play" / f"{stem}.mp4"


def _resolve_video_steps(video_seconds: float, step_dt: float) -> int:
    return max(1, int(round(float(video_seconds) / float(step_dt))))


def _get_ball_state(env) -> tuple[np.ndarray, np.ndarray]:
    ball = env.scene["soccer_ball"]
    pos = ball.data.root_pos_w[0, :3].detach().cpu().numpy()
    vel = ball.data.root_lin_vel_w[0, :3].detach().cpu().numpy()
    return pos, vel


def _get_ee_pose(env) -> tuple[np.ndarray, np.ndarray]:
    ee_frame = env.scene["ee_frame"]
    pos = ee_frame.data.target_pos_w[0, 0, :3].detach().cpu().numpy()
    quat = ee_frame.data.target_quat_w[0, 0, :4].detach().cpu().numpy()
    return pos, quat


def _quat_angle_deg(q0: np.ndarray, q1: np.ndarray) -> float:
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    q0 = q0 / max(float(np.linalg.norm(q0)), 1e-9)
    q1 = q1 / max(float(np.linalg.norm(q1)), 1e-9)
    # q and -q are the same orientation.
    dot = abs(float(np.dot(q0, q1)))
    return float(2.0 * np.degrees(np.arccos(np.clip(dot, -1.0, 1.0))))


def _heuristic_move_action(env, target_w: np.ndarray, *, gain: float = 6.0, limit: float = 1.0, grip: float = 0.0) -> np.ndarray:
    ee_pos = env.scene["ee_frame"].data.target_pos_w[0, 0, :3].detach().cpu().numpy()
    delta = (np.asarray(target_w, dtype=np.float64) - ee_pos) * float(gain)
    action = np.zeros(4, dtype=np.float32)
    action[:3] = np.clip(delta, -float(limit), float(limit)).astype(np.float32)
    action[3] = float(grip)
    return action


def _heuristic_kick_action(goal_dir: np.ndarray, *, speed: float = 1.0) -> np.ndarray:
    action = np.zeros(4, dtype=np.float32)
    action[:2] = np.asarray(goal_dir, dtype=np.float32) * float(speed)
    return action


def _vertical_action(dz: float, *, grip: float, speed: float = HEURISTIC_VERTICAL_SPEED) -> np.ndarray:
    action = np.zeros(4, dtype=np.float32)
    action[2] = float(np.clip(dz * speed, -1.0, 1.0))
    action[3] = float(grip)
    return action


def _debug_play_state(
    env,
    *,
    step_idx: int,
    phase: str,
    action_4d: torch.Tensor,
    init_ee_quat_w: np.ndarray | None,
    prev_ee_pos_w: np.ndarray | None,
    dt: float,
    ball_pos: np.ndarray,
    ball_to_pre_goal_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    ee_pos_w, ee_quat_w = _get_ee_pose(env)
    if init_ee_quat_w is None:
        init_ee_quat_w = ee_quat_w.copy()
    ee_speed = 0.0
    if prev_ee_pos_w is not None:
        ee_speed = float(np.linalg.norm(ee_pos_w - prev_ee_pos_w) / max(float(dt), 1e-9))

    action_4d_np = action_4d.detach().cpu().numpy()[0]
    manager_action_np = env.action_manager.action[0].detach().cpu().numpy()
    applied_4d = np.asarray(
        [manager_action_np[0], manager_action_np[1], manager_action_np[2], manager_action_np[-1]],
        dtype=np.float32,
    )
    rot_cmd_norm = float(np.linalg.norm(manager_action_np[3:6]))
    quat_drift_deg = _quat_angle_deg(init_ee_quat_w, ee_quat_w)

    print(
        "[DEBUG_PLAY] "
        f"step={step_idx:04d} phase={phase} "
        f"raw4=({action_4d_np[0]:+.3f},{action_4d_np[1]:+.3f},{action_4d_np[2]:+.3f},{action_4d_np[3]:+.3f}) "
        f"applied4=({applied_4d[0]:+.3f},{applied_4d[1]:+.3f},{applied_4d[2]:+.3f},{applied_4d[3]:+.3f}) "
        f"mgr7_rot=({manager_action_np[3]:+.3f},{manager_action_np[4]:+.3f},{manager_action_np[5]:+.3f}) "
        f"rot_norm={rot_cmd_norm:.4f} ee_quat_drift={quat_drift_deg:.2f}deg "
        f"ee_pos=({ee_pos_w[0]:+.3f},{ee_pos_w[1]:+.3f},{ee_pos_w[2]:+.3f}) ee_speed={ee_speed:.3f} "
        f"ball=({ball_pos[0]:+.3f},{ball_pos[1]:+.3f},{ball_pos[2]:+.3f}) "
        f"pre_xy_err=({ball_to_pre_goal_xy[0]:.3f},{ball_to_pre_goal_xy[1]:.3f})"
    )
    return ee_pos_w, init_ee_quat_w


_SCENE_SEED_MOD = 2**31


@hydra_task_config(PLAY_TASK, RSL_RL_AGENT_ENTRY)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with PPO-Push agent; always record video to disk."""
    _time_part = int(time.time_ns()) % _SCENE_SEED_MOD
    _base = 0 if PLAY_SEED is None else int(PLAY_SEED) % _SCENE_SEED_MOD
    scene_seed = (_base + _time_part) % _SCENE_SEED_MOD
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
    env_cfg.scene.num_envs = 1
    env_cfg.seed = agent_cfg.seed

    if PLAY_DEVICE is not None:
        env_cfg.sim.device = PLAY_DEVICE
        agent_cfg.device = PLAY_DEVICE

    if hasattr(env_cfg.terminations, "ball_reached_pre_goal"):
        env_cfg.terminations.ball_reached_pre_goal = None
        print("[INFO] Disabled training Pre_Goal termination for play heuristic phase switching.")

    log_root_path = os.path.join(OPENARM_REPO_ROOT, "logs", "rsl_rl", agent_cfg.experiment_name)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    resume_path = retrieve_file_path(CHECKPOINT_PATH)
    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    _preset_raw = os.environ.get("OPENARM_PLAY_SCENE_PRESET")
    if _preset_raw is not None and str(_preset_raw).strip() != "":
        preset_id = int(str(_preset_raw).strip())
    else:
        preset_id = PLAY_SCENE_PRESET_ID

    _preset = _apply_soccer_scene_materials(env_cfg, preset_id)
    print(
        f"[INFO] Play scene preset {_preset.id} ({_preset.name}); "
        f"valid ids: 0–{NUM_PLAY_SCENE_PRESETS - 1} (see play_scene_presets.py)."
    )

    env = gym.make(PLAY_TASK, cfg=env_cfg, render_mode="rgb_array")

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = Push3DEEActionWrapper(env)

    video_frames: list[np.ndarray] = []
    _vdir = os.environ.get("OPENARM_PLAY_VIDEO_DIR", "").strip()
    if not _vdir and PLAY_VIDEO_OUTPUT_DIR:
        _vdir = str(PLAY_VIDEO_OUTPUT_DIR).strip()
    video_path = _default_video_path(
        log_dir,
        resume_path,
        output_dir_override=_vdir if _vdir else None,
        preset_index=preset_id,
        preset_name=_preset.name,
    )
    print(f"[INFO] Recording play video to: {video_path}")
    print(f"[INFO] Requested video duration: {VIDEO_SECONDS:.2f} seconds")

    env = PolicyActionVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)

    policy = runner.get_inference_policy(device=env.unwrapped.device)

    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt
    video_steps = _resolve_video_steps(VIDEO_SECONDS, dt)
    video_fps = max(1.0, 1.0 / dt)

    obs = env.get_observations()
    update_soccer_target_markers(env.unwrapped)
    init_ee_quat_w: np.ndarray | None = None
    prev_debug_ee_pos_w: np.ndarray | None = None
    print(
        f"[INFO] Video target: {video_steps} steps at {video_fps:.2f} fps "
        f"(expected duration {video_steps / video_fps:.2f} s)"
    )
    phase = "policy_to_pregoal"
    stable_steps = 0
    stable_required_steps = max(1, int(round(0.1 / dt)))
    kick_steps = max(1, int(round(1.0 / dt)))
    phase_steps = 0

    for step_idx in range(video_steps):
        targets = compute_soccer_targets(env.unwrapped)
        ball_pos, ball_vel = _get_ball_state(env.unwrapped)
        ball_to_pre_goal_xy = np.abs(ball_pos[:2] - targets.pre_goal[:2])
        ball_speed_xy = float(np.linalg.norm(ball_vel[:2]))

        with torch.inference_mode():
            if phase == "policy_to_pregoal":
                actions = policy(obs)
                ball_xy_aligned = bool(np.all(ball_to_pre_goal_xy < float(PRE_GOAL_XY_AXIS_TOL)))
                ball_stable = ball_speed_xy < float(PRE_GOAL_SPEED_XY_TOL)
                if ball_xy_aligned and ball_stable:
                    stable_steps += 1
                else:
                    stable_steps = 0
                if stable_steps >= stable_required_steps:
                    phase = "lower_ball"
                    phase_steps = 0
                    print("Reached Pre_Goal, switch to kick phase")
                    print(f"[INFO] Switching to heuristic phase: {phase} at step {step_idx}")
            elif phase == "lower_ball":
                actions = torch.as_tensor(
                    _vertical_action(-1.0, grip=-1.0),
                    dtype=torch.float32,
                    device=env.unwrapped.device,
                ).unsqueeze(0)
                if ball_pos[2] <= float(BALL_RELEASE_Z_THRESHOLD):
                    phase = "release_and_lift"
                    phase_steps = 0
                    print(f"[INFO] Switching to heuristic phase: {phase} at step {step_idx}")
            elif phase == "release_and_lift":
                ee_z = float(env.unwrapped.scene["ee_frame"].data.target_pos_w[0, 0, 2].detach().cpu().item())
                actions = torch.as_tensor(
                    _vertical_action(float(ARM_LEAVE_BALL_Z) - ee_z, grip=1.0),
                    dtype=torch.float32,
                    device=env.unwrapped.device,
                ).unsqueeze(0)
                if ee_z >= float(ARM_LEAVE_BALL_Z) - float(ARM_Z_AXIS_TOL):
                    phase = "move_above_pkick"
                    phase_steps = 0
                    print("Arm leave the ball")
                    print(f"[INFO] Switching to heuristic phase: {phase} at step {step_idx}")
            elif phase == "move_above_pkick":
                p_kick_target = targets.p_kick.copy()
                p_kick_target[2] = float(ARM_LEAVE_BALL_Z)
                actions = torch.as_tensor(
                    _heuristic_move_action(
                        env.unwrapped,
                        p_kick_target,
                        gain=6.0,
                        limit=HEURISTIC_LATERAL_SPEED,
                        grip=-1.0,
                    ),
                    dtype=torch.float32,
                    device=env.unwrapped.device,
                ).unsqueeze(0)
                ee_pos = env.unwrapped.scene["ee_frame"].data.target_pos_w[0, 0, :3].detach().cpu().numpy()
                if (
                    np.linalg.norm(ee_pos[:2] - targets.p_kick[:2]) < float(ARM_PKICK_XY_TOL)
                    and abs(float(ee_pos[2]) - float(ARM_LEAVE_BALL_Z)) < float(ARM_Z_AXIS_TOL)
                ):
                    phase = "lower_to_field"
                    phase_steps = 0
                    print(f"[INFO] Switching to heuristic phase: {phase} at step {step_idx}")
            elif phase == "lower_to_field":
                ee_z = float(env.unwrapped.scene["ee_frame"].data.target_pos_w[0, 0, 2].detach().cpu().item())
                actions = torch.as_tensor(
                    _vertical_action(float(ARM_FIELD_TOUCH_Z) - ee_z, grip=-1.0),
                    dtype=torch.float32,
                    device=env.unwrapped.device,
                ).unsqueeze(0)
                if ee_z <= float(ARM_FIELD_TOUCH_Z) + float(ARM_Z_AXIS_TOL):
                    phase = "kick"
                    phase_steps = 0
                    print("Arm is ready to kick")
                    print(f"[INFO] Switching to heuristic phase: {phase} at step {step_idx}")
            else:
                actions = torch.as_tensor(
                    _heuristic_kick_action(targets.goal_dir, speed=1.0),
                    dtype=torch.float32,
                    device=env.unwrapped.device,
                ).unsqueeze(0)
                phase_steps += 1
                if phase_steps >= kick_steps:
                    actions = torch.zeros((1, 4), dtype=torch.float32, device=env.unwrapped.device)

            obs, _, _, _ = env.step(actions)
            if DEBUG_PLAY and (step_idx % int(DEBUG_PRINT_EVERY_STEPS) == 0):
                debug_ball_pos, _ = _get_ball_state(env.unwrapped)
                prev_debug_ee_pos_w, init_ee_quat_w = _debug_play_state(
                    env.unwrapped,
                    step_idx=step_idx,
                    phase=phase,
                    action_4d=actions,
                    init_ee_quat_w=init_ee_quat_w,
                    prev_ee_pos_w=prev_debug_ee_pos_w,
                    dt=dt,
                    ball_pos=debug_ball_pos,
                    ball_to_pre_goal_xy=ball_to_pre_goal_xy,
                )
        update_soccer_target_markers(env.unwrapped)
        video_frames.append(_frame_from_render(env.unwrapped))

        if (step_idx + 1) % 100 == 0 or step_idx + 1 == video_steps:
            print(f"[INFO] Recorded {step_idx + 1}/{video_steps} frames")

    _write_mp4(video_frames, video_path, fps=video_fps)
    print(
        f"[INFO] Saved play video to: {video_path} "
        f"({len(video_frames)} frames, {video_fps:.2f} fps, {len(video_frames) / video_fps:.2f} s)"
    )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
