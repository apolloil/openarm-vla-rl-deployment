"""Play (evaluate) a trained PPO checkpoint for the fixed-orientation EE policy.

Headless server workflow: always records an MP4. Edit the configuration block below
(no command-line arguments).

Loads a checkpoint trained by ``train_lift.py``, runs the policy in the Play
variant of the VLA environment, exports JIT/ONNX next to the checkpoint, and
writes a timestamped video under ``<checkpoint_dir>/videos/play/``.

    conda activate env_isaaclab
    cd /home/lcw/workspace/openarm
    python openarm_vla/RL/play_lift.py
"""

# =============================================================================
# Play configuration — edit here only (no CLI).
# =============================================================================
CHECKPOINT_PATH = "/home/lcw/workspace/openarm/logs/rsl_rl/openarm_lift/2026-04-21_16-57-06/model_3999.pt"
VIDEO_SECONDS = 30.0

PLAY_TASK = "Isaac-VLA-Lift-Cube-OpenArm-Play-v0"
RSL_RL_AGENT_ENTRY = "rsl_rl_cfg_entry_point"
# Mixed into the env seed as (PLAY_SEED + time_ns) % 2**31 so every run differs (cube / goal layout).
# Use None → equivalent to base 0; use an int to add a stable offset band (still unique per run).
PLAY_SEED: int | None = None
# None → keep Hydra/env defaults for sim and agent device.
PLAY_DEVICE: str | None = None
# False → headless + off-screen camera (server). True → local GUI viewport.
PLAY_GUI = False
# Visual presets for cube / table / floor (see ``play_scene_presets.py``). Valid: 0 … 9.
# Batch scripts may override via env ``OPENARM_PLAY_SCENE_PRESET``.
PLAY_SCENE_PRESET_ID = 1
# If set (e.g. by batch script), MP4s go here instead of ``<run>/videos/play/``.
# Env ``OPENARM_PLAY_VIDEO_DIR`` overrides this when non-empty.
PLAY_VIDEO_OUTPUT_DIR: str | None = None

OPENARM_REPO_ROOT = "/home/lcw/workspace/openarm"
# =============================================================================

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

os.environ.setdefault("CARB_LOGGING_LEVEL", "error")

# Hydra must not see stray argv; keep only the script name (same idea as train_lift.py).
sys.argv = [sys.argv[0]]

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

# Match train_lift.py AppLauncher bootstrap: same parser shape (RSL-RL + AppLauncher),
# then apply globals. A minimal AppLauncher-only Namespace misses fields and has led to
# headless viewport / render pipeline differences (white frames) on some setups.
_parser = argparse.ArgumentParser(description="Play PPO-EE (config via globals at top of file).")
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

# --- Overrides from globals (no user CLI) ---------------------------------
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

from lift_ee_action_wrapper import EulerEEActionWrapper
from play_scene_presets import NUM_PLAY_SCENE_PRESETS, apply_play_scene_materials
from policy_action_vecenv_wrapper import PolicyActionVecEnvWrapper


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
    """Create a timestamped MP4 path; optional flat output dir + preset tag for batch runs."""
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


_SCENE_SEED_MOD = 2**31


@hydra_task_config(PLAY_TASK, RSL_RL_AGENT_ENTRY)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with PPO-EE agent; always record video to disk."""
    _time_part = int(time.time_ns()) % _SCENE_SEED_MOD
    _base = 0 if PLAY_SEED is None else int(PLAY_SEED) % _SCENE_SEED_MOD
    scene_seed = (_base + _time_part) % _SCENE_SEED_MOD
    print(
        f"[INFO] scene_seed={scene_seed} = (PLAY_SEED_base={_base} + time_ns%{_SCENE_SEED_MOD}={_time_part}) "
        f"% {_SCENE_SEED_MOD}  → cube / goal / env RNG vary each run"
    )

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

    _preset = apply_play_scene_materials(env_cfg, preset_id)
    print(
        f"[INFO] Play scene preset {_preset.id} ({_preset.name}); "
        f"valid ids: 0–{NUM_PLAY_SCENE_PRESETS - 1} (see play_scene_presets.py)."
    )

    env = gym.make(PLAY_TASK, cfg=env_cfg, render_mode="rgb_array")

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = EulerEEActionWrapper(env)

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
    print(
        f"[INFO] Video target: {video_steps} steps at {video_fps:.2f} fps "
        f"(expected duration {video_steps / video_fps:.2f} s)"
    )
    for step_idx in range(video_steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
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
