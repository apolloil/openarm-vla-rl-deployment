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

"""Headless EE smoke test for the **Soccer** scene.

Mirror of ``test_ee_dims_video.py`` (Lift) but bound to the Soccer env.
Per dimension we record:

  * an MP4 to ``soccer_videos/`` showing what the EE perturbation looks like,
  * a ``.log`` file under ``soccer_logs/`` containing one line per env step
    with the world-frame ``EE / Ball / Goal`` positions captured by
    :func:`GetAllObj`. The log is the only thing we write to that file —
    no Carb noise, no ``[INFO]`` messages — so it is easy to grep / plot.

Defaults to the same train env + 4D action wrapper used by PPO.  Use
``--action-interface vla7 --env-variant play`` to test the generic VLA API.
Defaults to **headless** Kit; pass ``--gui`` for an on-screen viewport.
"""

from __future__ import annotations

import argparse
import faulthandler
import os
import sys
import threading
import traceback
from pathlib import Path

faulthandler.enable(all_threads=True)

os.environ.setdefault("CARB_LOGGING_LEVEL", "error")

from isaaclab.app import AppLauncher


def _stderr_line_is_carb_launcher_nvidia_smi_noise(line: str) -> bool:
    if "[carb.launcher.plugin]" not in line:
        return False
    return any(
        m in line
        for m in (
            "nvidia-smi",
            "failed to exec the new image",
            "failed to fork the child process",
            "launch descriptor for",
            "onReadStdout",
            "onReadStderr",
            "readStdoutContext",
            "readStderrContext",
            "interpreter = ",
        )
    )


class _FilterStderrFd2:
    _saved_fd: int | None = None

    @classmethod
    def install(cls) -> None:
        if sys.platform == "win32" or cls._saved_fd is not None:
            return
        cls._saved_fd = os.dup(2)
        read_end, write_end = os.pipe()
        try:
            os.dup2(write_end, 2)
        finally:
            os.close(write_end)

        def _pump() -> None:
            buf = b""
            reader = os.fdopen(read_end, "rb", buffering=0)
            try:
                while True:
                    chunk = reader.read(4096)
                    if not chunk:
                        break
                    buf += chunk
                    while b"\n" in buf:
                        i = buf.index(b"\n")
                        raw_line = buf[: i + 1]
                        buf = buf[i + 1 :]
                        cls._emit(raw_line)
                if buf:
                    cls._emit(buf)
            finally:
                reader.close()

        threading.Thread(target=_pump, name="stderr-carb-filter", daemon=True).start()

    @classmethod
    def _emit(cls, raw: bytes) -> None:
        if cls._saved_fd is None:
            return
        try:
            text = raw.decode("utf-8", errors="replace")
        except Exception:
            text = raw.decode("latin-1", errors="replace")
        if _stderr_line_is_carb_launcher_nvidia_smi_noise(text):
            return
        try:
            os.write(cls._saved_fd, raw)
        except OSError:
            pass


# --- CLI -------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Record MP4s + logs testing each EE control dimension on the Soccer scene."
)
parser.add_argument("--steps", type=int, default=150, help="Steps per dimension test.")
parser.add_argument(
    "--amplitude",
    type=float,
    default=1.0,
    help="VLA raw magnitude for Δx/Δy/Δz only (after DiffIK scale≈0.01: ~1 cm/step at 1.0).",
)
parser.add_argument(
    "--amplitude-rot",
    type=float,
    default=6.0,
    help="VLA raw magnitude for Δroll/Δpitch/Δyaw only.",
)
parser.add_argument(
    "--grip-settle-steps",
    type=int,
    default=24,
    help="Env steps after each reset with grip-closed command (fingers slow to close).",
)
parser.add_argument("--seed", type=int, default=0, help="Environment seed.")
parser.add_argument(
    "--video-dir",
    type=str,
    default=None,
    help="Directory for MP4 files (default: soccer_videos/ next to this script).",
)
parser.add_argument(
    "--log-dir",
    type=str,
    default=None,
    help="Directory for .log files (default: soccer_logs/ next to this script).",
)
parser.add_argument("--fps", type=float, default=30.0, help="Video frame rate.")
parser.add_argument(
    "--env-variant",
    choices=("train", "play"),
    default="train",
    help="Use the PPO training env by default; play is available for VLA API checks.",
)
parser.add_argument(
    "--action-interface",
    choices=("ppo4", "vla7"),
    default="ppo4",
    help="ppo4 matches train_soccer.py ([dx,dy,dz,grip]); vla7 tests raw VLA API.",
)
parser.add_argument(
    "--gui",
    action="store_true",
    help="Launch Kit with a window (default for this script is headless / off-screen).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.gui and args_cli.headless:
    parser.error("Use either --gui or --headless, not both.")

if args_cli.gui:
    os.environ["HEADLESS"] = "0"
    args_cli.headless = False
else:
    args_cli.headless = True

args_cli.enable_cameras = True

for _kit_flag in ("--/log/level=error",):
    if _kit_flag not in sys.argv:
        sys.argv.append(_kit_flag)

_FilterStderrFd2.install()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- Imports after AppLauncher ---------------------------------------------------

import numpy as np
import torch
import gymnasium as gym

try:
    from openarm_vla.tasks.utils.vla_env_wrapper import make_vla_env
except ModuleNotFoundError as _e:
    print(
        "\n[ERROR] Python package 'openarm_vla' is not installed.\n",
        file=sys.stderr,
    )
    simulation_app.close()
    raise SystemExit(1) from _e

import openarm.tasks  # noqa: F401
import openarm_vla.tasks  # noqa: F401

_RL_DIR = Path(__file__).resolve().parents[1] / "RL"
if str(_RL_DIR) not in sys.path:
    sys.path.insert(0, str(_RL_DIR))

from soccer_ee_action_wrapper import Push3DEEActionWrapper  # noqa: E402
from soccer_targets import compute_soccer_targets, update_soccer_target_markers  # noqa: E402

# Soccer goal local positions (relative to env_origin).  The visual USD origin
# is the goal bottom-centre; the reward target is the desired ball-centre point
# inside the goal, matching MetaWorld's ball/goal z convention.
from openarm.soccer import SOCCER_GOAL_POS, SOCCER_GOAL_TARGET_POS

# --- Constants -------------------------------------------------------------------

_PPO_DIM_NAMES = ("00_dx", "01_dy", "02_dz", "03_grip")
_VLA_DIM_NAMES = ("00_dx", "01_dy", "02_dz", "03_droll", "04_dpitch", "05_dyaw", "06_grip")


# --- Object position logging -----------------------------------------------------


def GetAllObj(env, log_file, step_idx: int) -> None:
    """Read EE / Ball / Goal world positions for env 0 and append one line.

    * EE  – ``ee_frame`` FrameTransformer's first target (``ee_tcp``).
    * Ball – ``soccer_ball`` RigidObject root pose.
    * Goal – ``soccer_goal`` is an :class:`AssetBaseCfg` (no rigid body), so
      we recover its world position from the constant local offset
      ``SOCCER_GOAL_POS`` plus ``scene.env_origins[0]``. The local origin of
      the goal USD coincides with the bottom-center of the goal mouth, so
      this is exactly what we want for reward shaping later.
    """
    base_env = env.unwrapped
    scene = base_env.scene

    ee_pos = scene["ee_frame"].data.target_pos_w[0, 0, :3].detach().cpu()
    ball_pos = scene["soccer_ball"].data.root_pos_w[0, :3].detach().cpu()

    env_origin_0 = scene.env_origins[0].detach().cpu()
    goal_bottom_pos = torch.tensor(SOCCER_GOAL_POS, dtype=ball_pos.dtype) + env_origin_0
    goal_target_pos = torch.tensor(SOCCER_GOAL_TARGET_POS, dtype=ball_pos.dtype) + env_origin_0
    if hasattr(base_env, "_soccer_goal_offsets"):
        goal_offset = base_env._soccer_goal_offsets[0, :3].detach().cpu()
        goal_bottom_pos = goal_bottom_pos + goal_offset
        goal_target_pos = goal_target_pos + goal_offset
    targets = compute_soccer_targets(base_env)

    line = (
        f"step={step_idx:04d}  "
        f"EE=({ee_pos[0].item():+.4f},{ee_pos[1].item():+.4f},{ee_pos[2].item():+.4f})  "
        f"Ball=({ball_pos[0].item():+.4f},{ball_pos[1].item():+.4f},{ball_pos[2].item():+.4f})  "
        f"GoalBottom=({goal_bottom_pos[0].item():+.4f},{goal_bottom_pos[1].item():+.4f},{goal_bottom_pos[2].item():+.4f})  "
        f"GoalTarget=({goal_target_pos[0].item():+.4f},{goal_target_pos[1].item():+.4f},{goal_target_pos[2].item():+.4f})  "
        f"PreGoal=({targets.pre_goal[0]:+.4f},{targets.pre_goal[1]:+.4f},{targets.pre_goal[2]:+.4f})  "
        f"PKick=({targets.p_kick[0]:+.4f},{targets.p_kick[1]:+.4f},{targets.p_kick[2]:+.4f})\n"
    )
    log_file.write(line)
    log_file.flush()


# --- Helpers ---------------------------------------------------------------------


def _frame_from_render(env) -> np.ndarray:
    rgb = env.render()
    if rgb is None:
        raise RuntimeError(
            "env.render() returned None; check render_mode='rgb_array' and --enable_cameras."
        )

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
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio

        imageio.mimsave(str(path), frames, fps=fps, codec="libx264")
    except Exception:
        import cv2

        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, float(fps), (w, h))
        if not writer.isOpened():
            raise RuntimeError(f"cv2.VideoWriter could not open {path}")
        for f in frames:
            bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
            writer.write(bgr)
        writer.release()


def _baseline_action(action_dim: int) -> np.ndarray:
    a = np.zeros(action_dim, dtype=np.float32)
    # PPO wrapper: grip logit <= 0 closes.  VLA wrapper: grip <= 0.5 closes.
    return a


def _step_env(env, action) -> tuple[bool, bool]:
    out = env.step(action)
    if len(out) == 5:
        _, _, terminated, truncated, _ = out
    else:
        _, _, terminated, _ = out
        truncated = False
    return bool(terminated), bool(truncated)


def _settle_grip_closed(env, n_steps: int, *, seed: int, action_dim: int) -> None:
    for _ in range(n_steps):
        terminated, truncated = _step_env(env, _baseline_action(action_dim))
        if terminated or truncated:
            env.reset(seed=seed)


def _run_one_dim(
    env,
    dim_index: int,
    *,
    steps: int,
    amplitude: float,
    amplitude_rot: float,
    grip_settle_steps: int,
    seed: int,
    log_file,
    action_dim: int,
) -> list[np.ndarray]:
    """Run one test: perturb a single action dimension; write per-step EE/Ball/Goal log."""
    print(f"[INFO] dim {dim_index}: reset …", flush=True)
    env.reset(seed=seed)
    print(
        f"[INFO] dim {dim_index}: grip settle ({grip_settle_steps}× closed cmd) …",
        flush=True,
    )
    _settle_grip_closed(env, grip_settle_steps, seed=seed, action_dim=action_dim)

    print(f"[INFO] dim {dim_index}: render warmup (baseline) …", flush=True)
    terminated, truncated = _step_env(env, _baseline_action(action_dim))
    update_soccer_target_markers(env, prefix="/World/EETest")
    if terminated or truncated:
        env.reset(seed=seed)
        _settle_grip_closed(env, grip_settle_steps, seed=seed, action_dim=action_dim)
        update_soccer_target_markers(env, prefix="/World/EETest")

    print(f"[INFO] dim {dim_index}: first render …", flush=True)
    frames: list[np.ndarray] = [_frame_from_render(env)]
    print(f"[INFO] dim {dim_index}: first frame OK, shape={frames[0].shape}", flush=True)
    GetAllObj(env, log_file, step_idx=0)

    action = _baseline_action(action_dim)
    if dim_index < 3:
        action[dim_index] = float(amplitude)
    elif action_dim == 7 and dim_index < 6:
        action[dim_index] = float(amplitude_rot)
    else:
        action[-1] = 1.0  # gripper open

    log_every = max(1, steps // 5)
    for step_i in range(steps):
        terminated, truncated = _step_env(env, action)
        update_soccer_target_markers(env, prefix="/World/EETest")
        frames.append(_frame_from_render(env))
        GetAllObj(env, log_file, step_idx=step_i + 1)
        if step_i % log_every == 0 or step_i == steps - 1:
            print(
                f"[INFO] dim {dim_index}: step {step_i + 1}/{steps}, {len(frames)} frames",
                flush=True,
            )
        if terminated or truncated:
            env.reset(seed=seed)
            _settle_grip_closed(env, grip_settle_steps, seed=seed, action_dim=action_dim)
            update_soccer_target_markers(env, prefix="/World/EETest")

    return frames


def _build_train_ppo_env(*, num_envs: int, seed: int, render_mode: str | None):
    gym_id = "Isaac-VLA-Soccer-OpenArm-v0"
    cfg_entry = gym.spec(gym_id).kwargs["env_cfg_entry_point"]
    if isinstance(cfg_entry, str) and ":" in cfg_entry:
        import importlib

        mod_name, cls_name = cfg_entry.rsplit(":", 1)
        env_cfg = getattr(importlib.import_module(mod_name), cls_name)()
    else:
        env_cfg = cfg_entry()

    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = seed
    raw_env = gym.make(gym_id, cfg=env_cfg, render_mode=render_mode)
    return Push3DEEActionWrapper(raw_env)


# --- Main ------------------------------------------------------------------------


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_suffix = f"{args_cli.env_variant}_{args_cli.action_interface}"
    video_dir = (
        Path(args_cli.video_dir).resolve()
        if args_cli.video_dir
        else (script_dir / f"soccer_videos_{default_suffix}").resolve()
    )
    log_dir = (
        Path(args_cli.log_dir).resolve()
        if args_cli.log_dir
        else (script_dir / f"soccer_logs_{default_suffix}").resolve()
    )
    video_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Video output: {video_dir}")
    print(f"[INFO] Log   output: {log_dir}")

    if args_cli.action_interface == "ppo4":
        dim_names = _PPO_DIM_NAMES
        action_dim = 4
        print("[INFO] Creating Soccer PPO train environment (4D action interface) …")
        env = _build_train_ppo_env(num_envs=1, seed=args_cli.seed, render_mode="rgb_array")
    else:
        dim_names = _VLA_DIM_NAMES
        action_dim = 7
        if args_cli.env_variant != "play":
            print("[WARN] vla7 interface currently uses make_vla_env(), which is registered to the Play env.")
        print("[INFO] Creating Soccer VLA environment (7D action interface) …")
        env = make_vla_env(
            "soccer",
            num_envs=1,
            seed=args_cli.seed,
            render_mode="rgb_array",
        )

    try:
        for i, name in enumerate(dim_names):
            video_path = video_dir / f"soccer_ee_dim_{name}.mp4"
            log_path = log_dir / f"soccer_dim_{name}.log"
            print(f"[INFO] Recording dim {i} ({name}) → {video_path.name} / {log_path.name}", flush=True)
            with open(log_path, "w") as log_fp:
                frames = _run_one_dim(
                    env,
                    i,
                    steps=args_cli.steps,
                    amplitude=args_cli.amplitude,
                    amplitude_rot=args_cli.amplitude_rot,
                    grip_settle_steps=args_cli.grip_settle_steps,
                    seed=args_cli.seed,
                    log_file=log_fp,
                    action_dim=action_dim,
                )
            print(f"[INFO] dim {i}: encoding {video_path.name} …", flush=True)
            _write_mp4(frames, video_path, args_cli.fps)
            print(
                f"[INFO] Wrote {video_path}  ({len(frames)} frames) + {log_path}",
                flush=True,
            )
        print("[INFO] All dimensions finished successfully.", flush=True)
    except Exception:
        print("[ERROR] Recording or encode failed (traceback on stderr).", flush=True)
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        raise
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
