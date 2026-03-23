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

"""Headless EE smoke test: one MP4 per left-arm scalar action dim (6 + gripper).

Uses the unified VLA wrapper with the Lift-Cube task scene so videos have a
visible robot, table, and coloured cube.

Defaults to **headless** Kit (no window); pass **--gui** for a local viewport.
Must run under Isaac Sim / Isaac Lab (AppLauncher first). See README.md next to this script.
"""

from __future__ import annotations

import argparse
import faulthandler
import os
import sys
import threading
import traceback
from pathlib import Path

# Native crashes (e.g. inside Kit during first render) leave no Python traceback unless enabled.
faulthandler.enable(all_threads=True)

# Fewer Kit / carb warnings on stderr (Python prints stay on real stdout/stderr).
os.environ.setdefault("CARB_LOGGING_LEVEL", "error")

# AppLauncher must run before most isaaclab / torch imports
from isaaclab.app import AppLauncher


def _stderr_line_is_carb_launcher_nvidia_smi_noise(line: str) -> bool:
    """True for known-benign Carb spam: broken nvidia-smi child launch (Exec format error).

    Kit still discovers GPUs through other paths; simulation usually runs fine.
    """
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
    """Interpose OS stderr (fd 2): Carb logs bypass Python's sys.stderr."""

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
    description="Record 7 MP4s testing each EE control dimension (Lift-Cube VLA env)."
)
parser.add_argument("--steps",     type=int,   default=150, help="Steps per dimension test.")
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
    help="VLA raw magnitude for Δroll/Δpitch/Δyaw only (larger than --amplitude so wrist rotation reads on video).",
)
parser.add_argument(
    "--grip-settle-steps",
    type=int,
    default=24,
    help=(
        "Env steps after each reset with grip-closed command so fingers close from default open pose "
        "(init 0.044 m, slow vel limit); increase if videos still look open at the start."
    ),
)
parser.add_argument("--seed",      type=int,   default=0,   help="Environment seed.")
parser.add_argument(
    "--output-dir",
    type=str,
    default=None,
    help="Directory for MP4 files (default: videos/ next to this script).",
)
parser.add_argument("--fps", type=float, default=30.0, help="Video frame rate.")
parser.add_argument(
    "--gui",
    action="store_true",
    help="Launch Kit with a window (default for this script is headless / off-screen).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.gui and args_cli.headless:
    parser.error("Use either --gui or --headless, not both.")

# Default headless so `isaaclab.python.headless.rendering.kit` is used with cameras.
# AppLauncher falls back to $HEADLESS when headless=False; force 0 so --gui is not overridden.
if args_cli.gui:
    os.environ["HEADLESS"] = "0"
    args_cli.headless = False
else:
    args_cli.headless = True

# Cameras required for rgb_array rendering
args_cli.enable_cameras = True

# Omniverse Kit: raise log threshold (warnings still possible from extensions).
for _kit_flag in ("--/log/level=error",):
    if _kit_flag not in sys.argv:
        sys.argv.append(_kit_flag)

_FilterStderrFd2.install()
app_launcher   = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- Imports after AppLauncher ---------------------------------------------------

import numpy as np
import torch

try:
    from openarm_vla.tasks.utils.vla_env_wrapper import make_vla_env
except ModuleNotFoundError as _e:
    print(
        "\n[ERROR] Python package 'openarm_vla' is not installed in this environment.\n"
        "        New / reset Docker containers do not keep pip installs unless the image was rebuilt.\n"
        "        Run once from the repo root, then re-run this script:\n\n"
        "          cd /workspace/openarm_isaac_lab\n"
        "          python -m pip install -e source/openarm --no-build-isolation\n"
        "          cd /workspace/openarm_vla\n"
        "          python -m pip install -e source/openarm_vla --no-build-isolation\n\n",
        file=sys.stderr,
    )
    simulation_app.close()
    raise SystemExit(1) from _e

# --- Constants -------------------------------------------------------------------

# 7-dim VLA unimanual action:
#   [Δx(0), Δy(1), Δz(2), Δroll(3), Δpitch(4), Δyaw(5), grip(6)]
_ACTION_DIM  = 7
_GRIP_IDX    = 6

_DIM_NAMES = (
    "00_dx",
    "01_dy",
    "02_dz",
    "03_droll",
    "04_dpitch",
    "05_dyaw",
    "06_grip",
)

# --- Helpers ---------------------------------------------------------------------


def _frame_from_render(env) -> np.ndarray:
    """Convert env.render() to HxWx3 uint8 RGB numpy."""
    rgb = env.render()
    if rgb is None:
        raise RuntimeError(
            "env.render() returned None; check render_mode='rgb_array' and --enable_cameras."
        )

    if isinstance(rgb, torch.Tensor):
        arr = rgb.detach().cpu().numpy()
    else:
        arr = np.asarray(rgb)

    # Possible shapes: (H,W,3), (1,H,W,3), (N,H,W,3)
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


def _baseline_action() -> np.ndarray:
    """7-dim baseline: no arm motion, gripper closed (VLA ≤0.5 → Isaac −1 → close target)."""
    a = np.zeros(_ACTION_DIM, dtype=np.float32)
    a[_GRIP_IDX] = 0.0  # closed (see gripper_to_binary)
    return a


def _settle_grip_closed(env, n_steps: int, *, seed: int) -> None:
    """Drive fingers toward closed after reset (default articulation pose is open at 0.044 m)."""
    for _ in range(n_steps):
        terminated, truncated = _step_env(env, _baseline_action())
        if terminated or truncated:
            env.reset(seed=seed)


def _step_env(env, action) -> tuple[bool, bool]:
    """Return (terminated, truncated)."""
    out = env.step(action)
    if len(out) == 5:
        _, _, terminated, truncated, _ = out
    else:
        _, _, terminated, _ = out
        truncated = False
    return bool(terminated), bool(truncated)


def _run_one_dim(
    env,
    dim_index: int,
    *,
    steps: int,
    amplitude: float,
    amplitude_rot: float,
    grip_settle_steps: int,
    seed: int,
) -> list[np.ndarray]:
    """Run one test: perturb a single action dimension for `steps` steps."""
    print(f"[INFO] dim {dim_index}: reset …", flush=True)
    env.reset(seed=seed)
    # Robot default finger pose is open (0.044 m); finger joints are slow (vel limit 0.2 m/s).
    # A single env step is only `decimation` physics substeps — not enough time to close.
    print(
        f"[INFO] dim {dim_index}: grip settle ({grip_settle_steps}× closed cmd) …",
        flush=True,
    )
    _settle_grip_closed(env, grip_settle_steps, seed=seed)
    # One more baseline step stabilizes viewport / render before first rgb_array.
    print(f"[INFO] dim {dim_index}: render warmup (baseline) …", flush=True)
    terminated, truncated = _step_env(env, _baseline_action())
    if terminated or truncated:
        env.reset(seed=seed)
        _settle_grip_closed(env, grip_settle_steps, seed=seed)
    print(f"[INFO] dim {dim_index}: first render …", flush=True)
    frames: list[np.ndarray] = [_frame_from_render(env)]
    print(f"[INFO] dim {dim_index}: first frame OK, shape={frames[0].shape}", flush=True)

    action = _baseline_action()
    if dim_index < 3:
        action[dim_index] = float(amplitude)
    elif dim_index < 6:
        # DiffIK applies the same scale to translation and axis-angle; rotation in rad
        # needs larger raw VLA values than metres for a comparable on-screen effect.
        action[dim_index] = float(amplitude_rot)
    else:
        # gripper test: open (+1 > 0.5 threshold)
        action[_GRIP_IDX] = 1.0

    log_every = max(1, steps // 5)
    for step_i in range(steps):
        terminated, truncated = _step_env(env, action)
        frames.append(_frame_from_render(env))
        if step_i % log_every == 0 or step_i == steps - 1:
            print(f"[INFO] dim {dim_index}: step {step_i + 1}/{steps}, {len(frames)} frames", flush=True)
        if terminated or truncated:
            env.reset(seed=seed)
            _settle_grip_closed(env, grip_settle_steps, seed=seed)

    return frames


# --- Main ------------------------------------------------------------------------


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    out_dir    = Path(args_cli.output_dir) if args_cli.output_dir else script_dir / "videos"
    out_dir    = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output directory (MP4s go here): {out_dir}")

    print("[INFO] Creating Lift-Cube VLA environment …")
    env = make_vla_env(
        "lift",
        num_envs=1,
        seed=args_cli.seed,
        render_mode="rgb_array",
    )

    try:
        for i, name in enumerate(_DIM_NAMES):
            print(f"[INFO] Recording dim {i} ({name}) …", flush=True)
            frames = _run_one_dim(
                env,
                i,
                steps=args_cli.steps,
                amplitude=args_cli.amplitude,
                amplitude_rot=args_cli.amplitude_rot,
                grip_settle_steps=args_cli.grip_settle_steps,
                seed=args_cli.seed,
            )
            out_path = out_dir / f"ee_dim_{name}.mp4"
            print(f"[INFO] dim {i}: encoding {out_path.name} …", flush=True)
            _write_mp4(frames, out_path, args_cli.fps)
            print(f"[INFO] Wrote {out_path}  ({len(frames)} frames)", flush=True)
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
