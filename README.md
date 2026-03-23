# OpenArm

Extension of [openarm_isaac_lab](https://github.com/enactic/openarm_isaac_lab) with a VLA (Vision-Language-Action) interface and DiffIK-based environments.

**中文文档：** [README_zh.md](README_zh.md)

## Structure

```
openarm/
├── openarm_isaac_lab/   # base Isaac Lab extension (upstream fork)
└── openarm_vla/         # VLA interface & DiffIK environments
    ├── EE_API_Test/     # smoke test script + API notes (README)
    ├── docs/
    └── source/openarm_vla/
```

## Installation

Run inside the Isaac Lab conda env or Docker image:

```bash
pip install -e openarm_isaac_lab/source/openarm --no-build-isolation
pip install -e openarm_vla/source/openarm_vla --no-build-isolation
```

See `openarm_vla/docs/conda_isaac_lab_setup.md` for full environment setup.

## Registered Gym Environments


| Gym ID                                  | Task               | Mode      |
| --------------------------------------- | ------------------ | --------- |
| `Isaac-VLA-Lift-Cube-OpenArm-Play-v0`   | Lift cube          | unimanual |
| `Isaac-VLA-Reach-OpenArm-Play-v0`       | Reach              | unimanual |
| `Isaac-VLA-Open-Drawer-OpenArm-Play-v0` | Open drawer        | unimanual |
| `Isaac-VLA-Reach-OpenArm-Bi-Play-v0`    | Reach              | bimanual  |
| `Isaac-VLA-OpenArm-Bi-Play-v0`          | Free (table scene) | bimanual  |


### Task names


| `task`        | Mode      | Action dims |
| ------------- | --------- | ----------- |
| `"lift"`      | unimanual | 7           |
| `"reach_uni"` | unimanual | 7           |
| `"cabinet"`   | unimanual | 7           |
| `"reach_bi"`  | bimanual  | 14          |


**Unimanual (7-dim):** `[Δx, Δy, Δz, Δroll, Δpitch, Δyaw, grip]` — deltas in robot base frame (m / rad).  
**Bimanual (14-dim):** `[left_arm(7), right_arm(7)]` — same layout per arm.  
**Gripper:** > 0.5 → open, ≤ 0.5 → close.

## Smoke Test

Records 7 MP4s (one per EE DoF) using the Lift-Cube scene:

```bash
python openarm_vla/EE_API_Test/test_ee_dims_video.py          # headless
python openarm_vla/EE_API_Test/test_ee_dims_video.py --gui    # with viewport
```

Output: `openarm_vla/EE_API_Test/videos/`. API details: [openarm_vla/EE_API_Test/README.md](openarm_vla/EE_API_Test/README.md).