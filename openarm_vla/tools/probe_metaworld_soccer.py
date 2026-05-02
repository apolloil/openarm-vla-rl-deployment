"""Probe MetaWorld SawyerSoccerV2 to collect numerical baselines.

Run with:
    conda activate baku
    cd /home/lcw/workspace/openarm
    python openarm_vla/tools/probe_metaworld_soccer.py

Outputs to openarm_vla/tools/metaworld_baseline/metaworld_soccer_baseline.log
"""

import os
import sys
import numpy as np

# -- ensure the baku-installed metaworld is importable
try:
    import metaworld
except ImportError:
    sys.exit("ERROR: metaworld not found. Are you in the `baku` conda env?")

import metaworld as mw
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_soccer_v2 import SawyerSoccerEnvV2
from metaworld.policies.sawyer_soccer_v2_policy import SawyerSoccerV2Policy

OUT_DIR = os.path.join(os.path.dirname(__file__), "metaworld_baseline")
os.makedirs(OUT_DIR, exist_ok=True)
LOG_PATH = os.path.join(OUT_DIR, "metaworld_soccer_baseline.log")

# ──────────────────────────────────────────────────────────────────────────────
# 1. Static environment info — use ML1 to get tasks
# ──────────────────────────────────────────────────────────────────────────────
ml1 = mw.ML1("soccer-v2")
env_cls = list(ml1.train_classes.values())[0]
env = env_cls()
task = ml1.train_tasks[0]
env.set_task(task)
obs, _ = env.reset()

lines = []
lines.append("=" * 70)
lines.append("MetaWorld SawyerSoccerV2 — environment constants")
lines.append("=" * 70)
lines.append(f"observation_space : {env.observation_space}")
lines.append(f"action_space      : {env.action_space}")
lines.append(f"OBJ_RADIUS        : {env.OBJ_RADIUS}")
lines.append(f"TARGET_RADIUS     : {env.TARGET_RADIUS}")
lines.append(f"init_config       : {env.init_config}")
lines.append(f"goal (default)    : {env.goal}")
lines.append(f"obj_low           : {env._random_reset_space.low}")
lines.append(f"obj_high          : {env._random_reset_space.high}")
lines.append(f"obs[0:3] (hand)   : {obs[:3]}")
lines.append(f"obs[3]   (gripper): {obs[3]}")
lines.append(f"obs[4:7] (ball)   : {obs[4:7]}")
lines.append(f"obs[-3:] (goal)   : {obs[-3:]}")
lines.append(f"obs_dim           : {len(obs)}")
lines.append("")

# ──────────────────────────────────────────────────────────────────────────────
# 2. Run 20 episodes with the scripted policy; collect reward statistics
# ──────────────────────────────────────────────────────────────────────────────
policy = SawyerSoccerV2Policy()

N_EPISODES = 20
MAX_STEPS  = 500

all_ep_returns = []
all_grasp_rewards   = []
all_in_place_rewards= []
all_obj_to_target   = []
all_unscaled        = []
successes = 0

lines.append("=" * 70)
lines.append(f"Policy rollout — {N_EPISODES} episodes x {MAX_STEPS} steps")
lines.append("=" * 70)

for ep in range(N_EPISODES):
    task = ml1.train_tasks[ep % len(ml1.train_tasks)]
    env.set_task(task)
    obs, _ = env.reset()

    ep_return = 0.0
    ep_grasp = []
    ep_in_place = []
    ep_obj_to_target = []
    ep_unscaled = []
    ep_success = False

    for step in range(MAX_STEPS):
        action = policy.get_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_return += reward
        ep_grasp.append(info.get("grasp_reward", 0.0))
        ep_in_place.append(info.get("in_place_reward", 0.0))
        ep_obj_to_target.append(info.get("obj_to_target", 0.0))
        ep_unscaled.append(info.get("unscaled_reward", reward))
        if info.get("success", 0):
            ep_success = True
        if terminated or truncated:
            break

    successes += int(ep_success)
    all_ep_returns.append(ep_return)
    all_grasp_rewards.extend(ep_grasp)
    all_in_place_rewards.extend(ep_in_place)
    all_obj_to_target.extend(ep_obj_to_target)
    all_unscaled.extend(ep_unscaled)

    lines.append(
        f"  ep {ep:02d}: return={ep_return:7.2f}  success={int(ep_success)}"
        f"  steps={step+1}"
    )

lines.append("")
lines.append(f"Success rate: {successes}/{N_EPISODES}")
lines.append(f"Mean return : {np.mean(all_ep_returns):.2f}")
lines.append("")

def percentiles(name, arr):
    a = np.array(arr)
    return (
        f"  {name:30s}  "
        f"min={a.min():+.4f}  "
        f"p10={np.percentile(a,10):+.4f}  "
        f"p50={np.percentile(a,50):+.4f}  "
        f"p90={np.percentile(a,90):+.4f}  "
        f"max={a.max():+.4f}"
    )

lines.append("Per-step reward component percentiles (across all episodes):")
lines.append(percentiles("unscaled_reward",    all_unscaled))
lines.append(percentiles("grasp_reward",        all_grasp_rewards))
lines.append(percentiles("in_place_reward",     all_in_place_rewards))
lines.append(percentiles("obj_to_target (m)",   all_obj_to_target))
lines.append("")

# ──────────────────────────────────────────────────────────────────────────────
# 3. Geometry cross-reference with OpenArm scene
# ──────────────────────────────────────────────────────────────────────────────
lines.append("=" * 70)
lines.append("Geometry cross-reference (MetaWorld ↔ OpenArm)")
lines.append("=" * 70)
lines.append("MetaWorld (y-forward):")
lines.append(f"  ball_init        : {env.init_config['obj_init_pos']}")
lines.append(f"  goal_default     : {env.goal}")
lines.append(f"  ball_z           : {env.init_config['obj_init_pos'][2]:.4f} m")
lines.append(f"  ball-goal dist   : {np.linalg.norm(np.array(env.init_config['obj_init_pos'][:2]) - np.array(env.goal[:2])):.4f} m")
lines.append(f"  OBJ_RADIUS       : {env.OBJ_RADIUS} m")
lines.append(f"  TARGET_RADIUS    : {env.TARGET_RADIUS} m")
lines.append("")
lines.append("OpenArm (x-forward, from soccer_logs):")
lines.append("  ball_init        : (0.376, 0.000, 0.039)")
lines.append("  goal_center      : (0.612, 0.000, 0.004)")
lines.append("  ball-goal dist   : 0.236 m")
lines.append("  SOCCER_BALL_RADIUS: 0.035 m")
lines.append("")
lines.append("Axis mapping: MetaWorld-Y ↔ OpenArm-X (depth), MetaWorld-X ↔ OpenArm-Y (lateral)")
lines.append("Scale ratio (depth): 0.30 / 0.236 ≈ 1.27  (MetaWorld slightly longer)")

output = "\n".join(lines)
print(output)

with open(LOG_PATH, "w") as f:
    f.write(output + "\n")

print(f"\n[INFO] Log written to {LOG_PATH}")
env.close()
