#!/usr/bin/env bash
# Cycle collect_dataset.py through all 10 scene presets, writing one LeRobot
# shard per preset under ${DATASET_ROOT}/preset<NN>_<name>_<timestamp>/. The
# shards share the same schema (see collect_dataset._build_features) so VLA
# training can load them as a concatenated dataset.
#
# Usage:
#   CHECKPOINT_PATH=logs/rsl_rl/openarm_lift/<run>/model_3999.pt \
#     bash openarm_vla/Data_Collector/run_collect_dataset.sh
#
# Optional overrides (any of them):
#   CHECKPOINT_PATH=... NUM_SUCCESS=20 bash openarm_vla/Data_Collector/run_collect_dataset.sh
#   DATASET_ROOT=/mnt/big/.../openarm_lift_vla_v0 \
#     DATASET_REPO_ID=openarm/lift_cube_expert_v0 \
#     CHECKPOINT_PATH=logs/rsl_rl/openarm_lift/<run>/model_3999.pt \
#     NUM_SUCCESS=20 \
#     bash openarm_vla/Data_Collector/run_collect_dataset.sh
#
# Requires: conda env ``env_isaaclab``.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
COLLECT_PY="${SCRIPT_DIR}/collect_dataset.py"

DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/openarm_vla/Datasets/lift_cube_expert_v0}"
DATASET_REPO_ID="${DATASET_REPO_ID:-openarm/lift_cube_expert_v0}"
NUM_SUCCESS="${NUM_SUCCESS:-20}"
PRESET_START="${PRESET_START:-0}"
PRESET_END="${PRESET_END:-9}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"

mkdir -p "${DATASET_ROOT}"

if [[ ! -f "${COLLECT_PY}" ]]; then
  echo "ERROR: missing ${COLLECT_PY}" >&2
  exit 1
fi
if [[ -z "${CHECKPOINT_PATH}" ]]; then
  echo "ERROR: set CHECKPOINT_PATH=/path/to/model.pt before running this script" >&2
  exit 1
fi

# shellcheck disable=SC1091
if [[ -f "${HOME}/miniforge3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/miniforge3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/workspace/miniforge3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/workspace/miniforge3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/mambaforge/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/mambaforge/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
else
  echo "ERROR: conda.sh not found (tried ~/miniforge3, ~/workspace/miniforge3, …)" >&2
  exit 1
fi

conda activate env_isaaclab

cd "${REPO_ROOT}"

echo "[INFO] Repo:           ${REPO_ROOT}"
echo "[INFO] Dataset root:   ${DATASET_ROOT}"
echo "[INFO] Repo id:        ${DATASET_REPO_ID}"
echo "[INFO] Checkpoint:     ${CHECKPOINT_PATH}"
echo "[INFO] Per-preset eps: ${NUM_SUCCESS}"
echo "[INFO] Presets:        ${PRESET_START}..${PRESET_END}  (serial, one Isaac process at a time)"

for i in $(seq "${PRESET_START}" "${PRESET_END}"); do
  echo ""
  echo "========================================"
  echo "[INFO] Collecting preset ${i} / ${PRESET_END}"
  echo "========================================"
  python "${COLLECT_PY}" \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --dataset_root "${DATASET_ROOT}" \
    --dataset_repo_id "${DATASET_REPO_ID}" \
    --num_success "${NUM_SUCCESS}" \
    --scene_preset_id "${i}"
  echo "[INFO] Finished preset ${i}"
done

echo ""
echo "[INFO] All presets done. Shards under: ${DATASET_ROOT}"
echo "[INFO] Tip: load multiple shards in VLA training via lerobot's"
echo "        MultiLeRobotDataset or a list of LeRobotDataset(roots=...)."
