#!/usr/bin/env bash
# Cycle collect_dataset.py through all 10 scene presets, writing one LeRobot
# shard per preset under ${DATASET_ROOT}/preset<NN>_<name>_<timestamp>/. The
# shards share the same schema (see collect_dataset._build_features) so VLA
# training can load them as a concatenated dataset.
#
# Usage:
#   bash openarm_vla/Data_Collector/run_collect_dataset.sh
#
# Optional overrides (any of them):
#   NUM_SUCCESS=20 bash openarm_vla/Data_Collector/run_collect_dataset.sh
#   DATASET_ROOT=/mnt/big/.../openarm_lift_vla_v0 \
#     DATASET_REPO_ID=openarm/lift_cube_expert_v0 \
#     NUM_SUCCESS=20 \
#     bash openarm_vla/Data_Collector/run_collect_dataset.sh
#
# Requires: conda env ``env_isaaclab``, checkpoint path edited at the top of
# collect_dataset.py (or passed via CHECKPOINT_PATH override is not wired —
# edit the file directly).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
COLLECT_PY="${SCRIPT_DIR}/collect_dataset.py"

DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/openarm_vla/Datasets/lift_cube_expert_v0}"
DATASET_REPO_ID="${DATASET_REPO_ID:-openarm/lift_cube_expert_v0}"
NUM_SUCCESS="${NUM_SUCCESS:-20}"
PRESET_START="${PRESET_START:-0}"
PRESET_END="${PRESET_END:-9}"

mkdir -p "${DATASET_ROOT}"

if [[ ! -f "${COLLECT_PY}" ]]; then
  echo "ERROR: missing ${COLLECT_PY}" >&2
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

export OPENARM_COLLECT_DATASET_ROOT="${DATASET_ROOT}"
export OPENARM_COLLECT_DATASET_REPO_ID="${DATASET_REPO_ID}"
export OPENARM_COLLECT_NUM_SUCCESS="${NUM_SUCCESS}"

echo "[INFO] Repo:           ${REPO_ROOT}"
echo "[INFO] Dataset root:   ${DATASET_ROOT}"
echo "[INFO] Repo id:        ${DATASET_REPO_ID}"
echo "[INFO] Per-preset eps: ${NUM_SUCCESS}"
echo "[INFO] Presets:        ${PRESET_START}..${PRESET_END}  (serial, one Isaac process at a time)"

for i in $(seq "${PRESET_START}" "${PRESET_END}"); do
  echo ""
  echo "========================================"
  echo "[INFO] Collecting preset ${i} / ${PRESET_END}"
  echo "========================================"
  export OPENARM_COLLECT_SCENE_PRESET="${i}"
  python "${COLLECT_PY}"
  echo "[INFO] Finished preset ${i}"
done

echo ""
echo "[INFO] All presets done. Shards under: ${DATASET_ROOT}"
echo "[INFO] Tip: load multiple shards in VLA training via lerobot's"
echo "        MultiLeRobotDataset or a list of LeRobotDataset(roots=...)."
